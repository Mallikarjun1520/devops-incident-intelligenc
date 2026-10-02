import json
import re

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import AnalysisResult
from app.services.llm_service import SYSTEM_PROMPT, extract_json_object
from tests.conftest import VALID_ANALYSIS, FakeGroq, build_service

INCIDENT = ("orders-api pods in CrashLoopBackOff on prod EKS. Error: required environment variable "
            "PAYMENT_GATEWAY_KEY is not set. Back-off restarting failed container. kubectl get pods shows 503s.")


def _client(tmp_path, curated, reply, api_key="test-key"):
    llm = FakeGroq(reply) if reply is not None else None
    service = build_service(tmp_path, llm, api_key)
    service.ingest(curated)
    return TestClient(create_app(service)), llm


# -- success path ------------------------------------------------------------------------------

def test_analyze_success(client):
    body = client.post("/api/analyze", json={"text": INCIDENT}).json()
    assert body["success"] is True
    analysis = body["analysis"]
    assert analysis["severity"] == "High" and 0 <= analysis["severity_score"] <= 100
    assert set(AnalysisResult.model_fields) <= set(analysis)
    severity = body["severity_assessment"]
    assert severity["score"] == 46 and severity["level"] == "Moderate"
    assert any("CrashLoopBackOff" in fact for fact in severity["evidence"])
    assert "Not a measure of SLO burn" in severity["basis"]
    ids = [s["incident_id"] for s in body["similar_incidents"]]
    assert ids and ids[0] == "CUR-001"
    assert body["extracted_signals"]["error_types"]


def test_hallucinated_evidence_ids_are_removed_and_warned(client):
    analysis = client.post("/api/analyze", json={"text": INCIDENT}).json()["analysis"]
    assert "INC-DOES-NOT-EXIST" not in analysis["evidence_incidents"]
    assert "CUR-001" in analysis["evidence_incidents"]
    assert any("INC-DOES-NOT-EXIST" in w for w in analysis["warnings"])
    assert analysis["warnings"][0].startswith("Human review required")


def test_prompt_contains_rules_and_only_retrieved_context(tmp_path, curated_incidents):
    client, llm = _client(tmp_path, curated_incidents, json.dumps(VALID_ANALYSIS))
    body = client.post("/api/analyze", json={"text": INCIDENT}).json()
    call = llm.calls[0]
    assert call["model"] == "test-model"
    system_prompt = call["messages"][0]["content"]
    for rule in ("Do not invent facts", "Never recommend destructive", "cannot and will not execute"):
        assert rule in system_prompt
    prompt = call["messages"][1]["content"]
    assert 'id="CUR-001"' in prompt and "PAYMENT_GATEWAY_KEY" in prompt
    # The prompt holds exactly the retrieved incidents (<= top_k), not the whole knowledge base.
    in_prompt = re.findall(r'<historical_incident id="([^"]+)"', prompt)
    assert in_prompt == [s["incident_id"] for s in body["similar_incidents"]]
    assert len(in_prompt) <= 3 < len(curated_incidents)


def test_file_upload_analyze(client):
    response = client.post("/api/analyze", files={"file": ("incident.log", INCIDENT.encode(), "text/plain")})
    assert response.status_code == 200 and response.json()["success"] is True


def test_json_file_upload_analyze(client):
    payload = json.dumps({"alert": "OrdersDown", "message": INCIDENT}).encode()
    assert client.post("/api/analyze", files={"file": ("alert.json", payload)}).json()["success"] is True


def test_openapi_documents_file_upload_fields(client):
    schema = client.get("/openapi.json").json()
    for path in ("/api/analyze", "/api/ingest"):
        operation = schema["paths"][path]["post"]
        request_body = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        if "$ref" in request_body:
            component_name = request_body["$ref"].rsplit("/", 1)[-1]
            request_body = schema["components"]["schemas"][component_name]
        assert "file" in request_body["properties"]
        file_schema = request_body["properties"]["file"]
        schemas = file_schema.get("anyOf", [file_schema])
        assert any(
            item.get("type") == "string"
            and item.get("contentMediaType") == "application/octet-stream"
            for item in schemas
        )


# -- input validation --------------------------------------------------------------------------

def test_empty_incident_rejected(client):
    for payload in ({"text": ""}, {"text": "   \n  "}, {}):
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 400
        assert response.json()["success"] is False and "empty" in response.json()["error"].lower()


def test_malformed_json_body_rejected(client):
    response = client.post("/api/analyze", content="{not json", headers={"content-type": "application/json"})
    assert response.status_code == 400 and "valid JSON" in response.json()["error"]


def test_wrong_text_type_rejected(client):
    assert client.post("/api/analyze", json={"text": 123}).status_code == 400
    assert client.post("/api/analyze", json=["a"]).status_code == 400


def test_unsupported_file_rejected(client):
    response = client.post("/api/analyze", files={"file": ("incident.exe", b"MZ")})
    assert response.status_code == 415 and "Unsupported" in response.json()["error"]


def test_malformed_json_file_rejected(client):
    response = client.post("/api/analyze", files={"file": ("a.json", b'{"broken": ')})
    assert response.status_code == 400 and "Malformed JSON" in response.json()["error"]


def test_oversized_file_rejected(client):
    big = b"x" * (client.app.state.incident_service.settings.max_upload_bytes + 10)
    assert client.post("/api/analyze", files={"file": ("big.log", big)}).status_code == 413


def test_non_utf8_file_rejected(client):
    assert client.post("/api/analyze", files={"file": ("a.txt", b"\xff\xfe\x00bad")}).status_code == 400


# -- LLM failure handling ----------------------------------------------------------------------

def test_missing_api_key(tmp_path, curated_incidents):
    client, _ = _client(tmp_path, curated_incidents, None, api_key="")
    response = client.post("/api/analyze", json={"text": INCIDENT})
    body = response.json()
    assert response.status_code == 503 and body["success"] is False
    assert "GROQ_API_KEY" in body["error"] and body["error_code"] == "llm_not_configured"
    assert body["severity_assessment"]["score"] == 46
    assert body["similar_incidents"]  # evidence is still returned


def test_invalid_api_key(tmp_path, curated_incidents):
    client, _ = _client(tmp_path, curated_incidents, RuntimeError("authentication failed: invalid API key"))
    body = client.post("/api/analyze", json={"text": INCIDENT}).json()
    assert body["success"] is False and body["error_code"] == "invalid_api_key" and body["similar_incidents"]


def test_groq_api_failure(tmp_path, curated_incidents):
    for error, code in [(RuntimeError("connection error"), "llm_unavailable"),
                        (RuntimeError("internal server error"), "llm_unavailable"),
                        (RuntimeError("429 rate limit exceeded"), "rate_limited")]:
        client, _ = _client(tmp_path, curated_incidents, error)
        body = client.post("/api/analyze", json={"text": INCIDENT}).json()
        assert body["success"] is False and body["error_code"] == code


def test_malformed_llm_output_does_not_crash_app(tmp_path, curated_incidents):
    client, _ = _client(tmp_path, curated_incidents, "Sorry, I cannot produce JSON today.")
    response = client.post("/api/analyze", json={"text": INCIDENT})
    assert response.status_code == 502 and response.json()["error_code"] == "malformed_llm_output"
    assert client.get("/health").status_code == 200  # app is still alive


def test_llm_json_in_code_fence_and_sloppy_fields(tmp_path, curated_incidents):
    sloppy = dict(VALID_ANALYSIS, severity="critical", severity_score="150", confidence_score=None,
                  recommended_resolution="single string step")
    client, _ = _client(tmp_path, curated_incidents, "```json\n" + json.dumps(sloppy) + "\n```")
    analysis = client.post("/api/analyze", json={"text": INCIDENT}).json()["analysis"]
    assert analysis["severity"] == "Critical" and analysis["severity_score"] == 100
    assert analysis["confidence_score"] == 0 and analysis["recommended_resolution"] == ["single string step"]


# -- retrieval degradation ---------------------------------------------------------------------

def test_zero_retrieval_results_still_analyses_with_warning(client):
    body = client.post("/api/analyze", json={"text": "quantum banana xylophone zebra"}).json()
    assert body["success"] is True and body["similar_incidents"] == []
    assert any("No sufficiently similar" in w for w in body["analysis"]["warnings"])


def test_empty_knowledge_base_warns(tmp_path):
    service = build_service(tmp_path, FakeGroq(json.dumps(VALID_ANALYSIS)))
    body = TestClient(create_app(service)).post("/api/analyze", json={"text": INCIDENT}).json()
    assert body["success"] is True
    assert any("ingest_incidents.py" in w for w in body["analysis"]["warnings"])


def test_chromadb_failure_degrades_instead_of_crashing(client):
    def broken(*_args, **_kwargs):
        raise RuntimeError("chroma is down")
    client.app.state.incident_service.retriever.count = broken
    body = client.post("/api/analyze", json={"text": INCIDENT}).json()
    assert body["success"] is True and body["similar_incidents"] == []
    assert any("unavailable" in w for w in body["analysis"]["warnings"])


# -- ingest + incident lookup ------------------------------------------------------------------

def test_ingest_and_lookup(empty_client):
    record = {"incident_id": "T-1", "title": "Test incident", "description": "d", "root_cause": "rc",
              "resolution": "restart the thing", "symptoms": ["s1"]}
    response = empty_client.post("/api/ingest", json=[record, {"title": "missing id"}])
    body = response.json()
    assert response.status_code == 200 and body["ingested"] == 1 and body["incident_ids"] == ["T-1"] and body["errors"]
    fetched = empty_client.get("/api/incidents/T-1").json()["incident"]
    assert fetched["resolution"] == ["restart the thing"] and fetched["symptoms"] == ["s1"]


def test_ingest_file_upload_and_bad_input(empty_client):
    good = json.dumps({"incidents": [{"incident_id": "T-2", "title": "x", "description": "d", "root_cause": "r"}]})
    assert empty_client.post("/api/ingest", files={"file": ("i.json", good.encode())}).json()["ingested"] == 1
    assert empty_client.post("/api/ingest", files={"file": ("i.txt", b"x")}).status_code == 415
    assert empty_client.post("/api/ingest", json=[{"title": "no id"}]).status_code == 400
    assert empty_client.post("/api/ingest", content="{bad", headers={"content-type": "application/json"}).status_code == 400


def test_unknown_incident_404(client):
    response = client.get("/api/incidents/NOPE-1")
    assert response.status_code == 404 and response.json()["success"] is False


# -- helpers -----------------------------------------------------------------------------------

def test_extract_json_object_variants():
    assert extract_json_object('prefix {"a": 1} suffix') == {"a": 1}
    assert extract_json_object('```json\n{"a": 2}\n```') == {"a": 2}


def test_system_prompt_schema_matches_model():
    for field in AnalysisResult.model_fields:
        assert f'"{field}"' in SYSTEM_PROMPT
