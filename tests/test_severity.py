from app.services.parser import parse_incident
from app.services.severity_service import assess_severity


def test_no_severity_evidence_does_not_invent_a_score():
    result = assess_severity(parse_incident("Routine deployment completed in development."))

    assert result["score"] is None
    assert result["level"] == "Unknown"
    assert result["evidence"] == []


def test_negated_severity_words_do_not_raise_the_score():
    result = assess_severity(parse_incident("No current outage; no critical incident."))

    assert result["score"] is None
    assert result["level"] == "Unknown"


def test_error_line_count_produces_a_low_evidence_score():
    result = assess_severity(parse_incident("2026-10-02 ERROR: a request failed"))

    assert result["score"] == 6
    assert result["level"] == "Low"
    assert result["evidence"] == ["1 ERROR/FATAL-level log line(s) (+5, capped at +20)"]


def test_production_availability_failure_combines_explicit_facts():
    result = assess_severity(parse_incident(
        "Production orders-api CrashLoopBackOff\n"
        "ERROR pod is restarting\n"
        "ERROR readiness probe failed"
    ))

    assert result["score"] == 56
    assert result["level"] == "High"
    assert len(result["evidence"]) == 3


def test_explicit_sev_one_declaration_is_scored_as_critical():
    result = assess_severity(parse_incident("SEV-1 declared: data loss in production."))

    assert result["score"] == 76
    assert result["level"] == "Critical"
    assert result["evidence"] == [
        "Explicit critical / SEV-1 / P0 / outage declaration (+75)",
    ]


def test_api_assets_are_available_at_relative_live_server_paths(client):
    assert client.get("/style.css").status_code == 200
    assert client.get("/script.js").status_code == 200
    sample = client.get("/sample")
    assert sample.status_code == 200
    assert "CrashLoopBackOff" in sample.text


def test_local_live_server_origin_can_call_the_api(client):
    response = client.options(
        "/api/analyze",
        headers={
            "Origin": "http://127.0.0.1:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5500"
