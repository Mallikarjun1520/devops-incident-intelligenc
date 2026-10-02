def test_health_reports_components(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["services"]["api"] == "up"
    assert body["services"]["chromadb"] == "up"
    assert body["services"]["llm"] == "configured"
    assert body["indexed_incidents"] > 0


def test_health_does_not_fail_without_llm_key(empty_client):
    response = empty_client.get("/health")
    assert response.status_code == 200
    assert response.json()["services"]["llm"] == "not_configured"


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "DevOps Incident Intelligence" in page.text
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/script.js").status_code == 200
