from app.services.parser import parse_incident
from app.services.retrieval_service import format_historical_context


def test_retrieval_finds_relevant_incident_first(service):
    query = parse_incident("payment-service container OOMKilled exit code 137 memory limit exceeded").retrieval_query
    results = service.retriever.retrieve_similar_incidents(query, top_k=3)
    assert results and results[0].incident_id == "CUR-002"
    top = results[0]
    assert 0 < top.similarity <= 1 and abs(top.similarity + top.distance - 1) < 1e-3
    assert top.title and top.severity and top.root_cause and top.resolution


def test_results_are_sorted_by_similarity(service):
    results = service.retriever.retrieve_similar_incidents("Redis connection refused cache failover", top_k=3)
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_irrelevant_query_returns_zero_results(service):
    assert service.retriever.retrieve_similar_incidents("quantum banana xylophone zebra") == []


def test_empty_collection_returns_no_results(empty_client):
    svc = empty_client.app.state.incident_service
    assert svc.retriever.retrieve_similar_incidents("anything") == []


def test_get_incident_roundtrip(service):
    incident = service.get_incident("CUR-003")
    assert incident.title.startswith("PostgreSQL") and incident.resolution and incident.preventive_actions
    assert service.get_incident("NOPE") is None


def test_context_contains_only_useful_fields(service):
    results = service.retriever.retrieve_similar_incidents("PostgreSQL too many connections pool", top_k=2)
    context = format_historical_context(results)
    assert f'id="{results[0].incident_id}"' in context
    for label in ("Title:", "Severity:", "Symptoms:", "Root cause:", "Resolution:", "Preventive actions:"):
        assert label in context
    assert "Log lines" not in context  # raw documents are not forwarded


def test_empty_context_message():
    assert "No sufficiently similar" in format_historical_context([])
