from app.services.parser import json_to_text, normalize_whitespace, parse_incident, redact_secrets

LOG = """2026-10-02T09:14:01Z ERROR orders-api-6c9d7f8b5-k2m4n Back-off restarting failed container
2026-10-02T09:14:01Z ERROR orders-api-6c9d7f8b5-k2m4n Back-off restarting failed container
orders-api-6c9d7f8b5-k2m4n   0/1   CrashLoopBackOff   6   11m
Exception in thread "main" java.lang.IllegalStateException: PAYMENT_GATEWAY_KEY missing
    at com.acme.Config.load(Config.java:42)
    at com.acme.Main.main(Main.java:10)
Deploying to prod EKS cluster, namespace payments
"""


def test_extracts_signals_and_metadata():
    parsed = parse_incident(LOG)
    assert "CrashLoopBackOff" in parsed.error_types
    assert "Stack Trace" in parsed.error_types
    assert {"Kubernetes", "AWS EKS"} <= set(parsed.affected_components)
    assert parsed.service == "orders-api"
    assert parsed.environment == "Production"
    assert parsed.category == "Kubernetes"


def test_cleaned_text_preserves_traces_and_timestamps():
    parsed = parse_incident(LOG)
    assert "    at com.acme.Config.load(Config.java:42)" in parsed.cleaned_text  # indentation kept
    assert "2026-10-02T09:14:01Z" in parsed.cleaned_text  # timeline kept for the LLM
    assert "PAYMENT_GATEWAY_KEY" in parsed.cleaned_text


def test_retrieval_query_strips_timestamps_and_collapses_repeats():
    query = parse_incident(LOG).retrieval_query
    assert "2026-10-02" not in query
    assert "(repeated x2)" in query
    assert query.startswith("Signals:")


def test_whitespace_normalisation_is_conservative():
    assert normalize_whitespace("a  b   \n\n\n\n  c\t\r\n") == "a  b\n\n  c"


def test_secrets_are_redacted():
    text, count = redact_secrets("password=hunter2 key AKIAABCDEFGHIJKLMNOP Authorization: Bearer abcdefghijklmnop1234")
    assert count == 3
    assert "hunter2" not in text and "AKIA" not in text and "abcdefghijklmnop1234" not in text


def test_json_incident_is_flattened():
    parsed = parse_incident('{"alert": "Redis Down", "labels": {"service": "cart", "env": "prod"}, "message": "connection refused"}')
    assert "alert: Redis Down" in parsed.cleaned_text
    assert parsed.service == "cart" and "Redis" in parsed.affected_components
    assert "Connection Refused" in parsed.error_types


def test_bracketed_log_line_is_not_mistaken_for_json():
    assert "[WARN] disk full" in parse_incident("[WARN] disk full").cleaned_text


def test_environment_ignores_dev_paths():
    assert parse_incident("cannot open /dev/null").environment is None


def test_long_input_is_truncated_in_the_middle():
    parsed = parse_incident("START " + "x " * 20000 + " END", llm_input_chars=1000)
    assert parsed.truncated and "characters omitted" in parsed.cleaned_text
    assert parsed.cleaned_text.startswith("START") and parsed.cleaned_text.endswith("END")


def test_json_to_text_nested():
    assert json_to_text({"a": [1, {"b": 2}]}) == "a:\n  - 1\n  - b: 2"
