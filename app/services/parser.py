"""Conservative incident preprocessing and technical-signal extraction.

Two text variants are produced on purpose:

* ``cleaned_text``    - sent to the LLM. Only whitespace is normalised and obvious secrets are
                        redacted. Timestamps, stack traces and indentation are preserved because
                        timelines matter for reasoning.
* ``retrieval_query`` - used for embedding. Leading timestamps are stripped and repeated lines
                        collapsed so they do not dilute similarity, and detected signals are
                        prepended (embedding models truncate long inputs, so signals go first).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.models.schemas import ParsedIncident

MAX_QUERY_CHARS = 1500

# --- Cleaning ---------------------------------------------------------------------------------

_LEADING_TIMESTAMP = re.compile(
    r"^\s*(?:"
    r"[\[(]?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?[\])]?"  # ISO
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"  # syslog
    r"|\[?\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\]?"  # bare time
    r")\s+"
)

_SECRET_PATTERNS = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)(\s*[=:]\s*)(?!\*)[^\s,;\"']+"),
]


def redact_secrets(text: str) -> tuple[str, int]:
    """Mask obvious credentials before the text leaves the machine. Returns (text, count)."""
    count = 0

    def mask(replacement: str):
        def _sub(match: re.Match) -> str:
            nonlocal count
            count += 1
            return replacement.format(*match.groups())
        return _sub

    access_key, bearer, keyed = _SECRET_PATTERNS
    text = access_key.sub(mask("[REDACTED]"), text)
    text = bearer.sub(mask("Bearer [REDACTED]"), text)
    text = keyed.sub(mask("{0}{1}[REDACTED]"), text)
    return text, count


def normalize_whitespace(text: str) -> str:
    """Trim trailing spaces, expand tabs, cap blank runs. Leading indentation is kept
    so stack traces stay readable."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank_run = 0
    for line in lines:
        blank_run = blank_run + 1 if not line else 0
        if blank_run <= 1:
            out.append(line)
    return "\n".join(out).strip()


def _collapse_repeats(lines: list[str]) -> list[str]:
    """Collapse consecutive identical lines: 'X' x3 -> 'X (repeated x3)'."""
    out: list[str] = []
    previous, run = None, 0
    for line in lines + [None]:  # sentinel flushes the last run
        if line == previous:
            run += 1
            continue
        if previous is not None:
            out.append(previous if run == 1 else f"{previous} (repeated x{run})")
        previous, run = line, 1
    return out


def truncate_middle(text: str, limit: int) -> tuple[str, bool]:
    """Keep the head and tail of oversized input; errors usually sit at either end."""
    if len(text) <= limit:
        return text, False
    head, tail = int(limit * 0.6), int(limit * 0.4)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} characters omitted ...]\n{text[-tail:]}", True


def json_to_text(payload: Any, indent: int = 0) -> str:
    """Flatten a JSON incident/alert payload into readable 'key: value' lines."""
    pad = "  " * indent
    if isinstance(payload, dict):
        lines = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(json_to_text(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {value}")
        return "\n".join(lines)
    if isinstance(payload, list):
        return "\n".join(
            f"{pad}- {json_to_text(item, indent + 1).lstrip()}" if isinstance(item, (dict, list))
            else f"{pad}- {item}"
            for item in payload
        )
    return f"{pad}{payload}"


def _maybe_flatten_json(text: str) -> str:
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            return json_to_text(json.loads(stripped))
        except (ValueError, RecursionError):
            pass  # not JSON after all (e.g. a log line starting with '[')
    return text


# --- Signal detection ---------------------------------------------------------------------------

_I = re.IGNORECASE

ERROR_SIGNALS: list[tuple[str, re.Pattern]] = [
    ("CrashLoopBackOff", re.compile(r"crash\s*loop\s*back\s*off", _I)),
    ("OOMKilled", re.compile(r"oom\s*-?killed|out of memory|exit code:?\s*137|\boom\b", _I)),
    ("ImagePullBackOff", re.compile(r"image\s*pull\s*back\s*off|errimagepull|failed to pull image", _I)),
    ("Pod Evicted", re.compile(r"\bevicted\b|disk\s*pressure|memory\s*pressure", _I)),
    ("Probe Failure", re.compile(r"(liveness|readiness|startup) probe failed", _I)),
    ("Node NotReady", re.compile(r"node\s*not\s*ready|\bNotReady\b", _I)),
    ("Connection Refused", re.compile(r"connection refused|ECONNREFUSED", _I)),
    ("Timeout", re.compile(r"timed?\s*out|timeout|deadline exceeded|ETIMEDOUT", _I)),
    ("Bad Gateway / Unavailable", re.compile(r"\b50[234]\b|bad gateway|service unavailable|gateway time-?out", _I)),
    ("Access Denied", re.compile(r"access\s*denied|not authorized|unauthorizedoperation|forbidden|\b403\b|AssumeRole", _I)),
    ("Unauthorized", re.compile(r"\b401\b|unauthorized|invalid (token|credentials)|authentication failed", _I)),
    ("DNS Failure", re.compile(r"nxdomain|no such host|could not resolve|name resolution|servfail", _I)),
    ("Disk Full", re.compile(r"no space left on device|disk (is )?full|ENOSPC", _I)),
    ("Too Many Connections", re.compile(r"too many connections|connection pool (is )?exhausted|remaining connection slots|max_connections", _I)),
    ("Deadlock", re.compile(r"deadlock", _I)),
    ("TLS/Certificate Error", re.compile(r"x509|certificate (has )?(expired|verify)|ssl handshake|tls handshake", _I)),
    ("Throttling", re.compile(r"throttl|rate limit exceeded|\b429\b|too many requests", _I)),
    ("Replication Lag", re.compile(r"replication lag|replica .*behind", _I)),
    ("Fatal/Panic", re.compile(r"\bFATAL\b|\bpanic\b|segmentation fault", _I)),
    ("Stack Trace", re.compile(r"traceback \(most recent|exception in thread|^\s+at [\w.$<>/:-]+\s*\(.*\)|\bcaused by:", _I | re.MULTILINE)),
]

COMPONENT_SIGNALS: list[tuple[str, re.Pattern]] = [
    ("Kubernetes", re.compile(r"kubectl|kubernetes|\bk8s\b|kubelet|\bpods?\b|namespace", _I)),
    ("AWS EKS", re.compile(r"\beks\b", _I)),
    ("AWS", re.compile(r"\baws\b|\bec2\b|\bs3\b|\brds\b|\belb\b|\balb\b|\bnlb\b|cloudwatch|\blambda\b", _I)),
    ("AWS IAM", re.compile(r"\biam\b|assumerole|arn:aws:iam|irsa", _I)),
    ("PostgreSQL", re.compile(r"postgres|psql|\bpg_|pgbouncer", _I)),
    ("MySQL", re.compile(r"mysql|mariadb", _I)),
    ("MongoDB", re.compile(r"mongo(db)?\b", _I)),
    ("Redis", re.compile(r"\bredis\b|elasticache", _I)),
    ("Kafka", re.compile(r"\bkafka\b|consumer lag", _I)),
    ("Ingress / NGINX", re.compile(r"ingress|nginx|\benvoy\b|istio", _I)),
    ("Load Balancer", re.compile(r"load.?balancer|\balb\b|\bnlb\b|haproxy", _I)),
    ("DNS", re.compile(r"\bdns\b|coredns|route\s?53", _I)),
    ("Docker", re.compile(r"\bdocker\b|containerd", _I)),
    ("CI/CD", re.compile(r"jenkins|github actions|gitlab|argocd|\bhelm\b|pipeline|rollout|deployment failed", _I)),
    ("Terraform", re.compile(r"terraform", _I)),
]

# Categories mirror those used in the historical knowledge base.
CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("Kubernetes", re.compile(r"crash\s*loop|kubectl|\bpods?\b|kubelet|imagepull|\bEKS\b|\bk8s\b", _I)),
    ("Memory", re.compile(r"oom|out of memory|memory (leak|usage)|heap", _I)),
    ("CPU", re.compile(r"\bcpu\b|throttl(ed|ing) cpu|load average", _I)),
    ("Database", re.compile(r"postgres|mysql|mongo|deadlock|too many connections|connection pool|replication|\brds\b|sql", _I)),
    ("Cache", re.compile(r"\bredis\b|memcached|elasticache|cache (miss|evict)", _I)),
    ("Storage", re.compile(r"no space left|disk|volume|\bpvc\b|\bebs\b|\bs3\b|inode", _I)),
    ("DNS", re.compile(r"\bdns\b|nxdomain|coredns|no such host|name resolution", _I)),
    ("Load Balancer", re.compile(r"load.?balancer|\balb\b|\bnlb\b|haproxy|target group", _I)),
    ("API Gateway", re.compile(r"api gateway|\bingress\b|nginx|\b50[234]\b|bad gateway|upstream", _I)),
    ("Authentication", re.compile(r"\biam\b|access\s*denied|unauthorized|forbidden|\b40[13]\b|assumerole|oauth|jwt|token", _I)),
    ("Deployment", re.compile(r"deploy|rollout|rollback|pipeline|helm|release", _I)),
    ("Networking", re.compile(r"connection (refused|reset|timed)|timeout|unreachable|network|packet loss|\bvpc\b", _I)),
]

_SEVERITY_TOKENS = re.compile(
    r"\b(critical|fatal|panic|outage|data loss|sev-?[12]|p[01]|service down|unavailable|"
    r"customer[- ]impact\w*|revenue|major incident)\b", _I)
_ERROR_LINE = re.compile(r"\b(ERROR|FATAL|CRITICAL)\b")
_WARN_LINE = re.compile(r"\b(WARN|WARNING)\b")

_SERVICE_PATTERNS = [
    re.compile(r"""["']?\bservice(?:[_-]?name)?["']?\s*[=:]\s*["']?([A-Za-z0-9][\w.-]{1,62})""", _I),
    re.compile(r"""\b(?:app|application|svc)[=:]\s*["']?([A-Za-z0-9][\w.-]{1,62})""", _I),
    re.compile(r"\b(?:deployment|deploy|statefulset|svc)/([a-z0-9][a-z0-9-]{1,62})"),
    re.compile(r"\bpods?/([a-z0-9][a-z0-9-]{1,80})"),
    re.compile(r"\b([a-z][a-z0-9-]{2,}?)-[a-z0-9]{8,10}-[a-z0-9]{5}\b"),  # bare pod name
]
_POD_HASH_SUFFIX = re.compile(r"(-[a-z0-9]{8,10})?-[a-z0-9]{5}$|-\d+$")
_PLACEHOLDER_SERVICES = {"unknown", "none", "null", "name", "the", "is", "in"}

_ENV_PATTERN = re.compile(
    r"(?<![\w/.-])(production|prod|staging|stage|development|dev|qa|uat|ci[-/]?cd)(?![\w/])", _I)
_ENV_NAMES = {"prod": "Production", "production": "Production", "staging": "Staging",
              "stage": "Staging", "dev": "Development", "development": "Development",
              "qa": "QA", "uat": "UAT"}


def _detect_labels(table: list[tuple[str, re.Pattern]], text: str) -> list[str]:
    return [label for label, pattern in table if pattern.search(text)]


def _detect_service(text: str) -> Optional[str]:
    for pattern in _SERVICE_PATTERNS:
        for match in pattern.finditer(text):
            name = _POD_HASH_SUFFIX.sub("", match.group(1)).strip(".-_")
            if name and name.lower() not in _PLACEHOLDER_SERVICES:
                return name
    return None


def _detect_environment(text: str) -> Optional[str]:
    match = _ENV_PATTERN.search(text)
    if not match:
        return None
    key = match.group(1).lower()
    return _ENV_NAMES.get(key, "CI/CD")


def _detect_category(text: str) -> Optional[str]:
    scores = {name: len(pattern.findall(text)) for name, pattern in CATEGORY_RULES}
    best = max(scores, key=scores.get)  # ties resolve by rule order (more specific first)
    return best if scores[best] > 0 else None


def _severity_indicators(text: str) -> list[str]:
    found = sorted({m.group(1).lower() for m in _SEVERITY_TOKENS.finditer(text)})
    errors, warns = len(_ERROR_LINE.findall(text)), len(_WARN_LINE.findall(text))
    if errors:
        found.append(f"{errors} ERROR/FATAL-level line(s)")
    if warns:
        found.append(f"{warns} WARN-level line(s)")
    return found


def build_retrieval_query(clean_text: str, info: dict[str, Any]) -> str:
    stripped = [_LEADING_TIMESTAMP.sub("", line) for line in clean_text.split("\n")]
    body = "\n".join(line for line in _collapse_repeats(stripped) if line.strip())
    header_parts = []
    if info["error_types"]:
        header_parts.append("Signals: " + ", ".join(info["error_types"]))
    if info["affected_components"]:
        header_parts.append("Components: " + ", ".join(info["affected_components"]))
    if info["category"]:
        header_parts.append(f"Category: {info['category']}")
    if info["service"]:
        header_parts.append(f"Service: {info['service']}")
    header = ". ".join(header_parts)
    return (f"{header}\n{body}" if header else body)[:MAX_QUERY_CHARS]


def parse_incident(raw_text: str, llm_input_chars: int = 12_000) -> ParsedIncident:
    """Clean the incident text and extract technical metadata. Never drops error content."""
    text, redactions = redact_secrets(_maybe_flatten_json(raw_text))
    clean = normalize_whitespace(text)

    info: dict[str, Any] = {
        "service": _detect_service(clean),
        "environment": _detect_environment(clean),
        "category": _detect_category(clean),
        "error_types": _detect_labels(ERROR_SIGNALS, clean),
        "affected_components": _detect_labels(COMPONENT_SIGNALS, clean),
        "severity_indicators": _severity_indicators(clean),
    }
    info["keywords"] = list(dict.fromkeys(
        label.lower() for label in info["error_types"] + info["affected_components"]))

    query = build_retrieval_query(clean, info)
    llm_text, truncated = truncate_middle(clean, llm_input_chars)
    return ParsedIncident(
        cleaned_text=llm_text, retrieval_query=query, truncated=truncated,
        redactions=redactions, **info,
    )
