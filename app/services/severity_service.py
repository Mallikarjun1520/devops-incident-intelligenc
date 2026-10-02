"""Deterministic severity index based only on explicit incident text signals."""
from __future__ import annotations

import re

from app.models.schemas import ParsedIncident

_CRITICAL_DECLARATION = re.compile(
    r"\b(?:critical|sev[- ]?1|p0|fatal|panic|outage|data loss|service down)\b",
    re.IGNORECASE,
)
_MAJOR_DECLARATION = re.compile(
    r"\b(?:sev[- ]?2|p1|major incident|customer[- ]impact(?:ed)?|revenue impact)\b",
    re.IGNORECASE,
)
_NEGATED_DECLARATION = re.compile(
    r"\b(?:no|not|never|without|resolved|cleared|fixed|mitigated)\b(?:\s+\w+){0,2}\s*$",
    re.IGNORECASE,
)
_ERROR_LINE_COUNT = re.compile(r"^\d+ ERROR/FATAL-level line\(s\)$")
_AVAILABILITY_SIGNALS = {
    "bad gateway / unavailable",
    "connection refused",
    "crashloopbackoff",
    "node notready",
    "probe failure",
}


def _has_non_negated_match(pattern: re.Pattern[str], text: str) -> bool:
    for match in pattern.finditer(text):
        preceding_text = text[max(0, match.start() - 45):match.start()]
        if not _NEGATED_DECLARATION.search(preceding_text):
            return True
    return False


def assess_severity(parsed: ParsedIncident) -> dict[str, object]:
    """Return an explainable 1-100 score, or no score when there is no severity evidence.

    Rules add points for explicit priority declarations, availability failures,
    production context paired with an availability failure, and observed error
    log-line counts. This is a signal index, not an SLO or customer-impact measure.
    """
    score = 1
    evidence: list[str] = []
    text = parsed.cleaned_text

    if _has_non_negated_match(_CRITICAL_DECLARATION, text):
        score += 75
        evidence.append("Explicit critical / SEV-1 / P0 / outage declaration (+75)")
    elif _has_non_negated_match(_MAJOR_DECLARATION, text):
        score += 55
        evidence.append("Explicit SEV-2 / P1 / major-impact declaration (+55)")

    availability = sorted(
        signal for signal in parsed.error_types
        if signal.casefold() in _AVAILABILITY_SIGNALS
    )
    if availability:
        score += 35
        evidence.append(f"Availability failure detected: {', '.join(availability)} (+35)")
        if parsed.environment == "Production":
            score += 10
            evidence.append("Production environment explicitly detected (+10)")

    error_count = next(
        (
            int(match.group().split(" ", 1)[0])
            for indicator in parsed.severity_indicators
            if (match := _ERROR_LINE_COUNT.fullmatch(indicator))
        ),
        0,
    )
    if error_count:
        points = min(error_count * 5, 20)
        score += points
        evidence.append(f"{error_count} ERROR/FATAL-level log line(s) (+{points}, capped at +20)")

    if not evidence:
        return {
            "score": None,
            "level": "Unknown",
            "evidence": [],
            "basis": "No explicit severity, availability, or ERROR/FATAL log signals detected.",
        }

    score = min(score, 100)
    if score >= 75:
        level = "Critical"
    elif score >= 50:
        level = "High"
    elif score >= 25:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "score": score,
        "level": level,
        "evidence": evidence,
        "basis": (
            "Rule-based signal index; Low 1-24, Moderate 25-49, High 50-74, "
            "Critical 75-100. Not a measure of SLO burn or users affected."
        ),
    }
