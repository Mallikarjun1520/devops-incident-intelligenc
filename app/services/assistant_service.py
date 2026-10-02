"""Guarded read-only assistant bound to a signed analyzer result."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import AnalysisResult, SimilarIncident
from app.services.llm_service import LLMService
from app.services.parser import redact_secrets

CONTEXT_TTL_SECONDS = 60 * 60
REFUSAL = (
    "I can't execute commands, change systems, access external services, make operational decisions, "
    "or reveal credentials. I can provide incident explanations and guidance for human review."
)
OUT_OF_SCOPE = (
    "I can help with this incident, its diagnostics, remediation options, and related reliability "
    "questions. I can't help with personal or unrelated requests."
)
COMMAND_REVIEW_NOTE = (
    "Suggested command(s) for human review only. Check the target, scope, permissions, and impact "
    "before running; nothing is executed by this assistant."
)

_UNSAFE_REQUEST = re.compile(
    r"^\s*(?:(?:please\s+)?(?:run|execute|perform|restart|delete|remove|scale|deploy|rollback|"
    r"modify|change|update|create|apply|destroy|terminate|kill|repair|fix|ssh|sudo|curl|wget)\b"
    r"|(?:can|could|would)\s+you\s+(?:please\s+)?(?:run|execute|perform|restart|delete|remove|"
    r"scale|deploy|rollback|modify|change|update|create|apply|destroy|terminate|kill|repair|fix)"
    r"\b)"
    r"|\b(?:decide for me|make the decision for me|run (?:this|the command) for me|"
    r"execute (?:this|the command) for me)\b",
    re.IGNORECASE,
)
_SECRET_REQUEST = re.compile(
    r"\b(?:show|reveal|print|tell me|give me|extract|what is|what's|return)\b.{0,60}"
    r"\b(?:password|token|api key|credential|secret value)\b"
    r"|\b(?:value|contents) of (?:the )?(?:password|token|api key|credential|secret)\b",
    re.IGNORECASE,
)
_INJECTION_REQUEST = re.compile(
    r"\b(?:ignore|disregard|override)\b.{0,40}\b(?:previous|prior|above|system|developer)"
    r"\b|\b(?:system prompt|developer message|hidden instructions)\b",
    re.IGNORECASE,
)
_COMMAND_OUTPUT = re.compile(
    r"\b(?:kubectl|helm|terraform|docker|aws|gcloud|az|curl|wget|ssh|sudo|"
    r"systemctl|service|rm|kill|chmod)\s+\S+",
    re.IGNORECASE,
)
_CLAIMED_ACTION = re.compile(
    r"\bI (?:have |will |am going to )?(?:executed|run|ran|restart|restarted|deleted|removed|"
    r"scaled|deployed|changed|modified|updated|created|fixed|repaired|accessed)\b",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"\b([A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?KEY|_KEY|SECRET|TOKEN|PASSWORD|PASSWD|"
    r"CREDENTIAL)[A-Z0-9_]*)(\s*[=:]\s*)(?!\*)[^\s,;\"']+",
    re.IGNORECASE,
)


def _redact_sensitive_text(value: str) -> str:
    redacted, _ = redact_secrets(value)
    return _SENSITIVE_ASSIGNMENT.sub(r"\1\2[REDACTED]", redacted)


class AssistantContextError(ValueError):
    """The signed analyzer context is invalid or expired."""


class VerifiedAssistantContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_text: str = Field(max_length=12_000)
    analysis: AnalysisResult
    similar_incidents: list[SimilarIncident] = Field(max_length=10)


def _clean_context_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_sensitive_text(value)[:4000]
    if isinstance(value, list):
        return [_clean_context_value(item) for item in value[:50]]
    if isinstance(value, dict):
        return {key: _clean_context_value(item) for key, item in value.items()}
    return value


def create_context_token(
    signing_key: bytes,
    incident_text: str,
    analysis: AnalysisResult,
    similar_incidents: list[SimilarIncident],
) -> str:
    """Sign redacted, size-bounded context so chat cannot substitute another incident."""
    safe_incident = _redact_sensitive_text(incident_text)
    context = VerifiedAssistantContext(
        incident_text=safe_incident[:12_000],
        analysis=_clean_context_value(analysis.model_dump()),
        similar_incidents=[
            _clean_context_value(item.model_dump()) for item in similar_incidents[:10]
        ],
    )
    payload = {
        "expires_at": int(time.time()) + CONTEXT_TTL_SECONDS,
        "context": context.model_dump(mode="json"),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = hmac.new(signing_key, encoded.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def verify_context_token(signing_key: bytes, token: str) -> VerifiedAssistantContext:
    if len(token) > 100_000 or "." not in token:
        raise AssistantContextError("Incident context is invalid or expired. Analyze the incident again.")
    encoded, signature = token.rsplit(".", 1)
    try:
        encoded_bytes = encoded.encode("ascii")
    except UnicodeError as exc:
        raise AssistantContextError(
            "Incident context is invalid or expired. Analyze the incident again."
        ) from exc
    expected = hmac.new(signing_key, encoded_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise AssistantContextError("Incident context is invalid or expired. Analyze the incident again.")
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if payload["expires_at"] < int(time.time()):
            raise AssistantContextError("Incident context is invalid or expired. Analyze the incident again.")
        return VerifiedAssistantContext.model_validate(payload["context"])
    except AssistantContextError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise AssistantContextError(
            "Incident context is invalid or expired. Analyze the incident again."
        ) from exc


class IncidentAssistantService:
    def __init__(self, llm: LLMService, signing_key: bytes) -> None:
        self._llm = llm
        self._signing_key = signing_key

    def answer(
        self,
        question: str,
        context_token: str,
        history: list[dict[str, str]],
    ) -> tuple[str, bool]:
        question = question.strip()
        if _INJECTION_REQUEST.search(question) or _UNSAFE_REQUEST.search(question):
            return REFUSAL, True
        if _SECRET_REQUEST.search(question):
            return "Credential values are not available to the assistant and will not be disclosed.", True
        if _is_personal_or_unrelated(question):
            return OUT_OF_SCOPE, True

        context = verify_context_token(self._signing_key, context_token)
        safe_history = [
            {"role": turn["role"], "content": _redact_sensitive_text(turn["content"])}
            for turn in history[-8:]
        ]
        answer = self._llm.answer_question(
            _redact_sensitive_text(question),
            context.model_dump(mode="json"),
            safe_history,
        )
        if _CLAIMED_ACTION.search(answer):
            return REFUSAL, True
        safe_answer = _redact_sensitive_text(answer)
        if _COMMAND_OUTPUT.search(safe_answer) and COMMAND_REVIEW_NOTE.lower() not in safe_answer.lower():
            answer_limit = 3000 - len(COMMAND_REVIEW_NOTE) - 2
            safe_answer = f"{safe_answer[:answer_limit].rstrip()}\n\n{COMMAND_REVIEW_NOTE}"
        return safe_answer[:3000], False


_PERSONAL_OR_UNRELATED = re.compile(
    r"\b(?:personal advice|personal problem|personal question|my personal life|"
    r"relationship advice|my relationship|dating advice|write (?:me )?(?:a )?"
    r"(?:joke|poem|story)|tell (?:me )?(?:a )?joke|recipe for|what should i eat|"
    r"vacation|travel itinerary|homework help|solve this math|stock pick|"
    r"investment advice|medical advice|diagnose my|legal advice|"
    r"what do you think of me|how was your day|who are you dating|"
    r"favorite color|favourite colour|tell me about yourself|"
    r"my life|my career)\b",
    re.IGNORECASE,
)


def _is_personal_or_unrelated(question: str) -> bool:
    return bool(_PERSONAL_OR_UNRELATED.search(question))
