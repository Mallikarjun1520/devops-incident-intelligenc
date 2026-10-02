"""Pydantic models shared across the API, services and LLM output validation."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SEVERITY_LEVELS = ("Critical", "High", "Medium", "Low", "Unknown")
Severity = Literal["Critical", "High", "Medium", "Low", "Unknown"]


def _as_str_list(value: Any) -> list[str]:
    """Coerce None / str / list into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


class Incident(BaseModel):
    """A historical incident stored in the knowledge base."""

    model_config = ConfigDict(extra="ignore")

    incident_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    severity: str = "Unknown"
    category: str = "Unknown"
    service: str = "Unknown"
    environment: str = "Unknown"
    description: str = ""
    symptoms: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    root_cause: str = ""
    resolution: list[str] = Field(default_factory=list)
    preventive_actions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    timestamp: Optional[str] = None
    impact: Optional[str] = None

    @field_validator(
        "symptoms", "logs", "affected_components", "resolution", "preventive_actions", "keywords",
        mode="before",
    )
    @classmethod
    def _coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)


class ParsedIncident(BaseModel):
    """Deterministic facts extracted from the submitted incident text."""

    cleaned_text: str  # whitespace-normalised, secrets redacted; timestamps preserved
    retrieval_query: str  # timestamp-stripped text prefixed with detected signals
    service: Optional[str] = None
    environment: Optional[str] = None
    category: Optional[str] = None
    error_types: list[str] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    severity_indicators: list[str] = Field(default_factory=list)
    truncated: bool = False
    redactions: int = 0


class SimilarIncident(BaseModel):
    """A retrieved historical incident with only the fields useful as evidence."""

    incident_id: str
    title: str
    severity: str
    category: str
    service: str
    similarity: float  # cosine similarity, 0..1
    distance: float  # cosine distance, 1 - similarity
    symptoms: list[str] = Field(default_factory=list)
    root_cause: str = ""
    resolution: list[str] = Field(default_factory=list)
    preventive_actions: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Structured LLM output. Validators make it tolerant of minor formatting drift."""

    incident_summary: str = ""
    severity: Severity = "Unknown"
    severity_score: int = Field(default=0, ge=0, le=100)
    confidence_score: int = Field(default=0, ge=0, le=100)
    affected_components: list[str] = Field(default_factory=list)
    observed_symptoms: list[str] = Field(default_factory=list)
    root_cause: str = ""
    reasoning: str = ""
    recommended_resolution: list[str] = Field(default_factory=list)
    preventive_actions: list[str] = Field(default_factory=list)
    verification_steps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_incidents: list[str] = Field(default_factory=list)

    @field_validator("severity", mode="before")
    @classmethod
    def _normalise_severity(cls, value: Any) -> str:
        text = str(value or "").strip().capitalize()
        return text if text in SEVERITY_LEVELS else "Unknown"

    @field_validator("severity_score", "confidence_score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> int:
        try:
            return max(0, min(100, round(float(value))))
        except (TypeError, ValueError):
            return 0

    @field_validator(
        "affected_components", "observed_symptoms", "recommended_resolution", "preventive_actions",
        "verification_steps", "warnings", "evidence_incidents",
        mode="before",
    )
    @classmethod
    def _coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("incident_summary", "root_cause", "reasoning", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()


class AnalyzeRequest(BaseModel):
    text: str = ""


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    services: dict[str, str]
    indexed_incidents: int = 0


class IngestResponse(BaseModel):
    success: bool
    ingested: int
    incident_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AssistantTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=3000)


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)
    context_token: str = Field(min_length=1, max_length=100_000)
    history: list[AssistantTurn] = Field(default_factory=list, max_length=8)
