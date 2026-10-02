"""Orchestrates the pipeline: parse -> retrieve -> reason -> ground the result.

Extension points (not implemented): webhook adapters (n8n, Alertmanager, PagerDuty) can call
``IncidentService.analyze_text`` directly; audit logging, incident history and approval workflows
would wrap the ``analyze_text`` call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from app.config import Settings
from app.models.schemas import AnalysisResult, Incident, ParsedIncident, SimilarIncident
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMError, LLMService
from app.services.parser import parse_incident
from app.services.retrieval_service import RetrievalService
from app.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".log", ".json"}

ADVISORY_WARNING = ("Human review required. This assessment is advisory only; nothing has been executed. "
                    "Review every suggested command before running it.")


class InvalidInputError(ValueError):
    """Client sent unusable input (empty incident, malformed JSON, ...)."""


class UnsupportedFileError(InvalidInputError):
    """Uploaded file has an unsupported extension."""


class PayloadTooLargeError(InvalidInputError):
    """Incident text or upload exceeds the configured limit."""


class AnalysisFailure(Exception):
    """Analysis could not complete; carries whatever evidence was gathered."""

    def __init__(self, message: str, code: str, status_code: int,
                 similar: list[SimilarIncident], signals: Optional[ParsedIncident]) -> None:
        super().__init__(message)
        self.code, self.status_code = code, status_code
        self.similar, self.signals = similar, signals


class AnalysisOutcome:
    def __init__(self, analysis: AnalysisResult, similar: list[SimilarIncident], signals: ParsedIncident):
        self.analysis, self.similar, self.signals = analysis, similar, signals


# -- input helpers -----------------------------------------------------------------------------

def decode_upload(filename: str, content: bytes, max_bytes: int,
                  allowed: frozenset[str] = frozenset(ALLOWED_UPLOAD_EXTENSIONS)) -> str:
    """Validate an uploaded file and return its text. For .json the syntax is verified."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise UnsupportedFileError(
            f"Unsupported file type '{suffix or filename}'. Allowed: {', '.join(sorted(allowed))}.")
    if len(content) > max_bytes:
        raise PayloadTooLargeError(f"File is too large (limit {max_bytes // 1000} KB).")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidInputError("File is not valid UTF-8 text.") from exc
    if suffix == ".json":
        try:
            json.loads(text)
        except ValueError as exc:
            raise InvalidInputError(f"Malformed JSON in uploaded file: {exc}") from exc
    return text


def parse_incident_records(payload: Any) -> tuple[list[Incident], list[str]]:
    """Validate historical-incident records (one object, a list, or {"incidents": [...]}).
    Returns (valid incidents deduplicated by ID, human-readable errors)."""
    if isinstance(payload, dict) and isinstance(payload.get("incidents"), list):
        payload = payload["incidents"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise InvalidInputError("Expected an incident object, a list of incidents, or {'incidents': [...]}.")

    valid: dict[str, Incident] = {}
    errors: list[str] = []
    for position, record in enumerate(payload):
        try:
            incident = Incident(**record) if isinstance(record, dict) else None
            if incident is None:
                raise TypeError("record is not an object")
        except (ValidationError, TypeError) as exc:
            label = record.get("incident_id", f"#{position}") if isinstance(record, dict) else f"#{position}"
            errors.append(f"{label}: invalid record ({type(exc).__name__})")
            continue
        if incident.incident_id in valid:
            errors.append(f"{incident.incident_id}: duplicate ID, kept the last occurrence")
        valid[incident.incident_id] = incident
    return list(valid.values()), errors


def load_incidents_from_directory(directory: Path) -> tuple[list[Incident], list[str]]:
    """Load every *.json file in ``directory`` as historical incidents."""
    records: list[Any] = []
    errors: list[str] = []
    for file in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(file.read_text(encoding="utf-8-sig"))
        except ValueError as exc:
            errors.append(f"{file.name}: malformed JSON ({exc})")
            continue
        records.extend(payload["incidents"] if isinstance(payload, dict) and "incidents" in payload
                       else payload if isinstance(payload, list) else [payload])
    incidents, record_errors = parse_incident_records(records)
    return incidents, errors + record_errors


# -- service -----------------------------------------------------------------------------------

class IncidentService:
    def __init__(self, settings: Settings, embedder: EmbeddingService,
                 retriever: RetrievalService, llm: LLMService) -> None:
        self.settings, self.embedder, self.retriever, self.llm = settings, embedder, retriever, llm

    def warm_up(self) -> None:
        """Best-effort startup: load the embedding model and open ChromaDB. Never raises."""
        try:
            self.embedder.load()
        except Exception:
            logger.warning("Embedding model not loaded at startup; retrieval will be unavailable.")
        self.retriever.ping()

    def analyze_text(self, text: str) -> AnalysisOutcome:
        text = (text or "").strip()
        if not text:
            raise InvalidInputError("Incident text is empty. Paste logs or an incident description.")
        if len(text) > self.settings.max_incident_chars:
            raise PayloadTooLargeError(
                f"Incident is too large ({len(text):,} characters; limit {self.settings.max_incident_chars:,}).")

        parsed = parse_incident(text, self.settings.llm_input_chars)
        similar, retrieval_note = self._retrieve(parsed)

        try:
            analysis = self.llm.analyze(parsed, similar)
        except LLMError as exc:
            raise AnalysisFailure(str(exc), exc.code, exc.status_code, similar, parsed) from exc
        except Exception as exc:  # last-resort guard: the API must not crash on LLM trouble
            logger.exception("Unexpected analysis failure")
            raise AnalysisFailure(f"Unexpected analysis error: {type(exc).__name__}",
                                  "analysis_error", 500, similar, parsed) from exc

        return AnalysisOutcome(self._ground(analysis, similar, retrieval_note), similar, parsed)

    def _retrieve(self, parsed: ParsedIncident) -> tuple[list[SimilarIncident], Optional[str]]:
        """Retrieval failures degrade the analysis instead of aborting it."""
        try:
            if self.retriever.count() == 0:
                return [], ("The historical knowledge base is empty. Run "
                            "`python scripts/ingest_incidents.py` to enable evidence retrieval.")
            similar = self.retriever.retrieve_similar_incidents(parsed.retrieval_query)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return [], f"Historical retrieval is unavailable ({type(exc).__name__}); analysis uses the current incident only."
        if not similar:
            return [], ("No sufficiently similar historical incidents were found; "
                        "this assessment relies on the current incident alone.")
        return similar, None

    @staticmethod
    def _ground(analysis: AnalysisResult, similar: list[SimilarIncident],
                retrieval_note: Optional[str]) -> AnalysisResult:
        """Server-side guarantees that do not depend on the LLM behaving."""
        retrieved_ids = {item.incident_id for item in similar}
        cited = list(dict.fromkeys(analysis.evidence_incidents))
        kept = [incident_id for incident_id in cited if incident_id in retrieved_ids]
        warnings = [ADVISORY_WARNING]
        if retrieval_note:
            warnings.append(retrieval_note)
        if len(kept) != len(cited):
            dropped = ", ".join(i for i in cited if i not in retrieved_ids)
            warnings.append(f"Removed evidence IDs that were not retrieved for this incident: {dropped}.")
        warnings.extend(w for w in analysis.warnings if w != ADVISORY_WARNING)
        return analysis.model_copy(update={"evidence_incidents": kept, "warnings": warnings})

    def ingest(self, incidents: list[Incident]) -> int:
        return self.retriever.index_incidents(incidents)

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        return self.retriever.get_incident(incident_id)


def build_incident_service(settings: Settings) -> IncidentService:
    embedder = EmbeddingService(settings.embedding_model)
    return IncidentService(settings, embedder, RetrievalService(settings, embedder), LLMService(settings))
