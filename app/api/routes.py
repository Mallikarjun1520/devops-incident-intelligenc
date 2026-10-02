"""HTTP routes. Thin layer: validation and response shaping only."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.config import DEFAULT_INCIDENTS_DIR, FRONTEND_DIR
from app.models.schemas import AssistantRequest, HealthResponse, IngestResponse
from app.services.assistant_service import (
    AssistantContextError, IncidentAssistantService, create_context_token,
)
from app.services.incident_service import (
    ALLOWED_UPLOAD_EXTENSIONS, AnalysisFailure, IncidentService, InvalidInputError, PayloadTooLargeError,
    UnsupportedFileError, decode_upload, parse_incident_records,
)
from app.services.llm_service import LLMError
from app.services.severity_service import assess_severity

root_router = APIRouter()
api_router = APIRouter(prefix="/api")


def _service(request: Request) -> IncidentService:
    return request.app.state.incident_service


def error_response(message: str, status_code: int, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": False, "error": message, **extra})


def _input_error(exc: InvalidInputError) -> JSONResponse:
    status = 415 if isinstance(exc, UnsupportedFileError) else 413 if isinstance(exc, PayloadTooLargeError) else 400
    return error_response(str(exc), status)


async def _read_payload(request: Request, max_bytes: int,
                        upload_file: UploadFile | None = None,
                        form_text: str | None = None,
                        allowed: frozenset[str] = frozenset(ALLOWED_UPLOAD_EXTENSIONS)) -> tuple[str, Any]:
    """Return (kind, data) from either a multipart upload or a JSON body.
    kind is 'file' (data = decoded text) or 'json' (data = parsed JSON)."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = upload_file or form.get("file")
        if upload is not None and hasattr(upload, "read"):
            content = await upload.read(max_bytes + 1)
            return "file", decode_upload(upload.filename or "", content, max_bytes, allowed)
        if form_text is not None:
            return "form", {"text": form_text}
        return "form", form
    try:
        return "json", await request.json()
    except ValueError as exc:
        raise InvalidInputError(f"Request body is not valid JSON: {exc}") from exc


# -- root -------------------------------------------------------------------------------------

@root_router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@root_router.get("/style.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@root_router.get("/script.js", include_in_schema=False)
def frontend_script() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "script.js", media_type="application/javascript")


@root_router.get("/sample", include_in_schema=False)
def sample_incident() -> FileResponse:
    return FileResponse(DEFAULT_INCIDENTS_DIR.parent / "sample_incident.txt", media_type="text/plain")


@root_router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Reports component state. Never depends on a live Groq call."""
    service = _service(request)
    chroma_up = service.retriever.ping()
    indexed = 0
    if chroma_up:
        try:
            indexed = service.retriever.count()
        except Exception:
            chroma_up = False
    embedder = service.embedder
    embed_state = "loaded" if embedder.is_loaded else ("error" if embedder.last_error else "not_loaded")
    services = {
        "api": "up",
        "chromadb": "up" if chroma_up else "down",
        "embedding_model": embed_state,
        "llm": "configured" if service.llm.is_configured else "not_configured",
    }
    return HealthResponse(status="healthy" if chroma_up and embed_state != "error" else "degraded",
                          services=services, indexed_incidents=indexed)


# -- api --------------------------------------------------------------------------------------

@api_router.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
) -> JSONResponse:
    """Accepts JSON ``{"text": "..."}`` or multipart with ``file`` (.txt/.log/.json) and/or ``text``."""
    service = _service(request)
    try:
        kind, data = await _read_payload(
            request, service.settings.max_upload_bytes, upload_file=file, form_text=text,
        )
        if kind == "file":
            text = data
        elif kind == "form":
            text = str(data.get("text") or "")
        else:
            if not isinstance(data, dict):
                raise InvalidInputError('Expected a JSON object like {"text": "..."}.')
            text = data.get("text")
            if text is not None and not isinstance(text, str):
                raise InvalidInputError("'text' must be a string.")
        outcome = await run_in_threadpool(service.analyze_text, text or "")
    except InvalidInputError as exc:
        return _input_error(exc)
    except AnalysisFailure as exc:
        return error_response(
            str(exc), exc.status_code, error_code=exc.code,
            similar_incidents=[item.model_dump() for item in exc.similar],
            extracted_signals=exc.signals.model_dump(exclude={"cleaned_text", "retrieval_query"}) if exc.signals else None,
            severity_assessment=assess_severity(exc.signals) if exc.signals else None,
        )
    return JSONResponse({
        "success": True,
        "analysis": outcome.analysis.model_dump(),
        "severity_assessment": assess_severity(outcome.signals),
        "similar_incidents": [item.model_dump() for item in outcome.similar],
        "extracted_signals": outcome.signals.model_dump(exclude={"cleaned_text", "retrieval_query"}),
        "assistant_context": create_context_token(
            request.app.state.assistant_signing_key,
            outcome.signals.cleaned_text,
            outcome.analysis,
            outcome.similar,
        ),
    })


@api_router.post("/assistant")
async def ask_assistant(payload: AssistantRequest, request: Request) -> JSONResponse:
    """Answer a question from a signed result context; the assistant has no tools or actions."""
    assistant = IncidentAssistantService(
        _service(request).llm,
        request.app.state.assistant_signing_key,
    )
    try:
        answer, refused = await run_in_threadpool(
            assistant.answer,
            payload.question,
            payload.context_token,
            [turn.model_dump() for turn in payload.history],
        )
    except AssistantContextError as exc:
        return error_response(str(exc), 409, error_code="invalid_assistant_context")
    except LLMError as exc:
        return error_response(str(exc), exc.status_code, error_code=exc.code)
    return JSONResponse({"success": True, "answer": answer, "refused": refused})


@api_router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: UploadFile | None = File(default=None),
) -> JSONResponse:
    """Add historical incidents to the knowledge base (JSON body or .json upload)."""
    service = _service(request)
    try:
        kind, data = await _read_payload(
            request, service.settings.max_upload_bytes,
            upload_file=file, allowed=frozenset({".json"}),
        )
        if kind == "form":
            raise InvalidInputError("Attach a .json file in the 'file' field.")
        if kind == "file":
            data = json.loads(data)  # decode_upload already validated .json syntax
        incidents, errors = parse_incident_records(data)
        if not incidents:
            return error_response("No valid incidents found. " + "; ".join(errors[:5]), 400)
        count = await run_in_threadpool(service.ingest, incidents)
    except InvalidInputError as exc:
        return _input_error(exc)
    except Exception as exc:
        return error_response(f"Ingestion failed: {type(exc).__name__}: {exc}", 503)
    return JSONResponse(IngestResponse(
        success=True, ingested=count, incident_ids=[i.incident_id for i in incidents], errors=errors,
    ).model_dump())


@api_router.get("/incidents/{incident_id}")
def get_incident(incident_id: str, request: Request) -> JSONResponse:
    try:
        incident = _service(request).get_incident(incident_id)
    except Exception as exc:
        return error_response(f"Knowledge base unavailable: {type(exc).__name__}", 503)
    if incident is None:
        return error_response(f"Incident '{incident_id}' not found.", 404)
    return JSONResponse({"success": True, "incident": incident.model_dump()})
