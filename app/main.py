"""FastAPI application factory. Run with: uvicorn app.main:app --reload"""
from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router, error_response, root_router
from app.config import FRONTEND_DIR, get_settings
from app.services.incident_service import IncidentService, build_incident_service


def create_app(incident_service: Optional[IncidentService] = None) -> FastAPI:
    """``incident_service`` can be injected (tests); by default it is built from the environment."""
    service = incident_service or build_incident_service(get_settings())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.warm_up()
        yield

    app = FastAPI(title="DevOps Incident Intelligence Platform", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.incident_service = service
    app.state.assistant_signing_key = secrets.token_bytes(32)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response("Invalid request: " + "; ".join(str(e["msg"]) for e in exc.errors()), 422)

    app.include_router(root_router)
    app.include_router(api_router)
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    return app


app = create_app()
