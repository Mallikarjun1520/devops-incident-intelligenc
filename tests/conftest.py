"""Shared fixtures. No network, real embedding model, or Groq key is needed."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models.schemas import Incident  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.incident_service import IncidentService, load_incidents_from_directory  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402

DIM = 512


class FakeEncoder:
    """Deterministic bag-of-words hashing encoder: texts sharing words get similar vectors."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


VALID_ANALYSIS = {
    "incident_summary": "orders-api pods crash at startup.",
    "severity": "High", "severity_score": 82, "confidence_score": 88,
    "affected_components": ["orders-api", "Kubernetes"],
    "observed_symptoms": ["CrashLoopBackOff", "503 errors"],
    "root_cause": "Inference: missing PAYMENT_GATEWAY_KEY environment variable.",
    "reasoning": "Observed: log says the variable is not set. Inference: matches CUR-001.",
    "recommended_resolution": ["Run `kubectl describe pod <pod> -n <namespace>` to confirm."],
    "preventive_actions": ["Validate env vars in CI."],
    "verification_steps": ["Confirm all pods are Ready."],
    "warnings": [],
    "evidence_incidents": ["CUR-001", "INC-DOES-NOT-EXIST"],
}


class FakeGroq:
    """Stands in for the Groq chat-completions client."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.reply),
                ),
            ],
        )


def make_settings(tmp_path: Path, api_key: str = "test-key") -> Settings:
    return Settings(groq_api_key=api_key, groq_model="test-model",
                    embedding_model="fake", chroma_path=tmp_path / "chroma", top_k=3,
                    min_similarity=0.15)


def build_service(tmp_path: Path, llm_client=None, api_key: str = "test-key") -> IncidentService:
    settings = make_settings(tmp_path, api_key)
    embedder = EmbeddingService("fake", loader=lambda _name: FakeEncoder())
    return IncidentService(settings, embedder, RetrievalService(settings, embedder),
                           LLMService(settings, client=llm_client))


@pytest.fixture(scope="session")
def curated_incidents() -> list[Incident]:
    incidents, errors = load_incidents_from_directory(
        Path(__file__).resolve().parent.parent / "data" / "incidents")
    assert not errors
    return [i for i in incidents if i.incident_id.startswith("CUR-") or i.incident_id == "INC-266"]


@pytest.fixture
def service(tmp_path, curated_incidents) -> IncidentService:
    svc = build_service(tmp_path, FakeGroq(json.dumps(VALID_ANALYSIS)))
    svc.ingest(curated_incidents)
    return svc


@pytest.fixture
def client(service) -> TestClient:
    return TestClient(create_app(service))


@pytest.fixture
def empty_client(tmp_path) -> TestClient:
    """App with an empty knowledge base and no API key."""
    return TestClient(create_app(build_service(tmp_path, llm_client=None, api_key="")))
