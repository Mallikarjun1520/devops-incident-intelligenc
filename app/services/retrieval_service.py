"""ChromaDB-backed retrieval of similar historical incidents + LLM context construction."""
from __future__ import annotations

import json
import threading
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.models.schemas import Incident, SimilarIncident
from app.services.embedding_service import EmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_LIST_FIELDS = ("symptoms", "logs", "affected_components", "resolution", "preventive_actions", "keywords")
_BATCH_SIZE = 64


def build_document(incident: Incident) -> str:
    """Text that gets embedded. It deliberately describes *what happened* (symptoms, logs,
    cause) and omits fixes, so a new incident matches on its symptoms rather than on remedies."""
    parts = [
        f"Incident: {incident.title}.",
        f"Category: {incident.category}. Service: {incident.service}.",
        f"Description: {incident.description}" if incident.description else "",
        f"Symptoms: {'; '.join(incident.symptoms)}." if incident.symptoms else "",
        f"Log lines: {' | '.join(incident.logs[:8])}." if incident.logs else "",
        f"Affected components: {', '.join(incident.affected_components)}." if incident.affected_components else "",
        f"Root cause: {incident.root_cause}" if incident.root_cause else "",
    ]
    return " ".join(part for part in parts if part)


def _to_metadata(incident: Incident) -> dict[str, Any]:
    """Chroma metadata only allows scalars, so list fields are stored as JSON strings."""
    data = incident.model_dump()
    meta: dict[str, Any] = {key: ("" if value is None else value) for key, value in data.items()
                            if key not in _LIST_FIELDS}
    for key in _LIST_FIELDS:
        meta[key] = json.dumps(data[key])
    return meta


def _from_metadata(meta: dict[str, Any]) -> Incident:
    data = dict(meta)
    for key in _LIST_FIELDS:
        data[key] = json.loads(data.get(key) or "[]")
    return Incident(**data)


class RetrievalService:
    def __init__(self, settings: Settings, embedder: EmbeddingService) -> None:
        self._settings = settings
        self._embedder = embedder
        self._client: Any = None
        self._collection: Any = None
        self._lock = threading.Lock()

    # -- connection --------------------------------------------------------------------------

    def _get_collection(self) -> Any:
        with self._lock:
            if self._collection is None:
                self._settings.chroma_path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self._settings.chroma_path),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self._settings.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,  # we supply our own embeddings
                )
            return self._collection

    def ping(self) -> bool:
        """True if ChromaDB is reachable and the collection can be opened."""
        try:
            self._get_collection()
            self._client.heartbeat()
            return True
        except Exception as exc:
            logger.error("ChromaDB unavailable: %s", exc)
            return False

    def count(self) -> int:
        return self._get_collection().count()

    def reset(self) -> None:
        """Drop and recreate the collection (used by `ingest_incidents.py --reset`)."""
        self._get_collection()
        self._client.delete_collection(self._settings.collection_name)
        with self._lock:
            self._collection = None
        self._get_collection()

    # -- indexing ----------------------------------------------------------------------------

    def index_incidents(self, incidents: list[Incident]) -> int:
        """Embed and upsert incidents (same incident_id overwrites). Returns number indexed."""
        collection = self._get_collection()
        for start in range(0, len(incidents), _BATCH_SIZE):
            batch = incidents[start:start + _BATCH_SIZE]
            documents = [build_document(incident) for incident in batch]
            collection.upsert(
                ids=[incident.incident_id for incident in batch],
                embeddings=self._embedder.embed(documents),
                documents=documents,
                metadatas=[_to_metadata(incident) for incident in batch],
            )
        return len(incidents)

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        result = self._get_collection().get(ids=[incident_id], include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return _from_metadata(metadatas[0]) if metadatas else None

    # -- querying ----------------------------------------------------------------------------

    def retrieve_similar_incidents(self, query: str, top_k: Optional[int] = None) -> list[SimilarIncident]:
        """Return up to ``top_k`` incidents whose cosine similarity >= MIN_SIMILARITY,
        best match first. Returns [] when nothing relevant exists."""
        top_k = top_k or self._settings.top_k
        collection = self._get_collection()
        available = collection.count()
        if available == 0 or not query.strip():
            return []

        result = collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=min(top_k, available),
            include=["metadatas", "distances"],
        )
        similar: list[SimilarIncident] = []
        for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            if similarity < self._settings.min_similarity:
                continue
            incident = _from_metadata(meta)
            similar.append(SimilarIncident(
                incident_id=incident.incident_id, title=incident.title, severity=incident.severity,
                category=incident.category, service=incident.service,
                similarity=round(similarity, 4), distance=round(float(distance), 4),
                symptoms=incident.symptoms, root_cause=incident.root_cause,
                resolution=incident.resolution, preventive_actions=incident.preventive_actions,
            ))
        return similar


def format_historical_context(similar: list[SimilarIncident]) -> str:
    """Context-construction layer: expose only the fields useful for reasoning, never whole
    documents. The IDs shown here are the only IDs the LLM may cite as evidence."""
    if not similar:
        return "No sufficiently similar historical incidents were retrieved."

    def bullets(items: list[str]) -> str:
        return "\n".join(f"  - {item}" for item in items) if items else "  - (none recorded)"

    blocks = []
    for item in similar:
        blocks.append(
            f'<historical_incident id="{item.incident_id}" similarity="{item.similarity:.2f}">\n'
            f"Title: {item.title}\nSeverity: {item.severity}\nCategory: {item.category}\n"
            f"Service: {item.service}\n"
            f"Symptoms:\n{bullets(item.symptoms)}\n"
            f"Root cause: {item.root_cause or '(not recorded)'}\n"
            f"Resolution:\n{bullets(item.resolution)}\n"
            f"Preventive actions:\n{bullets(item.preventive_actions)}\n"
            f"</historical_incident>"
        )
    return "\n\n".join(blocks)
