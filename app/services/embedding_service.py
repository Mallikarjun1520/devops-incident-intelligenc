"""Local sentence-transformers embeddings, loaded lazily and shared across requests."""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _load_sentence_transformer(model_name: str) -> Any:
    # Imported here so the API can start (and report status) even if the model is unavailable.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", model_name)
    return SentenceTransformer(model_name)


class EmbeddingService:
    """Wraps an embedding model. ``loader`` is injectable (used by tests)."""

    def __init__(self, model_name: str, loader: Optional[Callable[[str], Any]] = None) -> None:
        self.model_name = model_name
        self._loader = loader or _load_sentence_transformer
        self._model: Any = None
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                self._model = self._loader(self.model_name)
                self.last_error = None
            except Exception as exc:  # network failure, bad model name, missing package...
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Embedding model failed to load: %s", self.last_error)
                raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised embeddings, one vector per input text."""
        self.load()
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in vector] for vector in vectors]
