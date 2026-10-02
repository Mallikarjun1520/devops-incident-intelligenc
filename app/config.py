"""Application configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_INCIDENTS_DIR = PROJECT_ROOT / "data" / "incidents"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve_path(raw: str) -> Path:
    """Resolve relative paths against the project root, not the working directory."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_number(name: str, default: float, cast=float):
    try:
        return cast(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    embedding_model: str
    chroma_path: Path
    top_k: int
    min_similarity: float

    collection_name: str = "incidents"

    max_incident_chars: int = 200_000

    max_upload_bytes: int = 2_000_000

    llm_input_chars: int = 12_000


    @property
    def llm_configured(self) -> bool:
        return bool(self.groq_api_key)


def load_settings() -> Settings:

    # Real environment variables take precedence over .env
    load_dotenv(
        PROJECT_ROOT / ".env",
        override=False,
    )

    return Settings(

        groq_api_key=os.getenv(
            "GROQ_API_KEY",
            "",
        ).strip(),

        groq_model=os.getenv(
            "GROQ_MODEL",
            "",
        ).strip() or DEFAULT_GROQ_MODEL,

        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "",
        ).strip() or DEFAULT_EMBEDDING_MODEL,

        chroma_path=_resolve_path(
            os.getenv(
                "CHROMA_PATH",
                "",
            ).strip() or "./chroma_db"
        ),

        top_k=max(
            1,
            min(
                10,
                _read_number(
                    "TOP_K",
                    3,
                    int,
                ),
            ),
        ),

        min_similarity=max(
            0.0,
            min(
                1.0,
                _read_number(
                    "MIN_SIMILARITY",
                    0.30,
                ),
            ),
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()