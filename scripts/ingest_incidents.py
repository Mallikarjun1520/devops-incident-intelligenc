"""Load historical incidents from data/incidents/*.json into ChromaDB.

Usage:
    python scripts/ingest_incidents.py             # upsert all incidents
    python scripts/ingest_incidents.py --reset     # drop the collection first
    python scripts/ingest_incidents.py --path DIR  # read JSON files from another directory
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_INCIDENTS_DIR, load_settings  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.incident_service import load_incidents_from_directory  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=DEFAULT_INCIDENTS_DIR, help="directory containing *.json incident files")
    parser.add_argument("--reset", action="store_true", help="delete the existing collection before ingesting")
    args = parser.parse_args()

    settings = load_settings()
    if not args.path.is_dir():
        print(f"ERROR: incident directory not found: {args.path}")
        return 1

    incidents, errors = load_incidents_from_directory(args.path)
    for message in errors:
        print(f"  skipped: {message}")
    if not incidents:
        print("ERROR: no valid incidents found.")
        return 1

    print(f"Loaded {len(incidents)} incidents from {args.path}")
    print(f"Embedding model: {settings.embedding_model} (downloaded on first run, may take a minute)")
    embedder = EmbeddingService(settings.embedding_model)
    retriever = RetrievalService(settings, embedder)
    try:
        if args.reset:
            retriever.reset()
        indexed = retriever.index_incidents(incidents)
    except Exception as exc:
        print(f"ERROR: ingestion failed: {type(exc).__name__}: {exc}")
        if not embedder.is_loaded:
            print("The embedding model could not be loaded. Check your internet connection (first run "
                  "downloads the model) and that EMBEDDING_MODEL in .env is correct.")
        return 1

    print(f"Indexed {indexed} incidents into '{settings.collection_name}' at {settings.chroma_path} "
          f"(collection now holds {retriever.count()}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
