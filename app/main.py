"""Flask entry point for the SAP SFIM AI agent."""

from __future__ import annotations

import os
from typing import Optional

try:
    from flask import Flask, jsonify
except Exception:  # pragma: no cover - optional dependency fallback
    Flask = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> None:
        return None

from app.api.routes import register_routes
from app.data_pipeline.indexer import ChromaIndexer
from app.agent.retriever import SFIMRetriever
from app.agent.llm_engine import SFIMLLMEngine
from app.utils.logger import configure_root_logger, get_logger


logger = get_logger(__name__)
configure_root_logger()
load_dotenv()


def create_app() -> "Flask":
    if Flask is None:
        raise RuntimeError("Flask is not installed. Install project dependencies to run the API.")

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    max_request_mb = int(os.getenv("MAX_CONTENT_LENGTH_MB", "2"))
    app.config["MAX_CONTENT_LENGTH"] = max_request_mb * 1024 * 1024

    chroma_dir = os.getenv("CHROMA_DIR", "chromadb_store")

    indexer = ChromaIndexer(persist_dir=chroma_dir)
    app.extensions["sfim_indexer"] = indexer
    if indexer.get_collection().count() == 0:
        logger.warning(
            "Knowledge base is empty. Run `python -m app.cli ingest` before starting the API."
        )

    app.extensions["sfim_engine"] = SFIMLLMEngine(SFIMRetriever(indexer))
    register_routes(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy"})

    @app.get("/ready")
    def ready():
        collection = app.extensions["sfim_indexer"].get_collection()
        return jsonify(
            {
                "status": "ready",
                "knowledge_base_count": collection.count(),
            }
        )

    return app


app: Optional["Flask"]
app = create_app() if Flask is not None else None


if __name__ == "__main__":
    if app is None:
        raise RuntimeError("Flask is not installed. Install project dependencies to run the API.")
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes"},
    )
