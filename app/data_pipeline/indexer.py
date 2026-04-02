"""ChromaDB indexing utilities for PDF chunks and parsed XML rules."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except Exception:  # pragma: no cover - local fallback when chromadb is unavailable
    chromadb = None
    Collection = Any  # type: ignore[assignment]

from app.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class IndexedRecord:
    id: str
    text: str
    metadata: Dict[str, Any]


class OllamaEmbedder:
    """Embedder that calls the Ollama /api/embeddings endpoint."""

    def __init__(self, base_url: str, model: str = "nomic-embed-text") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_one(self, text: str) -> List[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc
        embedding = body.get("embedding")
        if not embedding:
            raise RuntimeError(f"Ollama returned no embedding for model '{self.model}'")
        return embedding

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True) -> List[List[float]]:
        return [self.embed_one(text) for text in texts]


class HashingEmbedder:
    """Fallback embedder that works without external model downloads."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                weight = (int.from_bytes(digest[4:8], "little") % 1000) / 1000.0 + 0.001
                vector[index] += weight
            if normalize_embeddings:
                norm = sum(value * value for value in vector) ** 0.5
                if norm:
                    vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


class LiteCollection:
    """Minimal local fallback when ChromaDB is unavailable."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._items: Dict[str, Dict[str, Any]] = {}
        self._load()

    def count(self) -> int:
        return len(self._items)

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        for idx, item_id in enumerate(ids):
            self._items[str(item_id)] = {
                "document": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "embedding": list(embeddings[idx]) if idx < len(embeddings) else [],
            }
        self._save()

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        query_embedding = list(query_embeddings[0]) if query_embeddings else []
        scored: List[tuple[float, str, Dict[str, Any]]] = []
        for item_id, payload in self._items.items():
            metadata = payload.get("metadata", {})
            if where and any(str(metadata.get(key)) != str(value) for key, value in where.items()):
                continue
            distance = self._cosine_distance(query_embedding, payload.get("embedding", []))
            scored.append((distance, item_id, payload))

        scored.sort(key=lambda entry: entry[0])
        top = scored[:n_results]
        return {
            "ids": [[item_id for _, item_id, _ in top]],
            "documents": [[payload.get("document", "") for _, _, payload in top]],
            "metadatas": [[payload.get("metadata", {}) for _, _, payload in top]],
            "distances": [[distance for distance, _, _ in top]],
        }

    def _cosine_distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 1.0
        size = min(len(left), len(right))
        dot = sum(left[i] * right[i] for i in range(size))
        left_norm = sum(value * value for value in left[:size]) ** 0.5
        right_norm = sum(value * value for value in right[:size]) ** 0.5
        if not left_norm or not right_norm:
            return 1.0
        return 1.0 - (dot / (left_norm * right_norm))

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            self._items = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            self._items = {}

    def _save(self) -> None:
        self.storage_path.write_text(json.dumps(self._items, indent=2, ensure_ascii=False), encoding="utf-8")


class LitePersistentClient:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._collections: Dict[str, LiteCollection] = {}

    def get_or_create_collection(self, name: str) -> LiteCollection:
        if name not in self._collections:
            self._collections[name] = LiteCollection(self.path / f"{name}.json")
        return self._collections[name]


class ChromaIndexer:
    """Persist and query XML rules plus PDF chunks in a single Chroma store."""

    def __init__(
        self,
        persist_dir: str | Path = "chromadb_store",
        collection_name: str = "sfim_knowledge_base",
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir)) if chromadb is not None else LitePersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_backend = (embedding_backend or os.getenv("EMBEDDING_BACKEND", "hashing")).strip().lower()
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embedder = self._load_embedder(self.embedding_backend, self.embedding_model)

    def index_xml_rules(self, parsed_rules: Sequence[Dict[str, Any]]) -> List[str]:
        records: List[IndexedRecord] = []
        for idx, rule in enumerate(parsed_rules):
            rule_name = str(rule.get("rule_name", "")).strip()
            text = self._rule_to_text(rule)
            metadata = self._build_metadata(
                document_type="xml_rule",
                source_type="xml",
                rule_name=rule_name,
                raw_rule=rule,
            )
            records.append(
                IndexedRecord(
                    id=self._make_id("xml_rule", rule_name or f"rule_{idx}", idx),
                    text=text,
                    metadata=metadata,
                )
            )
        return self._upsert(records)

    def index_pdf_chunks(
        self,
        chunks: Sequence[str],
        source_name: str = "unknown.pdf",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        records: List[IndexedRecord] = []
        for idx, chunk in enumerate(chunks):
            metadata = self._build_metadata(
                document_type="pdf",
                source_type="pdf",
                source_name=source_name,
                chunk_index=idx,
                extra_metadata=extra_metadata or {},
            )
            records.append(
                IndexedRecord(
                    id=self._make_id("pdf", Path(source_name).stem or "pdf", idx),
                    text=chunk,
                    metadata=metadata,
                )
            )
        return self._upsert(records)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_args: Dict[str, Any] = {
            "query_embeddings": [self._embed([query])[0]],
            "n_results": top_k,
        }
        if document_type:
            query_args["where"] = {"document_type": document_type}
        results = self.collection.query(**query_args)
        return self._format_query_results(results)

    def get_collection(self) -> Collection:
        return self.collection

    def _upsert(self, records: Sequence[IndexedRecord]) -> List[str]:
        if not records:
            return []
        ids = [record.id for record in records]
        documents = [record.text for record in records]
        metadatas = [record.metadata for record in records]
        embeddings = self._embed(documents)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return ids

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(list(texts), normalize_embeddings=True)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    def _load_embedder(self, embedding_backend: str, embedding_model: str) -> Any:
        if embedding_backend in {"hashing", "lite", "default"}:
            return HashingEmbedder()

        if embedding_backend == "ollama":
            ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            embedder = OllamaEmbedder(base_url=self.ollama_base_url, model=ollama_embed_model)
            logger.info("Using Ollama embedding backend: model=%s url=%s", ollama_embed_model, self.ollama_base_url)
            return embedder

        if embedding_backend in {"sentence-transformers", "sentence_transformers", "transformers", "st"}:
            try:
                from sentence_transformers import SentenceTransformer

                return SentenceTransformer(embedding_model)
            except Exception as exc:
                logger.warning("Falling back to hashing embeddings: %s", exc)
                return HashingEmbedder()

        logger.warning(
            "Unknown embedding backend '%s', falling back to hashing embeddings.",
            embedding_backend,
        )
        return HashingEmbedder()

    def _make_id(self, prefix: str, stem: str, index: int) -> str:
        return f"{prefix}:{stem}:{index}"

    def _rule_to_text(self, rule: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in ("rule_name", "description", "pseudo_code", "formula"):
            value = str(rule.get(key, "")).strip()
            if value:
                parts.append(f"{key}: {value}")
        variables = rule.get("variables") or []
        if variables:
            parts.append(f"variables: {', '.join(map(str, variables))}")
        return "\n".join(parts)

    def _build_metadata(self, **kwargs: Any) -> Dict[str, Any]:
        metadata = dict(kwargs)
        raw_rule = metadata.pop("raw_rule", None)
        extra_metadata = metadata.pop("extra_metadata", None)

        # Chroma metadata values should stay simple and JSON-safe.
        if raw_rule is not None:
            metadata["raw_rule"] = json.dumps(raw_rule, ensure_ascii=False)
        if extra_metadata:
            for key, value in extra_metadata.items():
                metadata[str(key)] = self._stringify_metadata_value(value)

        for key, value in list(metadata.items()):
            metadata[key] = self._stringify_metadata_value(value)
        return metadata

    def _stringify_metadata_value(self, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _format_query_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            formatted.append(
                {
                    "id": doc_id,
                    "document": documents[idx] if idx < len(documents) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                }
            )
        return formatted
