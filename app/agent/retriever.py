"""Retrieval helpers that query PDF docs and XML rules from ChromaDB."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.data_pipeline.indexer import ChromaIndexer


class SFIMRetriever:
    def __init__(self, indexer: Optional[ChromaIndexer] = None) -> None:
        self.indexer = indexer or ChromaIndexer()

    def search_pdf_docs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.indexer.search(query=query, top_k=top_k, document_type="pdf")

    def search_xml_rules(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.indexer.search(query=query, top_k=top_k, document_type="xml_rule")

    def extract_rule_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        extracted: List[Dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata") or {}
            raw_rule = metadata.get("raw_rule")
            parsed_rule: Dict[str, Any] = {}
            if isinstance(raw_rule, str) and raw_rule:
                try:
                    parsed_rule = json.loads(raw_rule)
                except json.JSONDecodeError:
                    parsed_rule = {}
            extracted.append(
                {
                    "id": item.get("id", ""),
                    "text": item.get("document", ""),
                    "distance": item.get("distance"),
                    "metadata": metadata,
                    "raw_rule": parsed_rule,
                }
            )
        return extracted

