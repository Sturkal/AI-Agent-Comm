"""Lightweight persistence for self-evaluation learnings."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SelfEvaluationMemoryStore:
    """Persist and retrieve short self-evaluation notes as a local knowledge library."""

    def __init__(self, memory_path: str | Path = "data/processed/self_eval_memory.jsonl") -> None:
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    def read_records(self) -> List[Dict[str, Any]]:
        return self._read_records()

    def export_json(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self._read_records(), indent=2, ensure_ascii=False), encoding="utf-8")
        return output

    def export_jsonl(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in self._read_records():
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        return output

    def render_summary(self, limit: int = 20, question: str = "", xml_hits: Optional[Sequence[Dict[str, Any]]] = None) -> str:
        records = self.find_relevant(question, xml_hits=xml_hits, limit=limit) if question else self._read_records()[-limit:]
        if not records:
            return "No self-evaluation memory entries found."

        lines = [f"Self-evaluation memory entries ({len(records)} shown):"]
        for record in records:
            timestamp = str(record.get("timestamp", "")).strip()
            summary = str(record.get("summary", "")).strip() or "(no summary)"
            confidence = record.get("confidence")
            status = "refine" if record.get("needs_refinement") else "accept"
            parts = [summary, f"status={status}"]
            if confidence is not None:
                parts.append(f"confidence={self._format_confidence(confidence)}")
            if timestamp:
                parts.append(f"timestamp={timestamp}")
            lines.append("- " + " | ".join(parts))
        return "\n".join(lines)

    def find_relevant(
        self,
        question: str,
        xml_hits: Optional[Sequence[Dict[str, Any]]] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        records = self._read_records()
        if not records:
            return []

        query_tokens = set(self._tokenize(question))
        query_tokens.update(self._extract_xml_tokens(xml_hits or []))
        if not query_tokens:
            return records[-limit:]

        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for index, record in enumerate(records):
            record_tokens = set(self._tokenize(self._record_text(record)))
            overlap = len(query_tokens & record_tokens)
            if overlap == 0:
                continue
            recency_bonus = (index + 1) / max(len(records), 1)
            score = overlap + recency_bonus * 0.25
            scored.append((score, index, record))

        if not scored:
            return records[-limit:]

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [item[2] for item in scored[:limit]]
        selected.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return selected

    def format_context(
        self,
        question: str,
        xml_hits: Optional[Sequence[Dict[str, Any]]] = None,
        limit: int = 3,
    ) -> str:
        relevant = self.find_relevant(question, xml_hits=xml_hits, limit=limit)
        if not relevant:
            return ""

        lines = ["SELF-EVALUATION MEMORY:"]
        for item in relevant:
            summary = str(item.get("summary", "")).strip()
            if not summary:
                continue
            confidence = item.get("confidence")
            confidence_text = ""
            if confidence is not None:
                confidence_text = f" (confidence={self._format_confidence(confidence)})"
            lines.append(f"- {summary}{confidence_text}")
        return "\n".join(lines)

    def _read_records(self) -> List[Dict[str, Any]]:
        if not self.memory_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self.memory_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    def _record_text(self, record: Dict[str, Any]) -> str:
        fields = [
            record.get("question", ""),
            record.get("summary", ""),
            " ".join(self._ensure_list(record.get("missing_info"))),
            " ".join(self._ensure_list(record.get("suggested_pdf_terms"))),
            " ".join(self._ensure_list(record.get("xml_terms"))),
        ]
        return " ".join(str(field) for field in fields if field)

    def _extract_xml_tokens(self, xml_hits: Sequence[Dict[str, Any]]) -> List[str]:
        tokens: List[str] = []
        for hit in xml_hits:
            metadata = hit.get("metadata") or {}
            raw_rule = metadata.get("raw_rule")
            if not raw_rule:
                continue
            parsed = self._safe_parse_json(raw_rule)
            if not parsed:
                continue
            tokens.extend(self._tokenize(str(parsed.get("rule_name", ""))))
            tokens.extend(self._tokenize(" ".join(self._ensure_list(parsed.get("variables")))))
            tokens.extend(self._tokenize(" ".join(self._ensure_list(parsed.get("references")))))
        return tokens

    def _safe_parse_json(self, raw_rule: Any) -> Dict[str, Any]:
        if isinstance(raw_rule, dict):
            return raw_rule
        if not isinstance(raw_rule, str) or not raw_rule.strip():
            return {}
        try:
            parsed = json.loads(raw_rule)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _ensure_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        text = str(value).strip()
        return [text] if text else []

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_.]+", text.lower())
        stopwords = {
            "a",
            "an",
            "and",
            "about",
            "answer",
            "because",
            "confidence",
            "could",
            "does",
            "how",
            "is",
            "it",
            "need",
            "not",
            "of",
            "or",
            "the",
            "to",
            "what",
            "when",
            "why",
        }
        return [token for token in tokens if token not in stopwords]

    def _format_confidence(self, value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "0.00"
