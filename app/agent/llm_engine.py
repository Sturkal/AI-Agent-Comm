"""ReAct-style answering engine for SFIM questions with self-evaluation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.agent.refinement_memory import SelfEvaluationMemoryStore
from app.agent.retriever import SFIMRetriever
from app.data_pipeline.indexer import ChromaIndexer
from app.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class AgentResponse:
    answer: str
    xml_hits: List[Dict[str, Any]]
    pdf_hits: List[Dict[str, Any]]
    self_evaluation_summary: str = ""
    self_evaluation: Dict[str, Any] = field(default_factory=dict)
    refinement_memory: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnswerEvaluation:
    confidence: float
    needs_refinement: bool
    missing_info: List[str]
    suggested_pdf_terms: List[str]
    notes: str = ""


class SFIMLLMEngine:
    """Lightweight ReAct-style engine with XML-first routing and self-checks."""

    def __init__(self, retriever: Optional[SFIMRetriever] = None) -> None:
        self.retriever = retriever or SFIMRetriever(ChromaIndexer())
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
        self._has_openai = bool(os.getenv("OPENAI_API_KEY"))
        self.max_refinement_rounds = int(os.getenv("SFIM_MAX_REFINEMENT_ROUNDS", "1"))
        self.min_confidence = float(os.getenv("SFIM_MIN_CONFIDENCE", "0.70"))
        self.memory_store = SelfEvaluationMemoryStore(
            os.getenv("SFIM_SELF_EVAL_MEMORY_PATH", "data/processed/self_eval_memory.jsonl")
        )
        self._last_generation_snapshot: Dict[str, Any] = {
            "summary": "",
            "evaluation": {},
            "memory": [],
        }

    def answer(self, question: str, user_phone_number: Optional[str] = None) -> AgentResponse:
        xml_hits = self._search_xml_first(question)
        pdf_hits = self._search_pdf_if_needed(question, xml_hits)
        refinement_memory = self.memory_store.format_context(question, xml_hits=xml_hits, limit=3)
        answer = self._generate_answer(
            question,
            xml_hits,
            pdf_hits,
            user_phone_number,
            refinement_memory=refinement_memory,
        )
        snapshot = dict(self._last_generation_snapshot)
        return AgentResponse(
            answer=answer,
            xml_hits=xml_hits,
            pdf_hits=pdf_hits,
            self_evaluation_summary=str(snapshot.get("summary", "")).strip(),
            self_evaluation=dict(snapshot.get("evaluation") or {}),
            refinement_memory=list(snapshot.get("memory") or []),
        )

    def _search_xml_first(self, question: str) -> List[Dict[str, Any]]:
        should_prioritize_xml = self._is_rule_question(question)
        if should_prioritize_xml:
            hits = self.retriever.search_xml_rules(question, top_k=5)
            if hits:
                return hits
        return self.retriever.search_xml_rules(question, top_k=3)

    def _search_pdf_if_needed(
        self,
        question: str,
        xml_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not xml_hits:
            return self.retriever.search_pdf_docs(question, top_k=5)

        if self._is_rule_question(question):
            expanded_query = self._build_variable_query(question, xml_hits)
            if expanded_query:
                return self.retriever.search_pdf_docs(expanded_query, top_k=5)

        return []

    def _generate_answer(
        self,
        question: str,
        xml_hits: List[Dict[str, Any]],
        pdf_hits: List[Dict[str, Any]],
        user_phone_number: Optional[str],
        refinement_memory: str = "",
    ) -> str:
        self._last_generation_snapshot = {
            "summary": "",
            "evaluation": {},
            "memory": self._memory_context_to_items(refinement_memory),
        }
        current_pdf_hits = list(pdf_hits)

        for round_index in range(self.max_refinement_rounds + 1):
            draft = self._generate_candidate_answer(
                question,
                xml_hits,
                current_pdf_hits,
                user_phone_number,
                refinement_memory=refinement_memory,
            )
            evaluation = self._self_evaluate_answer(
                question=question,
                candidate_answer=draft,
                xml_hits=xml_hits,
                pdf_hits=current_pdf_hits,
                user_phone_number=user_phone_number,
                refinement_memory=refinement_memory,
            )
            summary = self._build_self_evaluation_summary(round_index, evaluation, current_pdf_hits, xml_hits)
            self._store_generation_snapshot(summary, evaluation, refinement_memory)
            self._remember_self_evaluation(
                question=question,
                candidate_answer=draft,
                evaluation=evaluation,
                round_index=round_index,
                xml_hits=xml_hits,
                pdf_hits=current_pdf_hits,
            )

            if not evaluation.needs_refinement:
                return draft

            if round_index >= self.max_refinement_rounds:
                return draft

            refined_pdf_hits = self._expand_pdf_hits(question, xml_hits, current_pdf_hits, evaluation)
            if refined_pdf_hits != current_pdf_hits:
                current_pdf_hits = refined_pdf_hits
                continue

            return draft

        return self._template_answer(question, xml_hits, current_pdf_hits)

    def _generate_candidate_answer(
        self,
        question: str,
        xml_hits: List[Dict[str, Any]],
        pdf_hits: List[Dict[str, Any]],
        user_phone_number: Optional[str],
        refinement_memory: str = "",
    ) -> str:
        context = self._format_context(xml_hits, pdf_hits)
        memory_context = refinement_memory.strip()
        try:
            return self._ask_ollama(
                system_prompt=(
                    "You are an SFIM expert assistant. Use the XML rule context first, "
                    "then the PDF context to explain variables and business meaning. "
                    "If the rule is not fully found, say what is missing. Keep answers concise, "
                    "clear, and suitable for WhatsApp. If self-evaluation memory is present, "
                    "use it to avoid repeating previous weaknesses."
                ),
                user_prompt=(
                    f"User phone: {user_phone_number or 'unknown'}\n"
                    f"Question: {question}\n\n"
                    f"{memory_context + '\\n\\n' if memory_context else ''}"
                    f"Context:\n{context or 'No context retrieved.'}"
                ),
            )
        except Exception as exc:
            logger.warning("Ollama generation failed, trying fallback: %s", exc)
        if self._has_openai:
            try:
                return self._ask_openai(question, context, user_phone_number)
            except Exception as exc:
                logger.warning("OpenAI generation failed, falling back to template: %s", exc)
        return self._template_answer(question, xml_hits, pdf_hits)

    def _self_evaluate_answer(
        self,
        question: str,
        candidate_answer: str,
        xml_hits: List[Dict[str, Any]],
        pdf_hits: List[Dict[str, Any]],
        user_phone_number: Optional[str],
        refinement_memory: str = "",
    ) -> AnswerEvaluation:
        context = self._format_context(xml_hits, pdf_hits)
        system_prompt = (
            "You are evaluating an SFIM assistant draft answer. Return strict JSON with keys: "
            "`confidence` (0 to 1), `needs_refinement` (boolean), `missing_info` (array of strings), "
            "`suggested_pdf_terms` (array of strings), and `notes` (short string). "
            "Mark `needs_refinement` true when the draft is not sufficiently grounded in the XML/PDF context, "
            "when important variables are not explained, or when the answer is too vague. "
            "Keep notes short so they can be stored as reusable memory."
        )
        user_prompt = (
            f"User phone: {user_phone_number or 'unknown'}\n"
            f"Question: {question}\n\n"
            f"Draft answer:\n{candidate_answer}\n\n"
            f"{refinement_memory + '\\n\\n' if refinement_memory else ''}"
            f"Context:\n{context or 'No context retrieved.'}"
        )

        try:
            raw_output = self._call_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
            payload = self._parse_json_payload(raw_output)
            confidence = self._safe_float(payload.get("confidence"), default=0.0)
            needs_refinement = bool(payload.get("needs_refinement"))
            missing_info = self._coerce_str_list(payload.get("missing_info"))
            suggested_pdf_terms = self._coerce_str_list(payload.get("suggested_pdf_terms"))
            notes = str(payload.get("notes", "")).strip()

            if confidence < self.min_confidence:
                needs_refinement = True

            return AnswerEvaluation(
                confidence=confidence,
                needs_refinement=needs_refinement,
                missing_info=missing_info,
                suggested_pdf_terms=suggested_pdf_terms,
                notes=notes,
            )
        except Exception as exc:
            logger.warning("Self-evaluation failed, falling back to heuristic: %s", exc)
            return self._heuristic_evaluation(question, candidate_answer, xml_hits, pdf_hits)

    def _expand_pdf_hits(
        self,
        question: str,
        xml_hits: List[Dict[str, Any]],
        current_pdf_hits: List[Dict[str, Any]],
        evaluation: AnswerEvaluation,
    ) -> List[Dict[str, Any]]:
        if not evaluation.suggested_pdf_terms:
            return current_pdf_hits

        refined_query = self._build_variable_query(
            question,
            xml_hits,
            extra_terms=evaluation.suggested_pdf_terms,
        )
        extra_pdf_hits = self.retriever.search_pdf_docs(refined_query, top_k=5)
        return self._merge_hits(current_pdf_hits, extra_pdf_hits)

    def _build_self_evaluation_summary(
        self,
        round_index: int,
        evaluation: AnswerEvaluation,
        pdf_hits: List[Dict[str, Any]],
        xml_hits: List[Dict[str, Any]],
    ) -> str:
        confidence_text = f"{evaluation.confidence:.2f}"
        if evaluation.needs_refinement:
            missing = "; ".join(evaluation.missing_info[:2]) if evaluation.missing_info else "missing support"
            suggested = ", ".join(evaluation.suggested_pdf_terms[:3]) if evaluation.suggested_pdf_terms else ""
            if suggested:
                return (
                    f"Round {round_index + 1}: needs refinement because {missing}; "
                    f"search terms to add: {suggested}. confidence={confidence_text}"
                )
            return f"Round {round_index + 1}: needs refinement because {missing}. confidence={confidence_text}"

        if evaluation.notes:
            return f"Round {round_index + 1}: answer is sufficiently grounded; {evaluation.notes}. confidence={confidence_text}"
        if xml_hits or pdf_hits:
            return f"Round {round_index + 1}: enough evidence from XML/PDF to answer. confidence={confidence_text}"
        return f"Round {round_index + 1}: no strong evidence found, but confidence={confidence_text}"

    def _store_generation_snapshot(
        self,
        summary: str,
        evaluation: AnswerEvaluation,
        refinement_memory: str,
    ) -> None:
        self._last_generation_snapshot = {
            "summary": summary,
            "evaluation": {
                "confidence": evaluation.confidence,
                "needs_refinement": evaluation.needs_refinement,
                "missing_info": list(evaluation.missing_info),
                "suggested_pdf_terms": list(evaluation.suggested_pdf_terms),
                "notes": evaluation.notes,
            },
            "memory": self._memory_context_to_items(refinement_memory),
        }

    def _remember_self_evaluation(
        self,
        question: str,
        candidate_answer: str,
        evaluation: AnswerEvaluation,
        round_index: int,
        xml_hits: List[Dict[str, Any]],
        pdf_hits: List[Dict[str, Any]],
    ) -> None:
        summary = self._last_generation_snapshot.get("summary", "")
        xml_terms = self._extract_rule_terms(xml_hits)
        record = {
            "question": question,
            "round_index": round_index,
            "summary": summary,
            "confidence": evaluation.confidence,
            "needs_refinement": evaluation.needs_refinement,
            "missing_info": list(evaluation.missing_info),
            "suggested_pdf_terms": list(evaluation.suggested_pdf_terms),
            "notes": evaluation.notes,
            "xml_terms": xml_terms,
            "xml_hit_ids": [str(hit.get("id", "")) for hit in xml_hits if hit.get("id")],
            "pdf_hit_ids": [str(hit.get("id", "")) for hit in pdf_hits if hit.get("id")],
            "candidate_excerpt": candidate_answer[:500],
        }
        self.memory_store.append(record)

    def _build_variable_query(
        self,
        question: str,
        xml_hits: List[Dict[str, Any]],
        extra_terms: Optional[List[str]] = None,
    ) -> str:
        variables: List[str] = []
        for hit in xml_hits:
            metadata = hit.get("metadata") or {}
            raw_rule = metadata.get("raw_rule")
            if not raw_rule:
                continue
            try:
                parsed = json.loads(raw_rule) if isinstance(raw_rule, str) else raw_rule
            except Exception:
                continue
            variables.extend([str(v) for v in parsed.get("variables", []) if v])
            variables.extend([str(v) for v in parsed.get("references", []) if v])

        terms = list(dict.fromkeys(variables + (extra_terms or [])))
        if not terms:
            return question
        return f"{question} {' '.join(terms)}"

    def _memory_context_to_items(self, refinement_memory: str) -> List[Dict[str, Any]]:
        if not refinement_memory.strip():
            return []
        items: List[Dict[str, Any]] = []
        for line in refinement_memory.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            items.append({"summary": line[2:].strip()})
        return items

    def _extract_rule_terms(self, xml_hits: List[Dict[str, Any]]) -> List[str]:
        terms: List[str] = []
        for hit in xml_hits:
            metadata = hit.get("metadata") or {}
            raw_rule = metadata.get("raw_rule")
            if not raw_rule:
                continue
            try:
                parsed = json.loads(raw_rule) if isinstance(raw_rule, str) else raw_rule
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            terms.extend(self._coerce_str_list(parsed.get("rule_name")))
            terms.extend(self._coerce_str_list(parsed.get("variables")))
            terms.extend(self._coerce_str_list(parsed.get("references")))
        return list(dict.fromkeys(term for term in terms if term))

    def _ask_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        request = Request(
            f"{self.ollama_base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc.code} {exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.ollama_base_url}") from exc

        message = body.get("message") or {}
        content = message.get("content")
        if content:
            return str(content).strip()
        return str(body.get("response", "")).strip()

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Backward-compatible wrapper for callers that still expect _call_ollama."""
        return self._ask_ollama(system_prompt=system_prompt, user_prompt=user_prompt)

    def _ask_openai(self, question: str, context: str, user_phone_number: Optional[str]) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an SFIM expert assistant. Use the provided XML rule context "
                        "first, then PDF context for explanation. Keep answers concise, "
                        "practical, and grounded in the retrieved evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User phone: {user_phone_number or 'unknown'}\nQuestion: {question}\n\nContext:\n{context}",
                },
            ],
        )
        return response.choices[0].message.content or ""

    def _parse_json_payload(self, raw_output: str) -> Dict[str, Any]:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _heuristic_evaluation(
        self,
        question: str,
        candidate_answer: str,
        xml_hits: List[Dict[str, Any]],
        pdf_hits: List[Dict[str, Any]],
    ) -> AnswerEvaluation:
        lowered = candidate_answer.lower()
        vague_signals = ("could not find", "not enough", "unclear", "missing", "need more")
        needs_refinement = any(signal in lowered for signal in vague_signals)
        confidence = 0.8 if xml_hits else 0.5
        if pdf_hits:
            confidence += 0.1
        if needs_refinement:
            confidence = min(confidence, 0.4)
        if not xml_hits:
            needs_refinement = True

        suggested_terms: List[str] = []
        if not pdf_hits:
            suggested_terms.extend(self._extract_terms_from_question(question))

        return AnswerEvaluation(
            confidence=min(confidence, 0.95),
            needs_refinement=needs_refinement or confidence < self.min_confidence,
            missing_info=[] if not needs_refinement else ["Answer may be too vague or under-supported."],
            suggested_pdf_terms=suggested_terms,
            notes="heuristic fallback",
        )

    def _merge_hits(
        self,
        first_hits: List[Dict[str, Any]],
        second_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for hit in list(first_hits) + list(second_hits):
            hit_id = str(hit.get("id", ""))
            if hit_id and hit_id not in seen:
                seen.add(hit_id)
                merged.append(hit)
        return merged

    def _coerce_str_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in re.split(r"[,\n;]", value) if part.strip()]
        text = str(value).strip()
        return [text] if text else []

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _extract_terms_from_question(self, question: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_.]+", question)
        stopwords = {
            "how",
            "is",
            "the",
            "a",
            "an",
            "what",
            "does",
            "say",
            "about",
            "for",
            "to",
            "and",
            "or",
            "rule",
            "calculated",
            "calculate",
            "formula",
        }
        return [token for token in tokens if token.lower() not in stopwords]

    def _template_answer(self, question: str, xml_hits: List[Dict[str, Any]], pdf_hits: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        lines.append(f"Question: {question}")
        if xml_hits:
            lines.append("Relevant XML rules:")
            for hit in xml_hits[:3]:
                lines.append(f"- {hit.get('document', '').strip()}")
        if pdf_hits:
            lines.append("Relevant documentation:")
            for hit in pdf_hits[:3]:
                lines.append(f"- {hit.get('document', '').strip()}")
        if not xml_hits and not pdf_hits:
            lines.append("I could not find a strong match in the current knowledge base.")
        return "\n".join(lines)

    def _format_context(self, xml_hits: List[Dict[str, Any]], pdf_hits: List[Dict[str, Any]]) -> str:
        sections: List[str] = []
        if xml_hits:
            sections.append("XML RULES:\n" + "\n\n".join(hit.get("document", "") for hit in xml_hits[:5]))
        if pdf_hits:
            sections.append("PDF DOCS:\n" + "\n\n".join(hit.get("document", "") for hit in pdf_hits[:5]))
        return "\n\n".join(sections)

    def _is_rule_question(self, question: str) -> bool:
        lowered = question.lower()
        return any(
            phrase in lowered
            for phrase in (
                "how is",
                "how do",
                "rule",
                "formula",
                "calculated",
                "calculate",
                "computed",
                "computation",
            )
        )


def build_default_engine() -> SFIMLLMEngine:
    return SFIMLLMEngine()
