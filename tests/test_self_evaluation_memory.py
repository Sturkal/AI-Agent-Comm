import json

from app.agent.llm_engine import SFIMLLMEngine


class MemoryAwareRetriever:
    def search_xml_rules(self, query, top_k=5):
        return [
            {
                "id": "xml_rule:1",
                "document": "Rule A uses SalesTransaction.value.",
                "metadata": {
                    "raw_rule": json.dumps(
                        {
                            "rule_name": "Rule_A",
                            "variables": ["SalesTransaction.value"],
                            "references": ["F_A"],
                        }
                    )
                },
                "distance": 0.1,
            }
        ]

    def search_pdf_docs(self, query, top_k=5):
        return [
            {
                "id": "pdf:1",
                "document": "SalesTransaction.value is the transaction amount.",
                "metadata": {},
                "distance": 0.2,
            }
        ]


def test_self_evaluation_summary_is_persisted_and_reused(tmp_path, monkeypatch):
    memory_path = tmp_path / "self_eval_memory.jsonl"
    monkeypatch.setenv("SFIM_SELF_EVAL_MEMORY_PATH", str(memory_path))
    monkeypatch.setenv("SFIM_MAX_REFINEMENT_ROUNDS", "0")
    monkeypatch.setenv("SFIM_MIN_CONFIDENCE", "0.70")

    engine = SFIMLLMEngine(retriever=MemoryAwareRetriever())

    monkeypatch.setattr(
        engine,
        "_ask_ollama",
        lambda system_prompt, user_prompt: "Rule A is calculated using SalesTransaction.value.",
    )
    monkeypatch.setattr(
        engine,
        "_call_ollama",
        lambda system_prompt, user_prompt: json.dumps(
            {
                "confidence": 0.94,
                "needs_refinement": False,
                "missing_info": [],
                "suggested_pdf_terms": [],
                "notes": "clear and grounded",
            }
        ),
    )

    first = engine.answer("How is Rule A calculated?")

    assert first.self_evaluation_summary.startswith("Round 1:")
    assert first.self_evaluation["confidence"] == 0.94
    assert memory_path.exists()
    assert "clear and grounded" in memory_path.read_text(encoding="utf-8")

    memory_context = engine.memory_store.format_context("How is Rule A calculated?", xml_hits=first.xml_hits)
    assert "SELF-EVALUATION MEMORY:" in memory_context
    assert "clear and grounded" in memory_context
