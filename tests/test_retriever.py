import json

from app.agent.llm_engine import SFIMLLMEngine
from app.agent.retriever import SFIMRetriever


class FakeIndexer:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=5, document_type=None):
        self.calls.append((query, top_k, document_type))
        return [
            {
                "id": f"{document_type}:1",
                "document": f"{document_type} doc",
                "metadata": {},
                "distance": 0.1,
            }
        ]


class FakeRetriever:
    def __init__(self):
        self.xml_calls = []
        self.pdf_calls = []

    def search_xml_rules(self, query, top_k=5):
        self.xml_calls.append((query, top_k))
        return [
            {
                "id": "xml_rule:1",
                "document": "xml rule text",
                "metadata": {
                    "raw_rule": json.dumps(
                        {
                            "variables": ["SalesTransaction.value", "Position.status"],
                            "references": ["F_Example"],
                        }
                    )
                },
                "distance": 0.1,
            }
        ]

    def search_pdf_docs(self, query, top_k=5):
        self.pdf_calls.append((query, top_k))
        return [
            {
                "id": "pdf:1",
                "document": "pdf doc text",
                "metadata": {},
                "distance": 0.2,
            }
        ]


def test_retriever_filters_by_document_type():
    indexer = FakeIndexer()
    retriever = SFIMRetriever(indexer=indexer)

    pdf_hits = retriever.search_pdf_docs("premium")
    xml_hits = retriever.search_xml_rules("formula")

    assert indexer.calls[0][2] == "pdf"
    assert indexer.calls[1][2] == "xml_rule"
    assert pdf_hits[0]["id"] == "pdf:1"
    assert xml_hits[0]["id"] == "xml_rule:1"


def test_extract_rule_context_parses_raw_rule_json():
    retriever = SFIMRetriever(indexer=FakeIndexer())
    results = [
        {
            "id": "xml_rule:1",
            "document": "xml rule text",
            "metadata": {"raw_rule": json.dumps({"rule_name": "Rule_A"})},
            "distance": 0.1,
        }
    ]

    extracted = retriever.extract_rule_context(results)

    assert extracted[0]["raw_rule"]["rule_name"] == "Rule_A"
    assert extracted[0]["text"] == "xml rule text"


def test_engine_prefers_xml_then_expands_pdf_query(monkeypatch):
    fake_retriever = FakeRetriever()
    engine = SFIMLLMEngine(retriever=fake_retriever)

    monkeypatch.setattr(engine, "_generate_answer", lambda *args, **kwargs: "stub answer")

    response = engine.answer("How is Rule A calculated?")

    assert response.answer == "stub answer"
    assert fake_retriever.xml_calls == [("How is Rule A calculated?", 5)]
    assert fake_retriever.pdf_calls[0][0].startswith("How is Rule A calculated?")
    assert "SalesTransaction.value" in fake_retriever.pdf_calls[0][0]
    assert "Position.status" in fake_retriever.pdf_calls[0][0]
    assert "F_Example" in fake_retriever.pdf_calls[0][0]


def test_engine_uses_pdf_when_no_xml_hits(monkeypatch):
    class EmptyXmlRetriever(FakeRetriever):
        def search_xml_rules(self, query, top_k=5):
            self.xml_calls.append((query, top_k))
            return []

    fake_retriever = EmptyXmlRetriever()
    engine = SFIMLLMEngine(retriever=fake_retriever)
    monkeypatch.setattr(engine, "_generate_answer", lambda *args, **kwargs: "stub answer")

    engine.answer("What does the guide say about payout?")

    assert fake_retriever.xml_calls == [("What does the guide say about payout?", 3)]
    assert fake_retriever.pdf_calls == [("What does the guide say about payout?", 5)]

