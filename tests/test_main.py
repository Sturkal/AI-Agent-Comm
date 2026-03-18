from types import SimpleNamespace

import pytest

flask = pytest.importorskip("flask")

import app.main as main


class FakeCollection:
    def __init__(self, count_value=0):
        self._count_value = count_value

    def count(self):
        return self._count_value


class FakeIndexer:
    def __init__(self, persist_dir):
        self.persist_dir = persist_dir
        self.collection = FakeCollection(count_value=0)

    def get_collection(self):
        return self.collection


class FakeRetriever:
    def __init__(self, indexer):
        self.indexer = indexer


class FakeEngine:
    def __init__(self, retriever):
        self.retriever = retriever


def test_create_app_does_not_ingest_on_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ChromaIndexer", FakeIndexer)
    monkeypatch.setattr(main, "SFIMRetriever", FakeRetriever)
    monkeypatch.setattr(main, "SFIMLLMEngine", FakeEngine)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))

    app = main.create_app()

    assert app.extensions["sfim_engine"].retriever.indexer.persist_dir == str(tmp_path / "chroma")
    assert any(rule.rule == "/health" for rule in app.url_map.iter_rules())
    assert any(rule.rule == "/ready" for rule in app.url_map.iter_rules())
    assert any(rule.rule == "/webhook/whatsapp" for rule in app.url_map.iter_rules())


def test_ready_route_reports_collection_count(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ChromaIndexer", FakeIndexer)
    monkeypatch.setattr(main, "SFIMRetriever", FakeRetriever)
    monkeypatch.setattr(main, "SFIMLLMEngine", FakeEngine)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))

    app = main.create_app()
    client = app.test_client()

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ready"
    assert "knowledge_base_count" in payload
