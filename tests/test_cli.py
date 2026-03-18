from types import SimpleNamespace

import app.cli as cli


def test_run_ingest_uses_indexer_and_ingestor(monkeypatch, tmp_path, capsys):
    calls = {}

    class FakeIndexer:
        def __init__(self, persist_dir):
            calls["persist_dir"] = persist_dir

    class FakeIngestor:
        def __init__(self, raw_dir, processed_dir, indexer):
            calls["raw_dir"] = raw_dir
            calls["processed_dir"] = processed_dir
            calls["indexer"] = indexer

        def ingest_all(self):
            calls["ingested"] = True
            return {"xml_records": 2, "pdf_chunks": 3}

    monkeypatch.setattr(cli, "ChromaIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "RawDataIngestor", FakeIngestor)

    args = SimpleNamespace(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        chroma_dir=str(tmp_path / "chroma"),
    )

    cli.run_ingest(args)

    captured = capsys.readouterr()
    assert calls["persist_dir"] == str(tmp_path / "chroma")
    assert calls["raw_dir"] == str(tmp_path / "raw")
    assert calls["processed_dir"] == str(tmp_path / "processed")
    assert calls["ingested"] is True
    assert "xml_records" in captured.out


def test_main_dispatches_ingest(monkeypatch):
    called = {}

    def fake_run_ingest(args):
        called["args"] = args

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)

    cli.main(["ingest", "--raw-dir", "x", "--processed-dir", "y", "--chroma-dir", "z"])

    assert called["args"].raw_dir == "x"
    assert called["args"].processed_dir == "y"
    assert called["args"].chroma_dir == "z"


def test_run_query_prints_answer_and_hits(monkeypatch, capsys):
    calls = {}

    class FakeIndexer:
        def __init__(self, persist_dir):
            calls["persist_dir"] = persist_dir

    class FakeResult:
        def __init__(self):
            self.answer = "Rule A is calculated from SalesTransaction.value."
            self.xml_hits = [{"id": "xml_rule:1", "document": "xml rule text"}]
            self.pdf_hits = [{"id": "pdf:1", "document": "pdf doc text"}]
            self.self_evaluation_summary = "Round 1: enough evidence from XML/PDF to answer. confidence=0.91"
            self.self_evaluation = {"confidence": 0.91, "needs_refinement": False}
            self.refinement_memory = [{"summary": "Round 0: prior note"}]

    class FakeEngine:
        def __init__(self, retriever):
            calls["retriever"] = retriever

        def answer(self, question, user_phone_number=None):
            calls["question"] = question
            calls["phone_number"] = user_phone_number
            return FakeResult()

    class FakeRetriever:
        def __init__(self, indexer):
            calls["indexer"] = indexer

    monkeypatch.setattr(cli, "ChromaIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "SFIMRetriever", FakeRetriever)
    monkeypatch.setattr(cli, "SFIMLLMEngine", FakeEngine)

    args = SimpleNamespace(
        question="How is Rule A calculated?",
        phone_number="+10000000000",
        chroma_dir="kb",
        show_hits=True,
    )

    cli.run_query(args)

    captured = capsys.readouterr()
    assert calls["persist_dir"] == "kb"
    assert calls["question"] == "How is Rule A calculated?"
    assert calls["phone_number"] == "+10000000000"
    assert "Rule A is calculated" in captured.out
    assert "--- XML Hits ---" in captured.out
    assert "--- Self Evaluation ---" in captured.out
    assert "prior note" in captured.out


def test_main_dispatches_query(monkeypatch):
    called = {}

    def fake_run_query(args):
        called["args"] = args

    monkeypatch.setattr(cli, "run_query", fake_run_query)

    cli.main(["query", "How is Rule A calculated?", "--phone-number", "+1", "--chroma-dir", "z"])

    assert called["args"].question == "How is Rule A calculated?"
    assert called["args"].phone_number == "+1"
    assert called["args"].chroma_dir == "z"


def test_run_memory_show_prints_summary(tmp_path, capsys):
    memory_path = tmp_path / "self_eval_memory.jsonl"
    memory_path.write_text(
        "\n".join(
            [
                '{"timestamp":"2026-03-19T00:00:00Z","summary":"Need more PDF context","confidence":0.42,"needs_refinement":true}',
                '{"timestamp":"2026-03-19T00:01:00Z","summary":"Good XML match","confidence":0.91,"needs_refinement":false}',
            ]
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(memory_path=str(memory_path), limit=10, question="")
    cli.run_memory_show(args)

    captured = capsys.readouterr()
    assert "Self-evaluation memory entries" in captured.out
    assert "Good XML match" in captured.out


def test_run_memory_export_writes_output(tmp_path, capsys):
    memory_path = tmp_path / "self_eval_memory.jsonl"
    memory_path.write_text(
        '{"timestamp":"2026-03-19T00:00:00Z","summary":"Need more PDF context","confidence":0.42,"needs_refinement":true}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "exports" / "memory.json"

    args = SimpleNamespace(memory_path=str(memory_path), output=str(output_path), format="json")
    cli.run_memory_export(args)

    captured = capsys.readouterr()
    assert output_path.exists()
    assert "memory.json" in captured.out
    assert "Need more PDF context" in output_path.read_text(encoding="utf-8")


def test_main_dispatches_memory_show(monkeypatch):
    called = {}

    def fake_run_memory_show(args):
        called["args"] = args

    monkeypatch.setattr(cli, "run_memory_show", fake_run_memory_show)

    cli.main(["memory", "show", "--question", "How is Rule A calculated?"])

    assert called["args"].question == "How is Rule A calculated?"


def test_main_dispatches_memory_export(monkeypatch):
    called = {}

    def fake_run_memory_export(args):
        called["args"] = args

    monkeypatch.setattr(cli, "run_memory_export", fake_run_memory_export)

    cli.main(["memory", "export", "--output", "out.json"])

    assert called["args"].output == "out.json"
