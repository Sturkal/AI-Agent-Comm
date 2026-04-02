from unittest.mock import patch

from app.data_pipeline.indexer import ChromaIndexer, HashingEmbedder, OllamaEmbedder


def test_default_embedding_backend_is_hashing(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)

    indexer = ChromaIndexer(persist_dir=tmp_path, embedding_backend="hashing")

    assert isinstance(indexer.embedder, HashingEmbedder)
    assert indexer.embedding_backend == "hashing"


def test_unknown_embedding_backend_falls_back_to_hashing(tmp_path):
    indexer = ChromaIndexer(persist_dir=tmp_path, embedding_backend="not-a-real-backend")

    assert isinstance(indexer.embedder, HashingEmbedder)


def test_ollama_embedding_backend_selects_ollama_embedder(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    indexer = ChromaIndexer(persist_dir=tmp_path, embedding_backend="ollama")

    assert isinstance(indexer.embedder, OllamaEmbedder)
    assert indexer.embedder.model == "nomic-embed-text"


def test_ollama_embedder_encodes_texts(monkeypatch):
    fake_embedding = [0.1] * 768
    fake_response = b'{"embedding": ' + str(fake_embedding).replace(" ", "").encode() + b"}"

    class FakeResponse:
        def read(self):
            return fake_response

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch("app.data_pipeline.indexer.urlopen", return_value=FakeResponse()):
        embedder = OllamaEmbedder(base_url="http://localhost:11434", model="nomic-embed-text")
        result = embedder.encode(["test text"])

    assert len(result) == 1
    assert len(result[0]) == 768


def test_ollama_embedding_backend_falls_through_to_embedder_on_call(tmp_path, monkeypatch):
    """OllamaEmbedder is created without actually calling Ollama at init time."""
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Should not raise even if Ollama is unreachable — embedder is only called at index/search time
    indexer = ChromaIndexer(persist_dir=tmp_path, embedding_backend="ollama")
    assert isinstance(indexer.embedder, OllamaEmbedder)
