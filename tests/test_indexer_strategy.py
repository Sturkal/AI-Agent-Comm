from app.data_pipeline.indexer import ChromaIndexer, HashingEmbedder


def test_default_embedding_backend_is_hashing(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)

    indexer = ChromaIndexer(persist_dir=tmp_path, embedding_backend="hashing")

    assert isinstance(indexer.embedder, HashingEmbedder)
    assert indexer.embedding_backend == "hashing"


def test_unknown_embedding_backend_falls_back_to_hashing(tmp_path):
    indexer = ChromaIndexer(persist_dir=tmp_path, embedding_backend="not-a-real-backend")

    assert isinstance(indexer.embedder, HashingEmbedder)
