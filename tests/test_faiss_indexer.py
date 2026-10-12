import json
from pathlib import Path

import pytest

from app.ingestion.faiss_indexer import FaissIndexer


def test_indexer_validates_chunk_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FaissIndexer(tmp_path, chunk_size=0)
    with pytest.raises(ValueError):
        FaissIndexer(tmp_path, chunk_size=10, chunk_overlap=10)
    assert FaissIndexer(tmp_path, chunk_size=10, chunk_overlap=2).chunk_size == 10


def test_split_text_handles_empty_short_and_long_text(tmp_path: Path) -> None:
    indexer = FaissIndexer(tmp_path, chunk_size=10, chunk_overlap=2)
    assert indexer._split_text("") == []
    assert indexer._split_text("short") == ["short"]
    chunks = indexer._split_text("one two three four five six seven")
    assert len(chunks) > 1
    assert all(chunks)


def test_load_documents_supports_wrapped_and_list_formats(tmp_path: Path) -> None:
    path = tmp_path / "docs.json"
    indexer = FaissIndexer(tmp_path)
    path.write_text(json.dumps({"documents": [{"page_content": "A"}]}), encoding="utf-8")
    assert len(indexer._load_documents(path)) == 1
    path.write_text(json.dumps([{"page_content": "B"}]), encoding="utf-8")
    assert indexer._load_documents(path)[0]["page_content"] == "B"
    path.write_text(json.dumps({"documents": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError):
        indexer._load_documents(path)
    path.write_text(json.dumps("bad"), encoding="utf-8")
    with pytest.raises(ValueError):
        indexer._load_documents(path)
    with pytest.raises(FileNotFoundError):
        indexer._load_documents(tmp_path / "missing.json")


def test_build_documents_skips_invalid_entries_and_adds_chunk_metadata(tmp_path: Path) -> None:
    indexer = FaissIndexer(tmp_path, chunk_size=5, chunk_overlap=1)
    raw = [
        "invalid",
        {"page_content": "", "metadata": {}},
        {"page_content": "abcdef", "metadata": {"uid": "1"}},
        {"page_content": 123, "metadata": "invalid"},
    ]
    docs = indexer._build_documents(raw)
    assert len(docs) == 3
    assert docs[0].metadata["uid"] == "1"
    assert docs[0].metadata["chunk_count"] == 2
    assert docs[-1].metadata["source_document_index"] == 3


def test_get_embeddings_requires_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr("app.ingestion.faiss_indexer.load_dotenv", lambda **_: None)
    with pytest.raises(OSError, match="MISTRAL_API_KEY"):
        FaissIndexer(tmp_path)._get_embeddings()


def test_create_index_rejects_empty_documents(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FaissIndexer(tmp_path)._create_index([])


def test_save_index_uses_atomic_replacement(tmp_path: Path) -> None:
    indexer = FaissIndexer(tmp_path / "index")

    class Store:
        def save_local(self, path):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "index.faiss").write_text("new", encoding="utf-8")
            (p / "index.pkl").write_text("new", encoding="utf-8")

    indexer._save_index(Store())
    assert (indexer.index_path / "index.faiss").read_text() == "new"
    assert not indexer.index_path.with_name("index_backup").exists()
