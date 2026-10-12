import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT_DIR / "tests" / "fixtures" / "2026_10_12_documents.json"
DEFAULT_INDEX = ROOT_DIR / "data" / "processed" / "faiss_test_index"


def test_synthetic_documents_fixture_matches_processed_snapshot_contract() -> None:
    payload = json.loads(DEFAULT_DOCUMENTS.read_text(encoding="utf-8"))

    assert set(payload) == {"metadata", "documents"}
    assert payload["metadata"]["fixture"] is True
    assert payload["metadata"]["city"] == "Pau"
    assert len(payload["documents"]) >= 15

    for document in payload["documents"]:
        assert isinstance(document["id"], str)
        assert isinstance(document["page_content"], str) and document["page_content"]
        metadata = document["metadata"]
        assert metadata["uid"] == document["id"]
        assert metadata["title"]
        assert metadata["first_date"]
        assert metadata["last_date"]
        assert "price_status" in metadata
        assert "price_amounts_eur" in metadata
        assert "audience_categories" in metadata
        assert "timings" in metadata


def test_test_index_is_separate_from_production_index() -> None:
    assert DEFAULT_INDEX.name == "faiss_test_index"
    assert DEFAULT_INDEX != ROOT_DIR / "data" / "processed" / "faiss_index"
