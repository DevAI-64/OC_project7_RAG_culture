import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_retrieval import (  # noqa: E402
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

EVALUATION_PATH = ROOT_DIR / "tests" / "fixtures" / "retrieval_eval.json"
DOCUMENTS_PATH = ROOT_DIR / "tests" / "fixtures" / "2026_10_12_documents.json"


def test_precision_recall_and_reciprocal_rank() -> None:
    retrieved = ["TEST002", "TEST004", "TEST001"]
    relevant = {"TEST001", "TEST004"}

    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(1.0)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)


def test_metrics_handle_no_relevant_ids_and_invalid_k() -> None:
    assert recall_at_k([], set(), 5) == 1.0
    with pytest.raises(ValueError):
        precision_at_k([], set(), 0)


def test_evaluation_fixture_ids_exist_in_synthetic_documents() -> None:
    documents = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))["documents"]
    known_ids = {str(item["metadata"]["uid"]) for item in documents}
    payload = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))

    assert payload["reference_date"] == "2026-10-12"
    assert len(payload["questions"]) >= 12
    assert len({item["id"] for item in payload["questions"]}) == len(
        payload["questions"]
    )
    for question in payload["questions"]:
        assert question["expected_ids"]
        assert set(question["expected_ids"]) <= known_ids
