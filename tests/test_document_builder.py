import pytest

from app.domain.event import Event
from app.ingestion.document_builder import DocumentBuilder


def test_builder_requires_summarizer_when_enabled() -> None:
    with pytest.raises(ValueError):
        DocumentBuilder(use_summarization=True)


def test_text_helpers_and_classification() -> None:
    builder = DocumentBuilder()
    assert builder._clean_text("<p>Bonjour</p>   le monde") == "Bonjour le monde"
    assert builder._normalize("  Bonjour   MONDE ") == "bonjour monde"
    assert builder._unique(["a", "b", "a"]) == ["a", "b"]
    assert builder._matches_any("atelier enfant", [r"enfant"])
    assert builder._classify_audience("atelier pour enfants et familles") == [
        "enfants",
        "familles",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dès 8 ans", (8, None)),
        ("de 8 à 12 ans", (8, 12)),
        ("entre 10 et 15 ans", (10, 15)),
        ("jusqu'à 12 ans", (None, 12)),
        ("moins de 14 ans", (None, 13)),
        ("tout public", (None, None)),
    ],
)
def test_builder_extracts_age_ranges(text, expected) -> None:
    assert DocumentBuilder()._extract_age_from_text(text) == expected


def test_builder_prefers_structured_age_information(sample_event: Event) -> None:
    result = DocumentBuilder()._classify_age(sample_event, "dès 5 ans")
    assert result == {"age_min": 8, "age_max": 12, "age_source": "event"}


def test_builder_uses_text_age_as_fallback() -> None:
    event = Event(id="1", title="Test")
    result = DocumentBuilder()._classify_age(event, "atelier dès 6 ans")
    assert result == {"age_min": 6, "age_max": None, "age_source": "text"}


def test_builder_classifies_prices() -> None:
    builder = DocumentBuilder()
    assert builder._classify_price("Gratuit") == {
        "price_status": "gratuit",
        "price_amounts_eur": [],
    }
    assert builder._classify_price("Tarif : 8 €") == {
        "price_status": "payant",
        "price_amounts_eur": [8.0],
    }
    assert builder._classify_price("Gratuit, réservation 5 euros") == {
        "price_status": "mixte",
        "price_amounts_eur": [5.0],
    }
    assert builder._classify_price("Sur inscription") == {
        "price_status": "inconnu",
        "price_amounts_eur": [],
    }


def test_builder_builds_clean_content_without_duplicate_long_description() -> None:
    event = Event(
        id="1",
        title="<b>Titre</b>",
        description="Description",
        long_description="Description",
        location_description="Lieu intéressant",
    )
    content = DocumentBuilder()._build_cleaned_content(event)
    assert content == "Titre\n\nDescription\n\nLieu intéressant"


def test_builder_falls_back_when_summary_is_empty_or_fails(sample_event: Event) -> None:
    class EmptySummarizer:
        def summarize(self, text: str) -> str:
            return ""

    class FailingSummarizer:
        def summarize(self, text: str) -> str:
            raise RuntimeError("LLM indisponible")

    cleaned = DocumentBuilder()._build_cleaned_content(sample_event)
    assert (
        DocumentBuilder(EmptySummarizer(), True)._build_embedding_content(cleaned)
        == cleaned
    )
    assert (
        DocumentBuilder(FailingSummarizer(), True)._build_embedding_content(cleaned)
        == cleaned
    )


def test_builder_uses_summary_when_available(sample_event: Event) -> None:
    class Summarizer:
        def summarize(self, text: str) -> str:
            return " Résumé synthétique "

    builder = DocumentBuilder(Summarizer(), True)
    assert builder._build_embedding_content("Contenu") == "Résumé synthétique"


def test_builder_builds_structured_document(sample_event: Event) -> None:
    document = DocumentBuilder().build(sample_event)
    assert document["id"] == "evt-1"
    assert "Atelier céramique" in document["page_content"]
    metadata = document["metadata"]
    assert metadata["uid"] == "evt-1"
    assert metadata["age_min"] == 8
    assert metadata["age_source"] == "event"
    assert metadata["audience_categories"] == ["enfants"]
    assert metadata["price_status"] == "gratuit"
    assert metadata["price_amounts_eur"] == []
