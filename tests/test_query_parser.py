from datetime import date

import pytest

from app.infrastructure.rag.query_parser import QueryParser


@pytest.fixture
def parser() -> QueryParser:
    return QueryParser()


def test_parser_extracts_month_price_audience_and_semantic_query(parser: QueryParser) -> None:
    criteria = parser.parse(
        "Quelles activités autour de la céramique sont prévues en octobre pour les enfants à moins de 10 € ?",
        today=date(2026, 9, 30),
    )
    assert criteria.period == (date(2026, 10, 1), date(2026, 10, 31))
    assert criteria.max_price == 10
    assert criteria.max_price_inclusive is False
    assert criteria.audience == "enfants"
    assert criteria.semantic_query == "autour de la céramique"


def test_parser_handles_next_month_and_current_month(parser: QueryParser) -> None:
    today = date(2026, 12, 15)
    assert parser.parse("événements le mois prochain", today).period == (
        date(2027, 1, 1), date(2027, 1, 31)
    )
    assert parser.parse("événements ce mois-ci", today).period == (
        date(2026, 12, 1), date(2026, 12, 31)
    )


def test_parser_assigns_past_month_to_next_year(parser: QueryParser) -> None:
    assert parser.parse("sorties en janvier", date(2026, 9, 30)).period == (
        date(2027, 1, 1), date(2027, 1, 31)
    )


def test_parser_accepts_explicit_year(parser: QueryParser) -> None:
    assert parser.parse("événements en août 2026", date(2027, 1, 1)).period == (
        date(2026, 8, 1), date(2026, 8, 31)
    )


@pytest.mark.parametrize(
    ("question", "price", "inclusive"),
    [
        ("sorties sous 12 euros", 12, False),
        ("sorties maximum 15 €", 15, True),
        ("sorties jusqu'à 20", 20, True),
        ("sorties au plus 7,50 euros", 7.5, True),
    ],
)
def test_parser_extracts_price_variants(
    parser: QueryParser,
    question: str,
    price: float,
    inclusive: bool,
) -> None:
    criteria = parser.parse(question, date(2026, 9, 30))
    assert criteria.max_price == price
    assert criteria.max_price_inclusive is inclusive


@pytest.mark.parametrize(
    ("question", "audience"),
    [
        ("activité pour les enfants", "enfants"),
        ("sortie pour les ados", "adolescents"),
        ("sortie pour adultes", "adultes"),
        ("activité en famille", "familles"),
    ],
)
def test_parser_extracts_audience(
    parser: QueryParser,
    question: str,
    audience: str,
) -> None:
    assert parser.parse(question).audience == audience


def test_parser_detects_accessibility(parser: QueryParser) -> None:
    criteria = parser.parse("événement accessible PMR")
    assert criteria.accessibility_requested


def test_parser_returns_empty_semantic_query_for_generic_request(parser: QueryParser) -> None:
    assert parser.parse("Quels événements sont prévus en octobre à Pau?").semantic_query == ""


def test_parser_rejects_empty_question(parser: QueryParser) -> None:
    with pytest.raises(ValueError, match="vide"):
        parser.parse("  ")
