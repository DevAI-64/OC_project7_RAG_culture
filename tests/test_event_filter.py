from datetime import date

import pytest

from app.infrastructure.rag.event_filter import EventFilter, parse_event_date
from app.infrastructure.rag.query_parser import QueryCriteria


def document(**metadata):
    return {"metadata": metadata, "page_content": metadata.get("title", "")}


def test_parse_event_date_supports_date_datetime_and_iso() -> None:
    from datetime import datetime

    expected = date(2026, 10, 20)
    assert parse_event_date(expected) == expected
    assert parse_event_date(datetime(2026, 10, 20, 14)) == expected
    assert parse_event_date("2026-10-20T14:00:00Z") == expected
    assert parse_event_date("invalid") is None
    assert parse_event_date(None) is None
    assert parse_event_date(42) is None


def test_filter_excludes_past_events_without_explicit_period() -> None:
    criteria = QueryCriteria()
    docs = [document(last_date="2026-09-29"), document(last_date="2026-10-01")]
    result = EventFilter().filter(docs, criteria, date(2026, 9, 30))
    assert len(result) == 1


def test_explicit_period_can_target_past_events() -> None:
    criteria = QueryCriteria(period=(date(2026, 8, 1), date(2026, 8, 31)))
    docs = [document(first_date="2026-08-10", last_date="2026-08-10")]
    assert EventFilter().filter(docs, criteria, date(2026, 9, 30)) == docs


def test_filter_rejects_invalid_dates() -> None:
    criteria = QueryCriteria()
    assert (
        EventFilter().filter([document(last_date="bad")], criteria, date(2026, 9, 30))
        == []
    )


def test_price_filter_handles_free_amounts_and_unknown() -> None:
    event_filter = EventFilter()
    assert event_filter._matches_price({"price_status": "gratuit"}, 0, True)
    assert event_filter._matches_price({"price_amounts_eur": [5, 12]}, 10, True)
    assert event_filter._matches_price({"price_amounts_eur": "[5, 12]"}, 10, False)
    assert not event_filter._matches_price({"price_status": "inconnu"}, 10, True)
    assert not event_filter._matches_price({"price_amounts_eur": [10]}, 10, False)


@pytest.mark.parametrize(
    ("metadata", "audience", "expected"),
    [
        ({"audience_categories": ["enfants"]}, "enfants", True),
        ({"audience_categories": ["familles"]}, "enfants", True),
        ({"age_min": 8, "age_max": 12}, "enfants", True),
        ({"audience_categories": ["adultes"]}, "adultes", True),
        ({"age_min": 18}, "adultes", True),
        ({"audience_categories": ["adolescents"]}, "adolescents", True),
        ({"age_min": 13, "age_max": 17}, "adolescents", True),
        ({"audience_categories": ["familles"]}, "familles", True),
        ({"audience_categories": ["seniors"]}, "adultes", False),
        ({}, "adultes", False),
    ],
)
def test_audience_filter(metadata, audience, expected) -> None:
    assert EventFilter()._matches_audience(metadata, audience) is expected


def test_accessibility_filter() -> None:
    event_filter = EventFilter()
    assert event_filter._matches_accessibility({"accessibility": "Accessible PMR"})
    assert not event_filter._matches_accessibility({"accessibility": "Non accessible"})
    assert not event_filter._matches_accessibility({})


def test_numbers_and_int_handle_supported_values() -> None:
    assert EventFilter._numbers([1, 2.5, "3"]) == [1.0, 2.5]
    assert EventFilter._numbers('["1", 2]') == [2.0]
    assert EventFilter._numbers("5,50 € et 10 €") == [5.5, 10.0]
    assert EventFilter._numbers(None) == []
    assert EventFilter._int("12") == 12
    assert EventFilter._int("bad") is None
    assert EventFilter._int(None) is None


def test_filter_applies_all_constraints() -> None:
    criteria = QueryCriteria(
        period=(date(2026, 10, 1), date(2026, 10, 31)),
        max_price=10,
        max_price_inclusive=True,
        audience="enfants",
        accessibility_requested=True,
    )
    matching = document(
        first_date="2026-10-10",
        last_date="2026-10-10",
        price_amounts_eur=[8],
        audience_categories=["enfants"],
        accessibility="Accessible PMR",
    )
    assert EventFilter().filter([matching], criteria, date(2026, 12, 1)) == [matching]
