from datetime import date

from app.domain.event import Event


def test_event_overlaps_period() -> None:
    event = Event(
        id="1",
        title="Test",
        first_date=date(2026, 10, 10),
        last_date=date(2026, 10, 20),
    )
    assert event.is_in_period(date(2026, 10, 15), date(2026, 10, 25))
    assert not event.is_in_period(date(2026, 11, 1), date(2026, 11, 2))


def test_event_without_dates_does_not_overlap() -> None:
    assert not Event(id="1", title="Test").is_in_period(
        date(2026, 1, 1), date(2026, 1, 2)
    )
