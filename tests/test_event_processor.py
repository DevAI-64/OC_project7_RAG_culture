from datetime import date, datetime

from app.ingestion.event_processor import (
    EventProcessor,
    as_string,
    clean_html,
    extract_label,
    parse_date,
    parse_float,
    parse_int,
    parse_json,
    parse_keywords,
)


def test_scalar_parsers_cover_valid_and_invalid_values() -> None:
    assert parse_date(datetime(2026, 10, 1, 10)) == date(2026, 10, 1)
    assert parse_date(date(2026, 10, 1)) == date(2026, 10, 1)
    assert parse_date("2026-10-01T10:00:00Z") == date(2026, 10, 1)
    assert parse_date("bad") is None
    assert parse_date(None) is None
    assert clean_html("<p>Bonjour&nbsp;!</p><br>Monde") == "Bonjour !\n\nMonde"
    assert clean_html(None) == ""
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json("bad") == "bad"
    assert parse_json(3) == 3
    assert extract_label('{"label": {"fr": "Accessible"}}') == "Accessible"
    assert extract_label('{"label": "Direct"}') == "Direct"
    assert extract_label("Direct") == "Direct"
    assert extract_label(None) == ""
    assert parse_keywords('["photo", "photo", {"fr": "enfant"}]') == ("photo", "enfant")
    assert parse_keywords(None) == ()
    assert parse_keywords(12) == ()
    assert parse_float("2.5") == 2.5
    assert parse_float("bad") is None
    assert parse_int("12") == 12
    assert parse_int("bad") is None
    assert as_string(None) == ""
    assert as_string({"a": 1}) == '{"a": 1}'
    assert as_string(" x ") == "x"


def test_processor_converts_a_complete_event() -> None:
    raw = {
        "uid": "evt-42",
        "title_fr": "Titre",
        "description_fr": "<p>Description</p>",
        "longdescription_fr": "Longue description",
        "conditions_fr": "Gratuit",
        "keywords_fr": '["photo", "culture"]',
        "firstdate_begin": "2026-10-20T14:00:00Z",
        "firstdate_end": "2026-10-20T16:00:00Z",
        "lastdate_begin": "2026-10-20T14:00:00Z",
        "lastdate_end": "2026-10-20T16:00:00Z",
        "location_coordinates": '{"lat": 43.3, "lon": -0.37}',
        "location_address": "1 rue Test",
        "canonicalurl": "https://example.test",
        "attendancemode": '{"label": {"fr": "Sur place"}}',
        "status": '{"label": {"fr": "Publié"}}',
        "age_min": 8,
        "age_max": 12,
    }
    event = EventProcessor().process([raw])[0]
    assert event.id == "evt-42"
    assert event.title == "Titre"
    assert event.description == "Description"
    assert event.keywords == ("photo", "culture")
    assert event.latitude == 43.3
    assert event.longitude == -0.37
    assert event.attendance_mode == "Sur place"
    assert event.status == "Publié"
    assert event.age_min == 8


def test_processor_uses_fallback_fields_and_skips_invalid_records() -> None:
    raw = {
        "id": 7,
        "title": "Fallback",
        "firstDate": "2026-11-01",
        "lastDate": "2026-11-02",
        "keywords": ["x"],
        "location_coordinates": [],
    }
    events = EventProcessor().process([{}, raw])
    assert len(events) == 1
    assert events[0].id == "7"
    assert events[0].title == "Fallback"
