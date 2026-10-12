from datetime import date
from types import SimpleNamespace

from app.ingestion.openagenda_client import OpenAgendaClient


def test_build_where_clause() -> None:
    client = OpenAgendaClient(city="Pau")
    assert client._build_where_clause(date(2025, 10, 1)) == (
        'location_city = "Pau" AND lastdate_begin >= date\'2025-10-01\''
    )


def test_create_session_configures_retries() -> None:
    session = OpenAgendaClient._create_session()
    assert "https://" in session.adapters


def test_fetch_events_paginates_until_total_count(monkeypatch) -> None:
    responses = [
        SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"total_count": 2, "results": [{"uid": "1"}]},
        ),
        SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"total_count": 2, "results": [{"uid": "2"}]},
        ),
    ]

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return responses.pop(0)

    monkeypatch.setattr(OpenAgendaClient, "_create_session", staticmethod(lambda: Session()))
    result = OpenAgendaClient(page_size=1).fetch_events()
    assert result["metadata"]["count"] == 2
    assert [item["uid"] for item in result["events"]] == ["1", "2"]


def test_fetch_events_stops_on_empty_page(monkeypatch) -> None:
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"total_count": None, "results": []},
            )

    monkeypatch.setattr(OpenAgendaClient, "_create_session", staticmethod(lambda: Session()))
    assert OpenAgendaClient().fetch_events()["events"] == []
