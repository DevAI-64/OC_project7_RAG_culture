from datetime import date

import pytest

from app.domain.event import Event


@pytest.fixture
def sample_event() -> Event:
    return Event(
        id="evt-1",
        title="Atelier céramique",
        description="Un atelier pour les enfants de 8 à 12 ans.",
        long_description="Découverte de la terre et modelage.",
        conditions="Gratuit sur réservation.",
        keywords=("céramique", "modelage"),
        first_date=date(2026, 10, 20),
        last_date=date(2026, 10, 20),
        first_date_raw="2026-10-20T14:00:00+02:00",
        last_date_raw="2026-10-20T16:00:00+02:00",
        timings="14:00-16:00",
        location_name="Musée de Pau",
        address="1 rue Exemple, Pau",
        postal_code="64000",
        url="https://example.test/event/1",
        age_min=8,
        age_max=12,
        accessibility_label="Accessible PMR",
    )
