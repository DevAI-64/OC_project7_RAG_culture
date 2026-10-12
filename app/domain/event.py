from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class Event:
    """Represent a cultural event independently from its data source."""

    id: str
    title: str

    # Descriptions
    description: str = ""
    long_description: str = ""

    # Conditions et informations complémentaires
    conditions: str = ""
    keywords: tuple[str, ...] = field(default_factory=tuple)

    # Dates normalisées pour les filtres
    first_date: date | None = None
    last_date: date | None = None

    # Horaires et dates d'origine
    first_date_raw: str = ""
    first_date_end_raw: str = ""
    last_date_raw: str = ""
    last_date_end_raw: str = ""
    date_range: str = ""
    timings: str = ""

    # Localisation : pas de city, l'adresse précise est conservée
    location_name: str = ""
    address: str = ""
    district: str = ""
    postal_code: str = ""
    department: str = ""
    region: str = ""
    country: str = ""
    latitude: float | None = None
    longitude: float | None = None

    # Liens et médias
    url: str = ""
    slug: str = ""
    image_url: str = ""
    thumbnail_url: str = ""
    original_image_url: str = ""

    # Informations sur le lieu
    location_website: str = ""
    location_description: str = ""
    location_access: str = ""
    location_image: str = ""
    location_image_credits: str = ""

    # Informations pratiques
    accessibility: str = ""
    accessibility_label: str = ""
    attendance_mode: str = ""
    online_access_link: str = ""
    status: str = ""
    age_min: int | None = None
    age_max: int | None = None
    registration: str = ""

    # Agenda d'origine et informations diverses
    origin_agenda_title: str = ""
    origin_agenda_uid: str = ""
    updated_at: str = ""
    image_credits: str = ""
    links: str = ""
    location_links: str = ""

    def is_in_period(
        self,
        start_date: date,
        end_date: date,
    ) -> bool:
        """Return whether the event overlaps the requested date range.

        Args:
            start_date: Beginning of the requested period.
            end_date: End of the requested period.

        Returns:
            True when the event overlaps the requested period.
        """
        event_start = self.first_date or self.last_date
        event_end = self.last_date or self.first_date

        if event_start is None or event_end is None:
            return False

        return event_start <= end_date and event_end >= start_date
