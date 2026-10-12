import json
import re
from datetime import date, datetime
from html import unescape
from typing import Any

from app.domain.event import Event


def parse_date(value: Any) -> date | None:
    """Parse a date value into a Python date.

    Args:
        value: Value to convert to a date.

    Returns:
        The parsed date, or None if the value cannot be parsed.
    """
    if not value:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None

    return None


def clean_html(value: Any) -> str:
    """Remove HTML markup while preserving readable text.

    Args:
        value: HTML content to clean.

    Returns:
        Clean plain text, or an empty string for an empty value.
    """
    if not value:
        return ""

    text = str(value)

    text = re.sub(
        r"<br\s*/?>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"</p\s*>",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"<[^>]+>", " ", text)

    text = unescape(text)

    # HTML &nbsp; is decoded by unescape() as U+00A0.
    text = text.replace("\xa0", " ")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_json(value: Any) -> Any:
    """Parse a JSON string when possible.

    Args:
        value: Value that may contain JSON-encoded data.

    Returns:
        The decoded JSON value when parsing succeeds, otherwise the original
        value.
    """
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError, TypeError:
        return value


def extract_label(value: Any) -> str:
    """Extract the French label from an OpenAgenda value.

    Args:
        value: OpenAgenda value containing a label.

    Returns:
        The French label, or an empty string if no label is available.
    """
    value = parse_json(value)

    if isinstance(value, dict):
        label = value.get("label")

        if isinstance(label, dict):
            return str(label.get("fr") or "")

        if isinstance(label, str):
            return label

    if isinstance(value, str):
        return value

    return ""


def parse_keywords(value: Any) -> tuple[str, ...]:
    """Normalize OpenAgenda keywords into a unique tuple of strings.

    Args:
        value: Raw OpenAgenda keyword value.

    Returns:
        A tuple containing normalized, non-empty, unique keywords.
    """
    value = parse_json(value)

    if value is None:
        return ()

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return ()

    result: list[str] = []

    for item in values:
        if isinstance(item, dict):
            item = item.get("fr") or item.get("label") or ""

        keyword = str(item).strip()

        if keyword and keyword not in result:
            result.append(keyword)

    return tuple(result)


def parse_float(value: Any) -> float | None:
    """Convert a value to a float when possible.

    Args:
        value: Value to convert.

    Returns:
        The converted float, or None if conversion fails.
    """
    if value is None or value == "":
        return None

    try:
        return float(value)
    except TypeError, ValueError:
        return None


def parse_int(value: Any) -> int | None:
    """Convert a value to an integer when possible.

    Args:
        value: Value to convert.

    Returns:
        The converted integer, or None if conversion fails.
    """
    if value is None or value == "":
        return None

    try:
        return int(value)
    except TypeError, ValueError:
        return None


def as_string(value: Any) -> str:
    """Convert a value to a normalized string representation.

    Args:
        value: Value to convert.

    Returns:
        A stripped string representation, or an empty string for None.
    """
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value).strip()


class EventProcessor:
    """Convert raw OpenAgenda records into domain events."""

    def process(
        self,
        raw_events: list[dict[str, Any]],
    ) -> list[Event]:
        """Convert valid raw records into domain events.

        Args:
            raw_events: Raw event records returned by OpenAgenda.

        Returns:
            Domain events successfully converted from the raw records.
        """
        events: list[Event] = []

        for raw_event in raw_events:
            event = self._to_event(raw_event)

            if event is not None:
                events.append(event)

        return events

    def _to_event(
        self,
        raw: dict[str, Any],
    ) -> Event | None:
        """Convert one OpenAgenda record into a domain event.

        Args:
            raw: Raw OpenAgenda event record.

        Returns:
            The converted domain event, or None when required identifying
            fields are missing.
        """
        event_id = raw.get("uid") or raw.get("id")
        title = raw.get("title_fr") or raw.get("title")

        if not event_id or not title:
            return None

        first_date_raw = as_string(
            raw.get("firstdate_begin") or raw.get("firstDate") or raw.get("first_date")
        )
        first_date_end_raw = as_string(raw.get("firstdate_end"))
        last_date_raw = as_string(
            raw.get("lastdate_begin") or raw.get("lastDate") or raw.get("last_date")
        )
        last_date_end_raw = as_string(raw.get("lastdate_end"))

        coordinates = raw.get("location_coordinates") or {}

        if isinstance(coordinates, str):
            coordinates = parse_json(coordinates)

        if not isinstance(coordinates, dict):
            coordinates = {}

        keywords = parse_keywords(raw.get("keywords_fr") or raw.get("keywords"))

        return Event(
            id=str(event_id),
            title=str(title).strip(),
            description=clean_html(raw.get("description_fr") or raw.get("description")),
            long_description=clean_html(
                raw.get("longdescription_fr") or raw.get("long_description")
            ),
            conditions=clean_html(raw.get("conditions_fr") or raw.get("conditions")),
            keywords=keywords,
            first_date=parse_date(first_date_raw),
            last_date=parse_date(last_date_raw),
            first_date_raw=first_date_raw,
            first_date_end_raw=first_date_end_raw,
            last_date_raw=last_date_raw,
            last_date_end_raw=last_date_end_raw,
            date_range=as_string(raw.get("daterange_fr") or raw.get("date_range")),
            timings=as_string(raw.get("timings")),
            location_name=as_string(raw.get("location_name")),
            address=as_string(raw.get("location_address") or raw.get("address")),
            district=as_string(raw.get("location_district")),
            postal_code=as_string(raw.get("location_postalcode")),
            department=as_string(raw.get("location_department")),
            region=as_string(raw.get("location_region")),
            country=as_string(raw.get("country_fr")),
            latitude=parse_float(coordinates.get("lat")),
            longitude=parse_float(coordinates.get("lon")),
            url=as_string(
                raw.get("canonicalurl") or raw.get("canonicalUrl") or raw.get("url")
            ),
            slug=as_string(raw.get("slug")),
            image_url=as_string(raw.get("image")),
            thumbnail_url=as_string(raw.get("thumbnail")),
            original_image_url=as_string(raw.get("originalimage")),
            location_website=as_string(raw.get("location_website")),
            location_description=clean_html(raw.get("location_description_fr")),
            location_access=clean_html(raw.get("location_access_fr")),
            location_image=as_string(raw.get("location_image")),
            location_image_credits=as_string(raw.get("location_imagecredits")),
            accessibility=as_string(raw.get("accessibility")),
            accessibility_label=as_string(raw.get("accessibility_label_fr")),
            attendance_mode=extract_label(raw.get("attendancemode")),
            online_access_link=as_string(raw.get("onlineaccesslink")),
            status=extract_label(raw.get("status")),
            age_min=parse_int(raw.get("age_min")),
            age_max=parse_int(raw.get("age_max")),
            registration=as_string(raw.get("registration")),
            origin_agenda_title=as_string(raw.get("originagenda_title")),
            origin_agenda_uid=as_string(raw.get("originagenda_uid")),
            updated_at=as_string(raw.get("updatedat")),
            image_credits=as_string(raw.get("imagecredits")),
            links=as_string(raw.get("links")),
            location_links=as_string(raw.get("location_links")),
        )
