import json
import re
from datetime import date, datetime
from typing import Any

from app.infrastructure.rag.query_parser import QueryCriteria


def parse_event_date(value: Any) -> date | None:
    """Convert a supported value to a date.

    Args:
        value: Value to convert. Supported values are dates, datetimes,
            and ISO-formatted date strings.

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


class EventFilter:
    """Apply explicit user constraints before semantic retrieval."""

    def filter(
        self,
        documents: list[dict[str, Any]],
        criteria: QueryCriteria,
        today: date,
    ) -> list[dict[str, Any]]:
        """Filter documents according to explicit query criteria.

        Args:
            documents: Event documents with structured metadata.
            criteria: Explicit constraints extracted from the user query.
            today: Current date used when no period is explicitly requested.

        Returns:
            Documents satisfying all explicit constraints.
        """
        return [
            document
            for document in documents
            if self._matches(document, criteria, today)
        ]

    def _matches(
        self,
        document: dict[str, Any],
        criteria: QueryCriteria,
        today: date,
    ) -> bool:
        """Check whether a document satisfies all explicit constraints."""
        metadata = document.get("metadata") or {}

        # When the user does not specify a period, only future events
        # are considered. An explicit period may refer to the past.
        if criteria.period is None:
            last_date = parse_event_date(metadata.get("last_date"))

            if last_date is None or last_date < today:
                return False

        if criteria.period is not None and not self._matches_period(
            metadata,
            criteria.period,
        ):
            return False

        if criteria.max_price is not None and not self._matches_price(
            metadata,
            criteria.max_price,
            criteria.max_price_inclusive,
        ):
            return False

        if criteria.audience is not None and not self._matches_audience(
            metadata,
            criteria.audience,
        ):
            return False

        if criteria.accessibility_requested and not self._matches_accessibility(
            metadata
        ):
            return False

        return True

    def _matches_period(
        self,
        metadata: dict[str, Any],
        period: tuple[date, date],
    ) -> bool:
        """Check whether an event overlaps the requested period.

        Note:
            This currently uses the event's first and last dates.
            Recurring events should eventually be checked against their
            actual occurrences.
        """
        first_date = parse_event_date(metadata.get("first_date"))
        last_date = parse_event_date(metadata.get("last_date"))

        if first_date is None or last_date is None:
            return False

        start_date, end_date = period

        return first_date <= end_date and last_date >= start_date

    def _matches_price(
        self,
        metadata: dict[str, Any],
        maximum: float,
        inclusive: bool,
    ) -> bool:
        """Check whether at least one known price satisfies the limit."""
        status = str(metadata.get("price_status") or "").lower()
        amounts = self._numbers(metadata.get("price_amounts_eur"))

        if status == "gratuit":
            return True

        if not amounts:
            return False

        if inclusive:
            return any(amount <= maximum for amount in amounts)

        return any(amount < maximum for amount in amounts)

    def _matches_audience(
        self,
        metadata: dict[str, Any],
        audience: str,
    ) -> bool:
        """Check whether an event matches the requested audience."""
        categories = {
            str(value).lower()
            for value in metadata.get("audience_categories", [])
            if value
        }

        if audience == "enfants":
            if categories.intersection({"enfants", "familles"}):
                return True

            age_min = self._int(metadata.get("age_min"))
            age_max = self._int(metadata.get("age_max"))

            return (
                age_min is not None
                and age_min <= 12
                and (age_max is None or age_max >= age_min)
            )

        if audience == "familles":
            return bool(
                categories.intersection(
                    {"familles", "enfants"},
                )
            )

        if audience == "adultes":
            if "adultes" in categories:
                return True

            age_min = self._int(metadata.get("age_min"))

            return age_min is not None and age_min >= 18

        if audience == "adolescents":
            if "adolescents" in categories:
                return True

            age_min = self._int(metadata.get("age_min"))
            age_max = self._int(metadata.get("age_max"))

            return (
                age_min is not None
                and age_min >= 12
                and (age_max is None or age_max <= 18)
            )

        return audience in categories

    def _matches_accessibility(
        self,
        metadata: dict[str, Any],
    ) -> bool:
        """Check whether accessibility information indicates accessibility."""
        values = [
            metadata.get("accessibility"),
            metadata.get("accessibility_label"),
            metadata.get("location_access"),
        ]

        text = " ".join(str(value or "") for value in values).lower()

        if not text:
            return False

        negative_patterns = (
            "non accessible",
            "pas accessible",
            "inaccessible",
            "non-accessible",
        )

        if any(pattern in text for pattern in negative_patterns):
            return False

        positive_patterns = (
            "accessible",
            "accessibilité",
            "pmr",
            "fauteuil",
            "mobilité réduite",
            "handicap",
        )

        return any(pattern in text for pattern in positive_patterns)

    @staticmethod
    def _numbers(value: Any) -> list[float]:
        """Extract numeric prices from supported values."""
        if isinstance(value, list):
            return [float(item) for item in value if isinstance(item, (int, float))]

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return EventFilter._numbers(parsed)
            except json.JSONDecodeError:
                return [
                    float(match.replace(",", "."))
                    for match in re.findall(
                        r"\d+(?:[,.]\d+)?",
                        value,
                    )
                ]

        return []

    @staticmethod
    def _int(value: Any) -> int | None:
        """Convert a value to an integer when possible."""
        try:
            return int(value) if value is not None else None
        except TypeError, ValueError:
            return None
