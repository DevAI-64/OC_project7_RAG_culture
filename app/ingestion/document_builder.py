import logging
import re
from html import unescape

from bs4 import BeautifulSoup

from app.domain.event import Event
from app.infrastructure.llm.event_summarizer import EventSummarizer

logger = logging.getLogger(__name__)


class DocumentBuilder:
    """Build searchable documents and structured metadata from domain events."""

    AUDIENCE_PATTERNS = {
        "enfants": [
            r"\benfant(?:s)?\b",
            r"\bjeune public\b",
            r"\bpour les petits\b",
            r"\bpetite enfance\b",
            r"\bbébés?\b",
        ],
        "adolescents": [
            r"\badolescent(?:s)?\b",
            r"\badolescence\b",
            r"\bados?\b",
        ],
        "adultes": [
            r"\badulte(?:s)?\b",
            r"\bpour les grands\b",
        ],
        "familles": [
            r"\bfamille(?:s)?\b",
            r"\ben famille\b",
            r"\bparents et enfants\b",
        ],
        "scolaires": [
            r"\bscolaire(?:s)?\b",
            r"\bsortie scolaire\b",
            r"\bétablissements scolaires\b",
        ],
        "étudiants": [
            r"\bétudiant(?:s)?\b",
            r"\buniversitaire(?:s)?\b",
        ],
        "seniors": [
            r"\bsenior(?:s)?\b",
            r"\bpersonnes âgées\b",
        ],
        "tout_public": [
            r"\btout public\b",
            r"\btous publics\b",
            r"\btout âge\b",
            r"\btous les âges\b",
        ],
    }

    FREE_PATTERNS = [
        r"\bgratuit(?:e|s|es)?\b",
        r"\bentrée libre\b",
        r"\baccès libre\b",
        r"\bparticipation libre\b",
        r"\ben accès libre\b",
        r"\bsans frais\b",
        r"\bsans réservation et gratuit\b",
    ]

    PAID_PATTERNS = [
        r"\bpayant(?:e|s|es)?\b",
        r"\bentrée payante\b",
        r"\bsur réservation payante\b",
        r"\btarif(?:s)?\s*:",
        r"\bprix\s*:",
        r"\bbilletterie\b",
        r"\bacheter un billet\b",
        r"€",
        r"\beuros?\b",
    ]

    PRICE_PATTERN = re.compile(
        r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:€|euros?)(?!\w)",
        re.IGNORECASE,
    )

    AGE_PATTERNS = [
        (
            re.compile(
                r"\b(?:dès|à partir de|pour les)\s+" r"(\d{1,2})\s*ans?\b",
                re.IGNORECASE,
            ),
            "min",
        ),
        (
            re.compile(
                r"\bde\s+(\d{1,2})\s*" r"à\s+(\d{1,2})\s*ans?\b",
                re.IGNORECASE,
            ),
            "range",
        ),
        (
            re.compile(
                r"\bentre\s+(\d{1,2})\s+" r"et\s+(\d{1,2})\s*ans?\b",
                re.IGNORECASE,
            ),
            "range",
        ),
        (
            re.compile(
                r"\bjusqu'à\s+(\d{1,2})\s*ans?\b",
                re.IGNORECASE,
            ),
            "max",
        ),
        (
            re.compile(
                r"\bmoins de\s+(\d{1,2})\s*ans?\b",
                re.IGNORECASE,
            ),
            "max_exclusive",
        ),
    ]

    def __init__(
        self,
        summarizer: EventSummarizer | None = None,
        use_summarization: bool = False,
    ) -> None:
        """Initialize the document builder.

        Args:
            summarizer: Optional LLM summarizer used to improve semantic
                embedding content.
            use_summarization: Whether to use the summarizer when building
                embedding content.

        Raises:
            ValueError: If summarization is enabled without a summarizer.
        """
        self.summarizer = summarizer
        self.use_summarization = use_summarization

        if use_summarization and summarizer is None:
            raise ValueError(
                "Un EventSummarizer doit être fourni si use_summarization=True."
            )

    @staticmethod
    def _clean_text(value: str | None) -> str:
        """Remove HTML markup and normalize whitespace.

        Args:
            value: Text to clean.

        Returns:
            Cleaned plain text.
        """
        if not value:
            return ""

        text = unescape(value)

        text = BeautifulSoup(
            text,
            "html.parser",
        ).get_text(" ")

        text = text.replace("\xa0", " ")

        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for case-insensitive pattern matching.

        Args:
            text: Text to normalize.

        Returns:
            Lowercase text with normalized whitespace.
        """
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        """Remove duplicate values while preserving their order.

        Args:
            items: Values to deduplicate.

        Returns:
            Deduplicated values in their original order.
        """
        return list(dict.fromkeys(items))

    @staticmethod
    def _matches_any(
        text: str,
        patterns: list[str],
    ) -> bool:
        """Return whether the text matches at least one pattern.

        Args:
            text: Text to inspect.
            patterns: Regular expressions to test.

        Returns:
            True if at least one pattern matches.
        """
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def _classify_audience(
        self,
        text: str,
    ) -> list[str]:
        """Detect audience categories mentioned in event content.

        Args:
            text: Event content used for classification.

        Returns:
            Matching audience categories.
        """
        normalized = self._normalize(text)

        audiences = [
            audience
            for audience, patterns in self.AUDIENCE_PATTERNS.items()
            if self._matches_any(normalized, patterns)
        ]

        return self._unique(audiences)

    def _extract_age_from_text(
        self,
        text: str,
    ) -> tuple[int | None, int | None]:
        """Extract an explicit age range from event text.

        Args:
            text: Event content to inspect.

        Returns:
            A tuple containing the minimum and maximum ages detected.
        """
        normalized = self._normalize(text)

        age_min: int | None = None
        age_max: int | None = None

        for pattern, kind in self.AGE_PATTERNS:
            match = pattern.search(normalized)

            if not match:
                continue

            if kind == "min":
                age_min = int(match.group(1))

            elif kind == "range":
                age_min = int(match.group(1))
                age_max = int(match.group(2))

            elif kind == "max":
                age_max = int(match.group(1))

            elif kind == "max_exclusive":
                age_max = int(match.group(1)) - 1

            break

        return age_min, age_max

    def _classify_age(
        self,
        event: Event,
        text: str,
    ) -> dict[str, int | str | None]:
        """Determine the applicable age range for an event.

        Args:
            event: Event containing structured age information.
            text: Event content used as a fallback.

        Returns:
            Age minimum, age maximum, and the source of the information.
        """
        age_min = event.age_min
        age_max = event.age_max

        if age_min is not None or age_max is not None:
            source = "event"
        else:
            age_min, age_max = self._extract_age_from_text(text)
            source = "text" if age_min is not None or age_max is not None else "unknown"

        return {
            "age_min": age_min,
            "age_max": age_max,
            "age_source": source,
        }

    def _classify_price(
        self,
        text: str,
    ) -> dict[str, str | list[float]]:
        """Classify the pricing information found in event text.

        Args:
            text: Event content containing potential pricing information.

        Returns:
            Pricing status and detected euro amounts.
        """
        normalized = self._normalize(text)

        is_free = self._matches_any(
            normalized,
            self.FREE_PATTERNS,
        )
        is_paid = self._matches_any(
            normalized,
            self.PAID_PATTERNS,
        )

        if is_free and is_paid:
            status = "mixte"
        elif is_free:
            status = "gratuit"
        elif is_paid:
            status = "payant"
        else:
            status = "inconnu"

        amounts = [
            float(match.group(1).replace(",", "."))
            for match in self.PRICE_PATTERN.finditer(text)
        ]

        return {
            "price_status": status,
            "price_amounts_eur": sorted(set(amounts)),
        }

    def _build_cleaned_content(
        self,
        event: Event,
    ) -> str:
        """Build cleaned semantic content from the event descriptions.

        Args:
            event: Event used as the source of the content.

        Returns:
            Cleaned text suitable for semantic processing.
        """
        title = self._clean_text(event.title)
        description = self._clean_text(event.description)
        long_description = self._clean_text(event.long_description)
        location_description = self._clean_text(event.location_description)

        parts: list[str] = []

        if title:
            parts.append(title)

        if description:
            parts.append(description)

        if long_description and long_description != description:
            parts.append(long_description)

        if location_description:
            parts.append(location_description)

        return "\n\n".join(parts)

    def _build_embedding_content(
        self,
        cleaned_content: str,
    ) -> str:
        """Build the text used for semantic embeddings.

        Args:
            cleaned_content: Cleaned event content.

        Returns:
            Text to use for embedding.
        """
        if not cleaned_content:
            return ""

        if not self.use_summarization or self.summarizer is None:
            return cleaned_content

        try:
            summary = self.summarizer.summarize(cleaned_content)
            summary = self._clean_text(summary)

            if summary:
                return summary

            logger.warning("Résumé vide : utilisation du texte nettoyé.")

        except Exception:
            logger.exception("Échec du résumé LLM : utilisation du texte nettoyé.")

        return cleaned_content

    def build(
        self,
        event: Event,
    ) -> dict[str, object]:
        """Build an embedding document and its structured metadata.

        Args:
            event: Domain event to convert.

        Returns:
            A dictionary containing the event identifier, embedding content,
            and structured metadata.
        """
        cleaned_title = self._clean_text(event.title)
        cleaned_description = self._clean_text(event.description)
        cleaned_long_description = self._clean_text(event.long_description)
        cleaned_conditions = self._clean_text(event.conditions)
        cleaned_keywords = [
            self._clean_text(keyword) for keyword in (event.keywords or [])
        ]

        classification_text = "\n".join(
            value
            for value in [
                cleaned_title,
                cleaned_description,
                cleaned_long_description,
                cleaned_conditions,
                ", ".join(cleaned_keywords),
            ]
            if value
        )

        audiences = self._classify_audience(classification_text)

        age_info = self._classify_age(
            event,
            classification_text,
        )

        price_info = self._classify_price(
            "\n".join(
                value
                for value in [
                    cleaned_description,
                    cleaned_long_description,
                    cleaned_conditions,
                ]
                if value
            )
        )

        cleaned_content = self._build_cleaned_content(event)
        embedding_content = self._build_embedding_content(cleaned_content)

        metadata = {
            "uid": event.id,
            "slug": event.slug,
            "title": event.title,
            "url": event.url,
            "image_url": event.image_url,
            "thumbnail_url": event.thumbnail_url,
            "original_image_url": event.original_image_url,
            "description": event.description,
            "long_description": event.long_description,
            "conditions": event.conditions,
            "keywords": list(event.keywords or []),
            "location_name": event.location_name,
            "address": event.address,
            "district": event.district,
            "postal_code": event.postal_code,
            "department": event.department,
            "region": event.region,
            "country": event.country,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "first_date": (event.first_date.isoformat() if event.first_date else None),
            "last_date": (event.last_date.isoformat() if event.last_date else None),
            "first_date_raw": event.first_date_raw,
            "first_date_end_raw": event.first_date_end_raw,
            "last_date_raw": event.last_date_raw,
            "last_date_end_raw": event.last_date_end_raw,
            "date_range": event.date_range,
            "timings": event.timings,
            "location_website": event.location_website,
            "location_description": event.location_description,
            "location_access": event.location_access,
            "location_image": event.location_image,
            "location_image_credits": event.location_image_credits,
            "accessibility": event.accessibility,
            "accessibility_label": event.accessibility_label,
            "attendance_mode": event.attendance_mode,
            "online_access_link": event.online_access_link,
            "status": event.status,
            "age_min": age_info["age_min"],
            "age_max": age_info["age_max"],
            "age_source": age_info["age_source"],
            "audience_categories": audiences,
            "price_status": price_info["price_status"],
            "price_amounts_eur": price_info["price_amounts_eur"],
            "registration": event.registration,
            "origin_agenda_title": event.origin_agenda_title,
            "origin_agenda_uid": event.origin_agenda_uid,
            "updated_at": event.updated_at,
            "image_credits": event.image_credits,
            "links": event.links,
            "location_links": event.location_links,
        }

        metadata = {key: value for key, value in metadata.items() if value is not None}

        return {
            "id": event.id,
            "page_content": embedding_content,
            "metadata": metadata,
        }
