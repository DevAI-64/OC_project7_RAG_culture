import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date

MONTHS = {
    "janvier": 1,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}


@dataclass(frozen=True)
class QueryCriteria:
    """Structured criteria extracted from a user's question."""

    period: tuple[date, date] | None = None
    max_price: float | None = None
    max_price_inclusive: bool = True
    audience: str | None = None
    accessibility_requested: bool = False
    semantic_query: str = ""


class QueryParser:
    """Extract explicit filtering criteria from a user question."""

    PRICE_PATTERNS = (
        (
            re.compile(
                r"\b(?:moins de|inférieur(?:e)? à|sous)"
                r"\s+(\d+(?:[,.]\d+)?)"
                r"\s*(?:€|euros?)?",
                re.IGNORECASE,
            ),
            False,
        ),
        (
            re.compile(
                r"\b(?:maximum|max(?:imum)?|au plus|jusqu['’]à)"
                r"\s+(\d+(?:[,.]\d+)?)"
                r"\s*(?:€|euros?)?",
                re.IGNORECASE,
            ),
            True,
        ),
    )

    AUDIENCE_TERMS = (
        (
            "enfants",
            (
                "enfant",
                "enfants",
                "jeune public",
                "petite enfance",
                "bébé",
                "bebes",
            ),
        ),
        (
            "adolescents",
            ("adolescent", "adolescents", "ados"),
        ),
        (
            "adultes",
            ("adulte", "adultes"),
        ),
        (
            "familles",
            ("famille", "familles", "en famille"),
        ),
    )

    ACCESSIBILITY_TERMS = (
        "accessible aux personnes à mobilité réduite",
        "accessibilité",
        "accessible pmr",
        "pmr",
        "fauteuil roulant",
        "fauteuils roulants",
        "mobilité réduite",
        "personnes handicapées",
        "handicap",
    )

    def parse(
        self,
        question: str,
        today: date | None = None,
    ) -> QueryCriteria:
        """Parse a user question into structured query criteria.

        Args:
            question: Natural-language question asked by the user.
            today: Reference date used to resolve relative periods.

        Returns:
            Structured criteria used by deterministic filtering
            and semantic retrieval.

        Raises:
            ValueError: If the question is empty.
        """
        if not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        reference_date = today or date.today()

        price, inclusive = self._extract_price(question)

        return QueryCriteria(
            period=self._extract_period(
                question,
                reference_date,
            ),
            max_price=price,
            max_price_inclusive=inclusive,
            audience=self._extract_audience(question),
            accessibility_requested=self._extract_accessibility(
                question,
            ),
            semantic_query=self._build_semantic_query(question),
        )

    def _extract_period(
        self,
        question: str,
        today: date,
    ) -> tuple[date, date] | None:
        """Extract a requested calendar period.

        Args:
            question: User's natural-language question.
            today: Reference date used to resolve a month without a year.

        Returns:
            The requested period, or None if no calendar period is found.
        """
        normalized = question.lower()

        if "mois prochain" in normalized:
            year = today.year + (today.month == 12)
            month = 1 if today.month == 12 else today.month + 1
            return self._month_period(year, month)

        if "ce mois-ci" in normalized or "ce mois ci" in normalized:
            return self._month_period(
                today.year,
                today.month,
            )

        pattern = r"\b(" + "|".join(MONTHS.keys()) + r")\s*(?:de\s*)?(\d{4})?\b"

        match = re.search(
            pattern,
            normalized,
        )

        if not match:
            return None

        month = MONTHS[match.group(1)]
        year_text = match.group(2)

        if year_text:
            year = int(year_text)
        else:
            year = today.year

            if month < today.month:
                year += 1

        return self._month_period(year, month)

    @staticmethod
    def _month_period(
        year: int,
        month: int,
    ) -> tuple[date, date]:
        """Return the first and last day of a calendar month.

        Args:
            year: Calendar year.
            month: Calendar month.

        Returns:
            The first and last day of the requested month.
        """
        return (
            date(year, month, 1),
            date(
                year,
                month,
                monthrange(year, month)[1],
            ),
        )

    def _extract_price(
        self,
        question: str,
    ) -> tuple[float | None, bool]:
        """Extract a maximum price constraint.

        Args:
            question: User's natural-language question.

        Returns:
            The maximum price and whether the limit is inclusive.
        """
        for pattern, inclusive in self.PRICE_PATTERNS:
            match = pattern.search(question)

            if match:
                price = float(match.group(1).replace(",", "."))
                return price, inclusive

        return None, True

    def _extract_audience(
        self,
        question: str,
    ) -> str | None:
        """Extract the requested audience.

        Args:
            question: User's natural-language question.

        Returns:
            The detected audience category, or None.
        """
        normalized = question.lower()

        for audience, terms in self.AUDIENCE_TERMS:
            if any(
                re.search(
                    r"\b" + re.escape(term) + r"\b",
                    normalized,
                )
                for term in terms
            ):
                return audience

        return None

    def _extract_accessibility(
        self,
        question: str,
    ) -> bool:
        """Check whether accessibility is explicitly requested.

        Args:
            question: User's natural-language question.

        Returns:
            True when accessibility is explicitly requested.
        """
        normalized = question.lower()

        return any(term in normalized for term in self.ACCESSIBILITY_TERMS)

    def _build_semantic_query(
        self,
        question: str,
    ) -> str:
        """Remove deterministic constraints from the semantic query.

        Args:
            question: User's original question.

        Returns:
            The remaining semantic theme, or an empty string when the
            question contains no semantic theme.
        """
        query = question

        # Relative periods.
        query = re.sub(
            r"\b(?:le|en|pour|du|de|des|au|aux)\s+mois prochain\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        query = re.sub(
            r"\b(?:ce mois-ci|ce mois ci)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Explicit months.
        # Remove the preposition when it directly introduces the month,
        # for example: "en octobre", "pour octobre", "au mois d'octobre".
        query = re.sub(
            r"\b(?:en|pour|au mois de|le mois de)\s+"
            r"(?:" + "|".join(MONTHS.keys()) + r")"
            r"(?:\s+\d{4})?\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        query = re.sub(
            r"\b(?:" + "|".join(MONTHS.keys()) + r")" r"(?:\s+\d{4})?\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Price constraints.
        # The optional leading "à" handles formulations such as:
        # "à moins de 10 €".
        query = re.sub(
            r"(?:\bà\s+)?"
            r"(?:moins de|inférieur(?:e)? à|sous|"
            r"maximum|max(?:imum)?|au plus|jusqu['’]à)"
            r"\s+\d+(?:[,.]\d+)?"
            r"\s*(?:€|euros?|EUR)?",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Audience constraints.
        query = re.sub(
            r"\b(?:pour les?|destiné(?:e)?s? aux?|"
            r"à destination des?)\s+"
            r"(?:enfants?|adolescents?|adultes?|familles?)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Accessibility constraints.
        query = re.sub(
            r"\b(?:accessible aux personnes à mobilité réduite|"
            r"accessible pmr|pmr|fauteuils? roulants?|"
            r"mobilité réduite|personnes handicapées?)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Generic question formulations.
        query = re.sub(
            r"\b(?:quels?|quelles?)\s+" r"(?:événements?|activités?|sorties?)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Verbal formulations such as:
        # "sont prévues", "sont proposées" and "sont organisées".
        query = re.sub(
            r"\b(?:sont\s+(?:prévues?|proposées?|organisées?)|"
            r"sont|peut-on|peut on)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Remaining participles from generic formulations.
        query = re.sub(
            r"\b(?:prévus?|prévues?|proposés?|proposées?|" r"organisés?|organisées?)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Pau is already the fixed location of the application.
        query = re.sub(
            r"\b(?:à pau|sur pau|dans pau)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )

        # Normalize whitespace and punctuation.
        query = re.sub(r"\s+", " ", query)
        query = query.strip(" ,;?.")

        return query
