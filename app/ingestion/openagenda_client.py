import logging
from datetime import date, datetime, timedelta
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)

CITY = "Pau"
PAGE_SIZE = 100


class OpenAgendaClient:
    """Client for retrieving OpenAgenda events through the Opendatasoft API."""

    def __init__(
        self,
        api_url: str = API_URL,
        city: str = CITY,
        page_size: int = PAGE_SIZE,
    ) -> None:
        """Initialize the OpenAgenda client.

        Args:
            api_url: URL of the Opendatasoft API endpoint.
            city: City used to filter events.
            page_size: Maximum number of events requested per API call.
        """
        self._api_url = api_url
        self._city = city
        self._page_size = page_size

    @staticmethod
    def _create_session() -> requests.Session:
        """Create an HTTP session configured with retry handling.

        Returns:
            A configured HTTP session.
        """
        session = requests.Session()

        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)

        return session

    def _build_where_clause(self, start_date: date) -> str:
        """Build the ODSQL filter used to retrieve relevant events.

        Args:
            start_date: Earliest accepted event end date.

        Returns:
            An ODSQL expression filtering events by city and end date.
        """
        return (
            f'location_city = "{self._city}" '
            f"AND lastdate_begin >= date'{start_date.isoformat()}'"
        )

    def fetch_events(self) -> dict[str, Any]:
        """Fetch all matching events from the OpenAgenda dataset.

        Returns:
            A snapshot containing retrieval metadata and the raw event records.

        Raises:
            requests.RequestException: If an HTTP request fails after retries.
            ValueError: If an API response cannot be decoded as JSON.
        """
        today = date.today()
        start_date = today - timedelta(days=365)
        fetched_at = datetime.now().astimezone()
        where = self._build_where_clause(start_date)

        logging.info("Ville : %s", self._city)
        logging.info("Période à partir du : %s", start_date)
        logging.info("Filtre ODSQL : %s", where)

        events: list[dict[str, Any]] = []
        offset = 0
        total_count: int | None = None

        with self._create_session() as session:
            while True:
                response = session.get(
                    self._api_url,
                    params={
                        "where": where,
                        "limit": self._page_size,
                        "offset": offset,
                    },
                    timeout=(10, 60),
                )
                response.raise_for_status()

                payload = response.json()
                results = payload.get("results", [])

                if total_count is None:
                    total_count = payload.get("total_count")
                    logging.info(
                        "Total annoncé par l'API : %s",
                        total_count,
                    )

                if not results:
                    break

                events.extend(results)
                offset += len(results)

                logging.info(
                    "Événements récupérés : %d%s",
                    len(events),
                    (f" / {total_count}" if total_count is not None else ""),
                )

                if total_count is not None and offset >= total_count:
                    break

        return {
            "metadata": {
                "source": self._api_url,
                "city": self._city,
                "start_date": start_date.isoformat(),
                "end_date": None,
                "fetched_at": fetched_at.isoformat(),
                "count": len(events),
            },
            "events": events,
        }
