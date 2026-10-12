from datetime import date
from pathlib import Path
from typing import Any

from app.ingestion.document_builder import DocumentBuilder
from app.ingestion.event_processor import EventProcessor
from app.ingestion.openagenda_client import OpenAgendaClient
from app.repositories.json_snapshot_repository import (
    JsonSnapshotRepository,
)


class IngestionPipeline:
    """Orchestrate event ingestion, processing, and document generation."""

    def __init__(
        self,
        client: OpenAgendaClient,
        processor: EventProcessor,
        document_builder: DocumentBuilder,
        snapshot_repository: JsonSnapshotRepository,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            client: Client used to retrieve events from OpenAgenda.
            processor: Processor used to convert raw events into domain events.
            document_builder: Builder used to create searchable documents.
            snapshot_repository: Repository used to persist ingestion snapshots.
        """
        self._client = client
        self._processor = processor
        self._document_builder = document_builder
        self._snapshot_repository = snapshot_repository

    def run(self) -> dict[str, Any]:
        """Run the complete event ingestion pipeline.

        Returns:
            A summary containing the number of raw events, processed events,
            generated documents, and the paths of the two snapshots.

        Raises:
            OSError: If a snapshot cannot be persisted.
            KeyError: If the OpenAgenda response does not contain an
                ``events`` field.
        """
        date_suffix = date.today().strftime("%Y_%m_%d")

        raw_path = Path(f"data/raw/openagenda/{date_suffix}_events.json")
        processed_path = Path(f"data/processed/openagenda/{date_suffix}_documents.json")

        raw_snapshot = self._client.fetch_events()

        self._snapshot_repository.save(
            raw_path,
            raw_snapshot,
        )

        raw_events = raw_snapshot["events"]
        events = self._processor.process(raw_events)

        documents = [self._document_builder.build(event) for event in events]

        processed_snapshot = {
            "metadata": {
                **raw_snapshot["metadata"],
                "processed_at": date.today().isoformat(),
                "count": len(documents),
            },
            "documents": documents,
        }

        self._snapshot_repository.save(
            processed_path,
            processed_snapshot,
        )

        return {
            "raw_count": len(raw_events),
            "processed_count": len(events),
            "document_count": len(documents),
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
        }
