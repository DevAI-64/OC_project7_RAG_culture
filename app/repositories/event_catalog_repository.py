import json
from pathlib import Path
from typing import Any


class EventCatalogRepository:
    """Provide access to the structured event catalog used by RAG filters."""

    def __init__(self, root_dir: Path) -> None:
        """Initialize the repository with the project root directory.

        Args:
            root_dir: Root directory used to resolve relative catalog paths.
        """
        self._root_dir = root_dir
        self._documents: list[dict[str, Any]] = []
        self._source_path: Path | None = None

    @property
    def source_path(self) -> Path | None:
        """Return the path of the currently loaded catalog.

        Returns:
            Path of the loaded catalog, or None if no catalog has been loaded.
        """
        return self._source_path

    def load(self, path: str | Path) -> None:
        """Load event documents from a JSON catalog.

        Args:
            path: Path to the JSON catalog.

        Raises:
            FileNotFoundError: If the catalog does not exist.
            OSError: If the catalog cannot be opened or read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            ValueError: If the JSON structure does not contain a valid
                documents list.
        """
        source_path = Path(path)

        if not source_path.is_absolute():
            source_path = self._root_dir / source_path

        with source_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        documents = self._extract_documents(data)

        self._documents = [
            document for document in documents if isinstance(document, dict)
        ]
        self._source_path = source_path

    def load_latest(self) -> None:
        """Load the most recently modified processed event catalog.

        Raises:
            FileNotFoundError: If no document snapshot is available.
            OSError: If a snapshot cannot be inspected or loaded.
        """
        directory = self._root_dir / "data" / "processed" / "openagenda"
        files = sorted(
            directory.glob("*_documents.json"),
            key=lambda path: path.stat().st_mtime,
        )

        if not files:
            raise FileNotFoundError(
                f"Aucun snapshot de documents trouvé dans : {directory}"
            )

        self.load(files[-1])

    def all(self) -> list[dict[str, Any]]:
        """Return all documents from the currently loaded catalog.

        Returns:
            A new list containing the loaded event documents.
        """
        return list(self._documents)

    @staticmethod
    def _extract_documents(data: Any) -> list[Any]:
        """Extract the document list from a supported catalog structure.

        Args:
            data: Decoded JSON content.

        Returns:
            The list of documents contained in the catalog.

        Raises:
            ValueError: If the JSON structure is not supported or does not
                contain a valid documents list.
        """
        if isinstance(data, dict):
            documents = data.get("documents")
        elif isinstance(data, list):
            documents = data
        else:
            raise ValueError("Format JSON des documents non reconnu.")

        if not isinstance(documents, list):
            raise ValueError("Le fichier JSON doit contenir une liste 'documents'.")

        return documents
