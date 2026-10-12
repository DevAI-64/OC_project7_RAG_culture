import json
import os
import tempfile
from pathlib import Path
from typing import Any


class JsonSnapshotRepository:
    """Persist JSON snapshots safely on the filesystem."""

    def save(
        self,
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        """Save a JSON payload to the specified path.

        Args:
            path: Destination path of the JSON snapshot.
            payload: JSON-serializable data to persist.

        Raises:
            OSError: If the destination directory cannot be created,
                the temporary file cannot be written, the destination
                cannot be replaced, or the temporary file cannot be
                removed.
            TypeError: If the payload contains values that cannot be
                serialized to JSON.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = self._write_temp_file(path, payload)

        try:
            os.replace(temp_path, path)
        finally:
            self._remove_temp_file(temp_path)

    @staticmethod
    def _write_temp_file(
        path: Path,
        payload: dict[str, Any],
    ) -> Path:
        """Write a JSON payload to a temporary file.

        Args:
            path: Destination path used to determine the temporary file's
                parent directory.
            payload: JSON-serializable data to write.

        Returns:
            Path: Path to the newly created temporary file.

        Raises:
            OSError: If the temporary file cannot be created or written.
            TypeError: If the payload contains values that cannot be
                serialized to JSON.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                indent=2,
            )

            return Path(temp_file.name)

    @staticmethod
    def _remove_temp_file(path: Path) -> None:
        """Remove a temporary file if it still exists.

        Args:
            path: Path to the temporary file to remove.

        Raises:
            OSError: If the file exists but cannot be removed.
        """
        if path.exists():
            path.unlink()
