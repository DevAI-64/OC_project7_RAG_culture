import json
from pathlib import Path

import pytest

from app.repositories.event_catalog_repository import EventCatalogRepository
from app.repositories.json_snapshot_repository import JsonSnapshotRepository


def test_json_snapshot_repository_saves_atomically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "snapshot.json"
    JsonSnapshotRepository().save(path, {"documents": [{"id": 1}]})
    assert json.loads(path.read_text(encoding="utf-8")) == {"documents": [{"id": 1}]}
    assert list(path.parent.glob("*.tmp")) == []


def test_json_snapshot_repository_removes_temp_file_after_replace_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = JsonSnapshotRepository()
    path = tmp_path / "snapshot.json"

    monkeypatch.setattr("app.repositories.json_snapshot_repository.os.replace", lambda *_: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        repository.save(path, {"ok": True})
    assert list(tmp_path.glob("*.tmp")) == []


def test_catalog_repository_loads_wrapped_and_list_formats(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"documents": [{"id": "1"}, "ignored"]}), encoding="utf-8")
    repo = EventCatalogRepository(tmp_path)
    repo.load(path)
    assert repo.all() == [{"id": "1"}]
    assert repo.source_path == path

    path.write_text(json.dumps([{"id": "2"}]), encoding="utf-8")
    repo.load(path)
    assert repo.all() == [{"id": "2"}]


def test_catalog_repository_rejects_invalid_structure(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"foo": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        EventCatalogRepository(tmp_path).load(path)


def test_catalog_repository_load_latest(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "processed" / "openagenda"
    directory.mkdir(parents=True)
    old = directory / "2026_09_29_documents.json"
    new = directory / "2026_09_30_documents.json"
    old.write_text(json.dumps({"documents": [{"id": "old"}]}), encoding="utf-8")
    new.write_text(json.dumps({"documents": [{"id": "new"}]}), encoding="utf-8")
    repo = EventCatalogRepository(tmp_path)
    repo.load_latest()
    assert repo.all() == [{"id": "new"}]


def test_catalog_repository_load_latest_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EventCatalogRepository(tmp_path).load_latest()
