from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("dummy_mistral_key")


@pytest.fixture
def dummy_mistral_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


def import_main():
    import importlib
    return importlib.import_module("app.main")


def test_index_exists_and_path_resolution(tmp_path, monkeypatch) -> None:
    main = import_main()
    monkeypatch.setattr(main, "INDEX_FILE", tmp_path / "index.faiss")
    monkeypatch.setattr(main, "INDEX_METADATA_FILE", tmp_path / "index.pkl")
    assert not main._index_exists()
    (tmp_path / "index.faiss").touch()
    (tmp_path / "index.pkl").touch()
    assert main._index_exists()
    monkeypatch.setattr(main, "ROOT_DIR", tmp_path)
    assert main._resolve_path("data/test.json") == tmp_path / "data/test.json"
    absolute = tmp_path / "absolute.json"
    assert main._resolve_path(absolute) == absolute


def test_dependency_providers(monkeypatch) -> None:
    main = import_main()
    main.app.state.vector_store = "store"
    main.app.state.event_catalog = [{"metadata": {"uid": "1"}}]
    assert main.get_vector_store() == "store"
    assert main.get_event_catalog() == [{"metadata": {"uid": "1"}}]

    main.app.state.vector_store = None
    with pytest.raises(RuntimeError):
        main.get_vector_store()
    main.app.state.event_catalog = []
    with pytest.raises(RuntimeError):
        main.get_event_catalog()


def test_health_check() -> None:
    main = import_main()
    main.app.state.vector_store = None
    assert main.health_check() == {"status": "ok", "vector_store_available": False}
    main.app.state.vector_store = object()
    assert main.health_check()["vector_store_available"] is True


def test_chat_success(monkeypatch) -> None:
    main = import_main()

    class Result:
        question = "Question"
        answer = "Réponse"

    monkeypatch.setattr(main.ask_question_use_case, "execute", lambda _: Result())
    response = main.chat(main.ChatRequest(question="Question"))
    assert response.question == "Question"
    assert response.answer == "Réponse"


def test_chat_maps_known_errors(monkeypatch) -> None:
    main = import_main()
    monkeypatch.setattr(main.ask_question_use_case, "execute", lambda _: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(main.HTTPException) as exc:
        main.chat(main.ChatRequest(question="Question"))
    assert exc.value.status_code == 422

    monkeypatch.setattr(main.ask_question_use_case, "execute", lambda _: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(main.HTTPException) as exc:
        main.chat(main.ChatRequest(question="Question"))
    assert exc.value.status_code == 503

    monkeypatch.setattr(main.ask_question_use_case, "execute", lambda _: (_ for _ in ()).throw(Exception("boom")))
    with pytest.raises(main.HTTPException) as exc:
        main.chat(main.ChatRequest(question="Question"))
    assert exc.value.status_code == 500


def test_rebuild_publishes_new_state(monkeypatch, tmp_path) -> None:
    main = import_main()
    processed = tmp_path / "documents.json"
    processed.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(main.ingestion_pipeline, "run", lambda: {"processed_path": str(processed)})
    monkeypatch.setattr(main.faiss_indexer, "rebuild", lambda documents_path: 12)
    monkeypatch.setattr(main, "_load_vector_store", lambda: "new-store")
    monkeypatch.setattr(main.event_catalog_repository, "load", lambda path: None)
    monkeypatch.setattr(main.event_catalog_repository, "all", lambda: [{"metadata": {"uid": "1"}}])

    result = main.rebuild_index()
    assert result["status"] == "ok"
    assert result["indexed_count"] == 12
    assert main.app.state.vector_store == "new-store"
    assert len(main.app.state.event_catalog) == 1


def test_rebuild_returns_http_error_on_failure(monkeypatch) -> None:
    main = import_main()
    monkeypatch.setattr(main.ingestion_pipeline, "run", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(main.HTTPException) as exc:
        main.rebuild_index()
    assert exc.value.status_code == 500


def test_load_vector_store_requires_api_key(monkeypatch) -> None:
    main = import_main()
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(main, "load_dotenv", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        main._load_vector_store()
