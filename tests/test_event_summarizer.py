from types import SimpleNamespace

import pytest

from app.infrastructure.llm.event_summarizer import EventSummarizer


def test_summarizer_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        EventSummarizer("test-model")


def test_summarizer_returns_empty_for_empty_text(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    summarizer = EventSummarizer("test-model")
    summarizer.client = SimpleNamespace()
    assert summarizer.summarize("   ") == ""


def test_summarizer_returns_model_content(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    summarizer = EventSummarizer("test-model")
    summarizer.client = SimpleNamespace(
        chat=SimpleNamespace(
            complete=lambda **_: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" Résumé "))]
            )
        )
    )
    assert summarizer.summarize("Texte") == "Résumé"


def test_summarizer_rejects_empty_model_content(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    summarizer = EventSummarizer("test-model")
    summarizer.client = SimpleNamespace(
        chat=SimpleNamespace(
            complete=lambda **_: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" "))]
            )
        )
    )
    with pytest.raises(ValueError, match="vide"):
        summarizer.summarize("Texte")
