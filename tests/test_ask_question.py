import pytest

from app.core.application.use_cases.ask_question import AskQuestion


class FakeAnswerer:
    def answer(self, question: str) -> str:
        return f"Réponse à : {question}"


def test_execute_normalizes_question() -> None:
    result = AskQuestion(FakeAnswerer()).execute("  Que voir à Pau ?  ")
    assert result.question == "Que voir à Pau ?"
    assert result.answer == "Réponse à : Que voir à Pau ?"


def test_execute_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="vide"):
        AskQuestion(FakeAnswerer()).execute("   ")
