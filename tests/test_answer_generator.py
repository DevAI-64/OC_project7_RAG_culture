from types import SimpleNamespace

from app.infrastructure.rag.answer_generator import AnswerGenerator


def test_generate_returns_no_result_message_without_calling_model(monkeypatch) -> None:
    generator = object.__new__(AnswerGenerator)
    generator._chain = SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(AssertionError()))
    result = generator.generate("question", [])
    assert "pas trouvé" in result


def test_build_context_contains_structured_event_information() -> None:
    from langchain_core.documents import Document

    document = Document(
        page_content="Atelier de céramique",
        metadata={
            "address": "1 rue Test",
            "first_date": "2026-10-20",
            "last_date": "2026-10-20",
            "conditions": "Gratuit",
            "audience_categories": ["enfants"],
            "accessibility_label": "Accessible PMR",
            "url": "https://example.test",
        },
    )
    context = AnswerGenerator._build_context([(document, 0.1)])
    assert "Atelier de céramique" in context
    assert "1 rue Test" in context
    assert "enfants" in context
    assert "Accessible PMR" in context


def test_generate_invokes_chain_and_strips_response() -> None:
    generator = object.__new__(AnswerGenerator)
    calls = []

    class Chain:
        def invoke(self, payload):
            calls.append(payload)
            return SimpleNamespace(content="  Réponse finale  ")

    generator._chain = Chain()

    from langchain_core.documents import Document

    result = generator.generate(
        "Question",
        [(Document(page_content="Événement", metadata={}), 0.2)],
    )
    assert result == "Réponse finale"
    assert calls[0]["question"] == "Question"
