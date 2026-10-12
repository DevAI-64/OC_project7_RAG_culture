from datetime import date
from types import SimpleNamespace

from app.infrastructure.rag.rag_question_answerer import RagQuestionAnswerer


class FakeRetriever:
    def retrieve(self, **kwargs):
        return [(SimpleNamespace(metadata={"uid": "1"}, page_content="A"), 0.1)]

    def from_documents(self, documents, limit=5):
        return [(SimpleNamespace(metadata={"uid": "1"}, page_content="A"), 0.0)]


class FakeAnswerGenerator:
    def generate(self, question, results):
        return f"{question}: {len(results)}"


def make_answerer(catalog):
    answerer = RagQuestionAnswerer(
        vector_store_provider=lambda: object(),
        catalog_provider=lambda: catalog,
        today_provider=lambda: date(2026, 9, 30),
    )
    answerer._event_retriever = FakeRetriever()
    answerer._answer_generator = FakeAnswerGenerator()
    return answerer


def test_answer_uses_semantic_retrieval_when_theme_exists() -> None:
    catalog = [{"metadata": {"first_date": "2026-10-01", "last_date": "2026-10-01", "uid": "1"}, "page_content": "photo"}]
    assert make_answerer(catalog).answer("activité photo en octobre") == "activité photo en octobre: 1"


def test_answer_uses_filtered_documents_for_generic_query() -> None:
    catalog = [{"metadata": {"first_date": "2026-10-01", "last_date": "2026-10-01", "uid": "1"}, "page_content": "événement"}]
    assert make_answerer(catalog).answer("Quels événements sont prévus en octobre ?") == "Quels événements sont prévus en octobre ?: 1"


def test_answer_returns_no_match_message() -> None:
    answerer = make_answerer([])
    assert "pas trouvé" in answerer.answer("événements en octobre")
