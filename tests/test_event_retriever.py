from langchain_core.documents import Document

from app.infrastructure.rag.event_retriever import EventRetriever


def test_from_documents_converts_documents_and_applies_limit() -> None:
    retriever = EventRetriever()
    documents = [
        {"page_content": "A", "metadata": {"uid": "1"}},
        {"page_content": "B", "metadata": {"uid": "2"}},
    ]
    results = retriever.from_documents(documents, limit=1)
    assert len(results) == 1
    assert results[0][0].page_content == "A"
    assert results[0][1] == 0.0


def test_retrieve_filters_by_uid_and_deduplicates() -> None:
    class Index:
        ntotal = 3

    class Store:
        index = Index()

        def similarity_search_with_score(self, question, k, fetch_k, filter):
            candidates = [
                (Document(page_content="A1", metadata={"uid": "1"}), 0.2),
                (Document(page_content="A2", metadata={"uid": "1"}), 0.3),
                (Document(page_content="B", metadata={"uid": "2"}), 0.4),
            ]
            return [item for item in candidates if filter(item[0].metadata)]

    results = EventRetriever().retrieve(
        "photo",
        Store(),
        [{"metadata": {"uid": "1"}}, {"metadata": {"uid": "2"}}],
        limit=2,
    )
    assert [doc.page_content for doc, _ in results] == ["A1", "B"]


def test_retrieve_rejects_invalid_limit_and_empty_documents() -> None:
    retriever = EventRetriever()
    try:
        retriever.retrieve("q", None, [], 0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
    assert retriever.retrieve("q", None, [], 1) == []


def test_deduplicate_and_extract_uids() -> None:
    candidates = [
        (Document(page_content="A", metadata={"uid": "1"}), 1.0),
        (Document(page_content="A2", metadata={"uid": "1"}), 2.0),
        (Document(page_content="B", metadata={"uid": "2"}), 3.0),
    ]
    assert EventRetriever._extract_uids([{"metadata": {"uid": "1"}}]) == {"1"}
    assert [
        doc.page_content for doc, _ in EventRetriever._deduplicate(candidates, 2)
    ] == ["A", "B"]
