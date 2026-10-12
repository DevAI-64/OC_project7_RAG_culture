from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_community.vectorstores import FAISS

from app.infrastructure.rag.answer_generator import AnswerGenerator
from app.infrastructure.rag.event_filter import EventFilter
from app.infrastructure.rag.event_retriever import EventRetriever
from app.infrastructure.rag.query_parser import QueryParser


class RagQuestionAnswerer:
    """Orchestrate query parsing, filtering, retrieval, and generation."""

    def __init__(
        self,
        vector_store_provider: Callable[[], FAISS],
        catalog_provider: Callable[[], list[dict[str, Any]]],
        today_provider: Callable[[], date] | None = None,
    ) -> None:
        """Initialize the RAG question answerer.

        Args:
            vector_store_provider: Provider returning the loaded FAISS
                vector store.
            catalog_provider: Provider returning the structured event
                catalog.
            today_provider: Provider returning the current date.
                Defaults to date.today.
        """
        self._vector_store_provider = vector_store_provider
        self._catalog_provider = catalog_provider
        self._today_provider = today_provider or date.today

        self._query_parser = QueryParser()
        self._event_filter = EventFilter()
        self._event_retriever = EventRetriever()
        self._answer_generator = AnswerGenerator()

    def answer(self, question: str) -> str:
        """Answer a user question using the RAG pipeline.

        Args:
            question: Natural-language question asked by the user.

        Returns:
            A natural-language answer based on matching events.
        """
        today = self._today_provider()

        criteria = self._query_parser.parse(
            question,
            today=today,
        )

        catalog = self._catalog_provider()

        eligible = self._event_filter.filter(
            catalog,
            criteria,
            today=today,
        )

        if not eligible:
            return (
                "Je n'ai pas trouvé d'événement correspondant "
                "aux critères demandés dans les données disponibles."
            )

        if criteria.semantic_query:
            vector_store = self._vector_store_provider()

            results = self._event_retriever.retrieve(
                question=criteria.semantic_query,
                vector_store=vector_store,
                eligible_documents=eligible,
                limit=5,
            )
        else:
            results = self._event_retriever.from_documents(
                eligible,
                limit=5,
            )

        return self._answer_generator.generate(
            question,
            results,
        )
