from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class EventRetriever:
    """Retrieve eligible events by semantic relevance."""

    def retrieve(
        self,
        question: str,
        vector_store: FAISS,
        eligible_documents: list[dict[str, Any]],
        limit: int = 5,
    ) -> list[tuple[Document, float]]:
        """Retrieve the most relevant eligible events.

        Args:
            question: Semantic query used to rank the eligible events.
            vector_store: Loaded FAISS vector store.
            eligible_documents: Documents that passed deterministic filters.
            limit: Maximum number of distinct events to return.

        Returns:
            A list of distinct events paired with their relevance scores.

        Raises:
            ValueError: If limit is less than 1.
        """
        if limit < 1:
            raise ValueError("limit doit être supérieur ou égal à 1.")

        if not eligible_documents:
            return []

        allowed_uids = self._extract_uids(
            eligible_documents,
        )

        total = max(
            1,
            vector_store.index.ntotal,
        )

        candidates = vector_store.similarity_search_with_score(
            question,
            k=total,
            fetch_k=total,
            filter=lambda metadata: (str(metadata.get("uid")) in allowed_uids),
        )

        return self._deduplicate(
            candidates,
            limit,
        )

    def from_documents(
        self,
        documents: list[dict[str, Any]],
        limit: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Convert filtered event documents into answerable documents.

        Args:
            documents: Event documents that passed deterministic filtering.
            limit: Optional maximum number of documents to return.

        Returns:
            Event documents converted to LangChain documents with a
            neutral score of zero.
        """
        results: list[tuple[Document, float]] = []

        for document in documents:
            metadata = document.get("metadata") or {}

            results.append(
                (
                    Document(
                        page_content=str(document.get("page_content") or ""),
                        metadata=metadata,
                    ),
                    0.0,
                )
            )

            if limit is not None and len(results) >= limit:
                break

        return results

    @staticmethod
    def _extract_uids(
        documents: list[dict[str, Any]],
    ) -> set[str]:
        """Extract event identifiers from eligible documents.

        Args:
            documents: Documents that passed deterministic filtering.

        Returns:
            Set of event identifiers allowed in semantic retrieval.
        """
        return {
            str((document.get("metadata") or {}).get("uid")) for document in documents
        }

    @staticmethod
    def _deduplicate(
        candidates: list[tuple[Document, float]],
        limit: int,
    ) -> list[tuple[Document, float]]:
        """Keep one FAISS result per event.

        Args:
            candidates: Semantic search results potentially containing
                multiple chunks from the same event.
            limit: Maximum number of distinct events to return.

        Returns:
            Deduplicated semantic search results.
        """
        results: list[tuple[Document, float]] = []
        seen: set[str] = set()

        for document, score in candidates:
            uid = str(document.metadata.get("uid"))

            if uid in seen:
                continue

            seen.add(uid)
            results.append((document, score))

            if len(results) >= limit:
                break

        return results
