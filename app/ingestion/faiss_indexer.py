import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "mistral-embed"

# Taille prudente pour rester sous la limite de tokens Mistral.
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300


class FaissIndexer:
    """Build and persist a FAISS index from processed event documents."""

    def __init__(
        self,
        index_path: str | Path | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """Initialize the FAISS indexer.

        Args:
            index_path: Destination directory for the FAISS index. If omitted,
                the default processed-data directory is used.
            chunk_size: Maximum number of characters in a document fragment.
            chunk_overlap: Number of characters shared between consecutive
                fragments.

        Raises:
            ValueError: If ``chunk_size`` is not positive or if
                ``chunk_overlap`` is outside the valid range.
        """
        root_dir = Path(__file__).resolve().parents[2]

        env_path = root_dir / ".env"
        load_dotenv(dotenv_path=env_path)

        self.index_path = Path(
            index_path or root_dir / "data" / "processed" / "faiss_index"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if self.chunk_size <= 0:
            raise ValueError("chunk_size doit être supérieur à zéro.")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap doit être positif et inférieur à chunk_size."
            )

        # Initialized lazily so FastAPI can start without requiring
        # the Mistral API key immediately.
        self._embeddings: MistralAIEmbeddings | None = None

    def _get_embeddings(self) -> MistralAIEmbeddings:
        """Return the configured Mistral embeddings instance.

        Returns:
            The Mistral embeddings instance.

        Raises:
            OSError: If ``MISTRAL_API_KEY`` is not available.
        """
        if self._embeddings is not None:
            return self._embeddings

        root_dir = Path(__file__).resolve().parents[2]
        env_path = root_dir / ".env"

        load_dotenv(dotenv_path=env_path, override=False)

        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise OSError(
                f"MISTRAL_API_KEY introuvable. Vérifie le fichier : {env_path}"
            )

        self._embeddings = MistralAIEmbeddings(
            model=EMBEDDING_MODEL,
            mistral_api_key=api_key,
        )

        return self._embeddings

    def _load_documents(
        self,
        documents_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Load processed documents from a JSON file.

        Args:
            documents_path: Path to the processed JSON document file.

        Returns:
            A list of raw document dictionaries.

        Raises:
            FileNotFoundError: If the document file does not exist.
            OSError: If the file cannot be opened or read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            ValueError: If the JSON structure is not supported.
        """
        path = Path(documents_path)

        if not path.is_absolute():
            root_dir = Path(__file__).resolve().parents[2]
            path = root_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Fichier de documents introuvable : {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            raw_documents = data.get("documents")

            if not isinstance(raw_documents, list):
                raise ValueError("Le fichier JSON doit contenir une liste 'documents'.")

        elif isinstance(data, list):
            raw_documents = data

        else:
            raise ValueError("Format JSON des documents non reconnu.")

        logger.info(
            "%d documents chargés depuis %s",
            len(raw_documents),
            path,
        )

        return raw_documents

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping fragments.

        Args:
            text: Text to split into fragments.

        Returns:
            A list of non-empty text fragments.
        """
        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        text_length = len(text)
        start = 0

        while start < text_length:
            end = min(
                start + self.chunk_size,
                text_length,
            )

            if end < text_length:
                search_start = start + int(self.chunk_size * 0.6)
                window = text[search_start:end]

                cut_positions = [
                    window.rfind("\n\n"),
                    window.rfind("\n"),
                    window.rfind(" "),
                ]

                best_cut = max(cut_positions)

                if best_cut >= 0:
                    end = search_start + best_cut

            if end <= start:
                end = min(
                    start + self.chunk_size,
                    text_length,
                )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _build_documents(
        self,
        raw_documents: list[dict[str, Any]],
    ) -> list[Document]:
        """Convert raw documents into chunked LangChain documents.

        Args:
            raw_documents: Raw documents loaded from the processed JSON file.

        Returns:
            LangChain documents ready for embedding and indexing.
        """
        documents: list[Document] = []

        for document_index, item in enumerate(raw_documents):
            if not isinstance(item, dict):
                logger.warning(
                    "Document ignoré à l'index %d : format incorrect.",
                    document_index,
                )
                continue

            page_content = item.get("page_content", "")

            if not isinstance(page_content, str):
                page_content = str(page_content or "")

            page_content = page_content.strip()

            if not page_content:
                logger.warning(
                    "Document vide ignoré à l'index %d.",
                    document_index,
                )
                continue

            raw_metadata = item.get("metadata", {})

            if isinstance(raw_metadata, dict):
                metadata = raw_metadata.copy()
            else:
                logger.warning(
                    "Métadonnées invalides à l'index %d.",
                    document_index,
                )
                metadata = {}

            chunks = self._split_text(page_content)
            chunk_count = len(chunks)

            for chunk_index, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "source_document_index": document_index,
                    }
                )

                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=chunk_metadata,
                    )
                )

            if chunk_count > 1:
                logger.info(
                    "Document %d découpé en %d fragments (%d caractères).",
                    document_index,
                    chunk_count,
                    len(page_content),
                )

        logger.info(
            "Nombre total de fragments à indexer : %d",
            len(documents),
        )

        return documents

    def _create_index(
        self,
        documents: list[Document],
    ) -> FAISS:
        """Create a FAISS vector store from the supplied documents.

        Args:
            documents: Documents to embed and add to the FAISS index.

        Returns:
            The constructed FAISS vector store.

        Raises:
            ValueError: If no documents are available.
            Exception: If embedding generation or index construction fails.
        """
        if not documents:
            raise ValueError("Aucun document disponible pour construire l'index.")

        embeddings = self._get_embeddings()

        logger.info(
            "Génération des embeddings pour %d fragments...",
            len(documents),
        )

        try:
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embeddings,
            )
        except Exception:
            logger.exception(
                "Échec lors de la génération des embeddings "
                "ou de la construction de l'index FAISS."
            )
            raise

        logger.info("Index FAISS construit avec succès.")

        return vector_store

    def _save_index(
        self,
        vector_store: FAISS,
    ) -> None:
        """Save a FAISS index while preserving the previous version.

        Args:
            vector_store: FAISS vector store to persist.

        Raises:
            OSError: If the index or its backup cannot be created, moved,
                replaced, or removed.
        """
        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = Path(
            tempfile.mkdtemp(
                prefix=f"{self.index_path.name}_tmp_",
                dir=self.index_path.parent,
            )
        )

        backup_path = self.index_path.with_name(f"{self.index_path.name}_backup")

        try:
            logger.info(
                "Sauvegarde temporaire : %s",
                temp_path,
            )

            vector_store.save_local(str(temp_path))

            if backup_path.exists():
                shutil.rmtree(backup_path)

            if self.index_path.exists():
                self.index_path.rename(backup_path)

            try:
                temp_path.rename(self.index_path)

            except Exception:
                if backup_path.exists() and not self.index_path.exists():
                    backup_path.rename(self.index_path)

                raise

            if backup_path.exists():
                shutil.rmtree(backup_path)

            logger.info(
                "Index FAISS sauvegardé dans : %s",
                self.index_path,
            )

        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path)

    def rebuild(
        self,
        documents_path: str | Path,
    ) -> int:
        """Rebuild the FAISS index from a processed document snapshot.

        Args:
            documents_path: Path to the JSON file produced by the ingestion
                pipeline.

        Returns:
            Number of document fragments indexed.

        Raises:
            FileNotFoundError: If the document snapshot does not exist.
            OSError: If the Mistral API key is missing or the index cannot
                be persisted.
            ValueError: If the document snapshot is invalid or contains no
                usable documents.
        """
        logger.info("Début de la reconstruction de l'index FAISS.")

        raw_documents = self._load_documents(
            documents_path=documents_path,
        )

        documents = self._build_documents(
            raw_documents=raw_documents,
        )

        vector_store = self._create_index(
            documents=documents,
        )

        self._save_index(
            vector_store=vector_store,
        )

        indexed_count = len(documents)

        logger.info(
            "Reconstruction terminée : %d fragments indexés.",
            indexed_count,
        )

        return indexed_count
