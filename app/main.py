import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from pydantic import BaseModel, Field

from app.core.application.use_cases.ask_question import AskQuestion
from app.infrastructure.llm.event_summarizer import EventSummarizer
from app.infrastructure.rag.rag_question_answerer import (
    RagQuestionAnswerer,
)
from app.ingestion.document_builder import DocumentBuilder
from app.ingestion.event_processor import EventProcessor
from app.ingestion.faiss_indexer import FaissIndexer
from app.ingestion.openagenda_client import OpenAgendaClient
from app.ingestion.pipeline import IngestionPipeline
from app.repositories.event_catalog_repository import EventCatalogRepository
from app.repositories.json_snapshot_repository import (
    JsonSnapshotRepository,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = ROOT_DIR / "data" / "processed" / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
INDEX_METADATA_FILE = INDEX_DIR / "index.pkl"


# -------------------------------------------------------------------
# Dépendances
# -------------------------------------------------------------------

event_summarizer = EventSummarizer(
    model="ministral-3b-latest",
)

document_builder = DocumentBuilder(
    summarizer=event_summarizer,
    use_summarization=True,
)

ingestion_pipeline = IngestionPipeline(
    client=OpenAgendaClient(),
    processor=EventProcessor(),
    document_builder=document_builder,
    snapshot_repository=JsonSnapshotRepository(),
)

faiss_indexer = FaissIndexer()
event_catalog_repository = EventCatalogRepository(ROOT_DIR)


def _load_vector_store() -> FAISS:
    """Load the persisted FAISS vector store.

    Returns:
        The loaded FAISS vector store.

    Raises:
        RuntimeError: If the Mistral API key is not configured.
    """
    load_dotenv(ROOT_DIR / ".env")

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError("La variable d'environnement MISTRAL_API_KEY est absente.")

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key,
    )

    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vector_store() -> FAISS:
    """Return the currently loaded FAISS vector store.

    Returns:
        The vector store currently published in the application state.

    Raises:
        RuntimeError: If no vector store is currently available.
    """
    vector_store = getattr(
        app.state,
        "vector_store",
        None,
    )

    if vector_store is None:
        raise RuntimeError(
            "Le vector store FAISS n'est pas disponible. "
            "Veuillez appeler POST /rebuild."
        )

    return vector_store


def get_event_catalog() -> list[dict[str, Any]]:
    """Return the currently loaded structured event catalog.

    Returns:
        The catalog currently published in the application state.

    Raises:
        RuntimeError: If no event catalog is currently available.
    """
    catalog = getattr(
        app.state,
        "event_catalog",
        None,
    )

    if not catalog:
        raise RuntimeError(
            "Le catalogue des événements n'est pas disponible. "
            "Veuillez appeler POST /rebuild."
        )

    return catalog


ask_question_use_case = AskQuestion(
    answerer=RagQuestionAnswerer(
        vector_store_provider=get_vector_store,
        catalog_provider=get_event_catalog,
    ),
)


# -------------------------------------------------------------------
# Cycle de vie de l'application
# -------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load persisted application data when the API starts.

    Raises:
        Exception: If an existing index or its catalog cannot be loaded.
    """
    app.state.vector_store = None
    app.state.event_catalog = []

    if not _index_exists():
        logger.warning(
            "Index FAISS absent. "
            "L'API démarre sans index. "
            "Utilisez POST /rebuild pour le reconstruire."
        )
    else:
        try:
            app.state.vector_store = _load_vector_store()

            event_catalog_repository.load_latest()

            app.state.event_catalog = event_catalog_repository.all()

            logger.info(
                "Index FAISS et catalogue chargés (%d événements).",
                len(app.state.event_catalog),
            )

        except Exception:
            logger.exception("L'index FAISS existe, mais son chargement a échoué.")
            raise

    yield

    logger.info("Arrêt de l'API Pau Culture.")


def _index_exists() -> bool:
    """Return whether all required FAISS index files are present.

    Returns:
        True when the FAISS index and its metadata file both exist.
    """
    return INDEX_FILE.is_file() and INDEX_METADATA_FILE.is_file()


# -------------------------------------------------------------------
# Application FastAPI
# -------------------------------------------------------------------

app = FastAPI(
    title="Pau Culture API",
    description="API du chatbot culturel de Pau",
    version="0.1.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# Schémas de requêtes et réponses
# -------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request payload for the chatbot."""

    question: str = Field(
        min_length=1,
        description="Question posée au chatbot",
    )


class ChatResponse(BaseModel):
    """Response payload returned by the chatbot."""

    question: str
    answer: str


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------


@app.get("/health")
def health_check() -> dict[str, bool | str]:
    """Return the current API health status.

    Returns:
        Health status and FAISS availability.
    """
    vector_store_available = (
        getattr(
            app.state,
            "vector_store",
            None,
        )
        is not None
    )

    return {
        "status": "ok",
        "vector_store_available": vector_store_available,
    }


@app.post("/ask", response_model=ChatResponse)
@app.post(
    "/chat",
    response_model=ChatResponse,
    include_in_schema=False,
)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using the Pau Culture chatbot."""
    try:
        result = ask_question_use_case.execute(
            request.question,
        )

        return ChatResponse(
            question=result.question,
            answer=result.answer,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Échec du traitement du chat",
        )

        raise HTTPException(
            status_code=500,
            detail="Une erreur est survenue lors du traitement.",
        ) from error


@app.post("/rebuild")
def rebuild_index() -> dict[str, Any]:
    """Rebuild the event catalog and FAISS index.

    Returns:
        A summary of the ingestion and indexing operations.

    Raises:
        HTTPException: If any rebuild step fails.
    """
    try:
        result = ingestion_pipeline.run()

        processed_path = _resolve_path(
            result["processed_path"],
        )

        if not processed_path.exists():
            raise FileNotFoundError(
                f"Fichier de données traitées introuvable : {processed_path}"
            )

        indexed_count = faiss_indexer.rebuild(
            documents_path=processed_path,
        )

        new_vector_store = _load_vector_store()

        event_catalog_repository.load(
            processed_path,
        )

        new_catalog = event_catalog_repository.all()

        app.state.vector_store = new_vector_store
        app.state.event_catalog = new_catalog

        logger.info(
            "Rebuild terminé. Documents indexés : %s",
            indexed_count,
        )

        return {
            "status": "ok",
            "message": "Ingestion et index FAISS terminés.",
            "ingestion": result,
            "indexed_count": indexed_count,
        }

    except Exception as error:
        logger.exception(
            "Échec de l'endpoint /rebuild",
        )

        raise HTTPException(
            status_code=500,
            detail=(f"Échec de /rebuild : " f"{type(error).__name__}: {error}"),
        ) from error


def _resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when necessary.

    Args:
        path: Absolute or project-relative path.

    Returns:
        An absolute path.
    """
    resolved_path = Path(path)

    if not resolved_path.is_absolute():
        resolved_path = ROOT_DIR / resolved_path

    return resolved_path
