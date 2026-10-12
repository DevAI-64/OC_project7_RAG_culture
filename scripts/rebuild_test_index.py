"""Rebuild the isolated test FAISS index from a prepared documents snapshot."""

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.ingestion.faiss_indexer import FaissIndexer  # noqa: E402

DEFAULT_DOCUMENTS = ROOT_DIR / "tests" / "fixtures" / "2026_10_12_documents.json"
DEFAULT_INDEX = ROOT_DIR / "data" / "processed" / "faiss_test_index"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the test index rebuild."""
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruit l'index FAISS de test depuis un snapshot de documents "
            "déjà préparés, sans appeler OpenAgenda ni synthétiser les descriptions."
        )
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS,
        help=f"Snapshot documents JSON (défaut : {DEFAULT_DOCUMENTS})",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"Répertoire de l'index de test (défaut : {DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche les logs détaillés.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the test index from the provided documents snapshot."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    documents_path = args.documents.resolve()
    index_path = args.index_path.resolve()

    if not documents_path.is_file():
        raise FileNotFoundError(f"Snapshot de documents introuvable : {documents_path}")

    # Intentionally call the indexer directly: do not run IngestionPipeline,
    # EventProcessor, DocumentBuilder, or EventSummarizer here.
    indexer = FaissIndexer(index_path=index_path)
    indexed_count = indexer.rebuild(documents_path)

    print(f"Index de test reconstruit : {index_path}")
    print(f"Fragments indexés : {indexed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
