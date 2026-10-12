"""Évalue la recherche RAG sur un jeu de questions annotées local."""

import json
import os
import sys
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT_DIR / "tests" / "fixtures" / "2026_10_12_documents.json"
EVALUATION_PATH = ROOT_DIR / "tests" / "fixtures" / "retrieval_eval.json"
INDEX_PATH = ROOT_DIR / "data" / "processed" / "faiss_test_index"
JSON_REPORT_PATH = ROOT_DIR / "data" / "processed" / "retrieval_evaluation.json"
MARKDOWN_REPORT_PATH = ROOT_DIR / "data" / "processed" / "retrieval_evaluation.md"
EMBEDDING_MODEL = "mistral-embed"

# Permet l'exécution directe : uv run python scripts/evaluate_retrieval.py
sys.path.insert(0, str(ROOT_DIR))


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Précision standard@k : pertinents dans le top-k / k (dénominateur fixe)."""
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")
    return len(set(retrieved_ids[:k]) & relevant_ids) / k


def precision_on_returned(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Précision parmi les résultats effectivement retournés, jusqu'à k."""
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")
    returned = retrieved_ids[:k]
    if not returned:
        return 0.0
    return len(set(returned) & relevant_ids) / len(returned)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Rappel@k : pertinents retrouvés / pertinents attendus."""
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")
    if not relevant_ids:
        return 1.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Rang réciproque du premier résultat pertinent (la liste est déjà limitée à k)."""
    for rank, uid in enumerate(retrieved_ids, start=1):
        if uid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Le fichier doit contenir un objet JSON : {path}")
    return payload


def _average(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    """Écrit un rapport lisible, avec les définitions et les cas à examiner."""
    k = summary["k"]
    lines = [
        "# Rapport d'évaluation de la recherche RAG",
        "",
        f"- Date de référence : **{summary['reference_date']}**",
        f"- Questions évaluées : **{summary['question_count']}**",
        f"- Taille du classement : **{k}**",
        f"- Index : `{summary['index_path']}`",
        "",
        "## Résumé des métriques",
        "",
        "| Métrique | Résultat | Interprétation |",
        "|---|---:|---|",
        f"| Précision@{k} | {_pct(summary['mean_precision_at_k'])} | Pertinents dans le top-k, divisé par k, même si moins de k résultats sont retournés |",  # noqa: E501
        f"| Précision sur résultats retournés | {_pct(summary['mean_precision_on_returned'])} | Pertinents divisés par le nombre effectivement retourné, jusqu'à k |",  # noqa: E501
        f"| Rappel@{k} | {_pct(summary['mean_recall_at_k'])} | Part des identifiants attendus retrouvés dans le top-k |",  # noqa: E501
        f"| Rappel conditionnel après filtres@{k} | {_pct(summary['mean_conditional_recall_at_k'])} | Rappel parmi les événements attendus qui ont survécu aux filtres |",  # noqa: E501
        f"| MRR@{k} | {_pct(summary['mrr_at_k'])} | Rang du premier résultat pertinent ; plus élevé est meilleur |",  # noqa: E501
        f"| Questions avec au moins un résultat pertinent | {summary['questions_with_hit']}/{summary['question_count']} | Au moins un identifiant attendu dans le top-k |",  # noqa: E501
        f"| Questions sans résultat | {summary['questions_with_zero_results']}/{summary['question_count']} | Aucun événement retourné après filtrage/recherche |",  # noqa: E501
        f"| Couverture moyenne des attendus par les filtres | {_pct(summary['mean_filter_coverage'])} | Part des attendus présents dans le corpus qui restent éligibles après les filtres |",  # noqa: E501
        f"| Identifiants attendus absents du corpus | {summary['expected_ids_missing_from_corpus_count']} | Problème de fixture ou d'annotation, pas de classement |",  # noqa: E501
        f"| Identifiants attendus exclus par les filtres | {summary['expected_ids_excluded_by_filters_count']} | Cas à vérifier dans le parseur ou les filtres métier |",  # noqa: E501
        "",
        "## Résultats par question",
        "",
        f"| Question | Candidats | Retournés | Pertinents retrouvés | P@{k} | Précision retournés | R@{k} | MRR@{k} |",  # noqa: E501
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        hits = ", ".join(row["relevant_found_ids"]) or "—"
        lines.append(
            f"| `{row['id']}` — {row['question']} | {row['eligible_count']} | "
            f"{row['returned_count']} | {hits} | {_pct(row['precision_at_k'])} | "
            f"{_pct(row['precision_on_returned'])} | {_pct(row['recall_at_k'])} | "
            f"{_pct(row['reciprocal_rank_at_k'])} |"
        )
    lines.extend(["", "## Diagnostics", ""])
    diagnostic_count = 0
    for row in summary["results"]:
        problems = []
        if row["expected_ids_missing_from_corpus"]:
            problems.append(
                "attendus absents du corpus : "
                + ", ".join(row["expected_ids_missing_from_corpus"])
            )
        if row["expected_ids_excluded_by_filters"]:
            problems.append(
                "attendus exclus par les filtres : "
                + ", ".join(row["expected_ids_excluded_by_filters"])
            )
        if row["missing_expected_ids_after_retrieval"]:
            problems.append(
                "attendus non retrouvés : "
                + ", ".join(row["missing_expected_ids_after_retrieval"])
            )
        if row["retrieved_ids_not_annotated_relevant"]:
            problems.append(
                "résultats non annotés pertinents : "
                + ", ".join(row["retrieved_ids_not_annotated_relevant"])
            )
        if row["returned_count"] == 0:
            problems.append("aucun résultat retourné")
        if problems:
            diagnostic_count += 1
            lines.append(f"### {row['id']}")
            lines.append("")
            lines.append(f"Question : {row['question']}")
            lines.extend(f"- {problem}" for problem in problems)
            lines.append("")
    if diagnostic_count == 0:
        lines.append(
            "Aucun cas à signaler selon les contrôles automatiques. Cela ne remplace pas une revue humaine de la pertinence."  # noqa: E501
        )
        lines.append("")
    lines.extend(
        [
            "## Définitions et limites",
            "",
            f"- **Précision@{k}** utilise un dénominateur fixe égal à k. Elle baisse donc si le système retourne moins de k résultats, même si tous sont pertinents.",  # noqa: E501
            "- **Précision sur résultats retournés** permet de lire séparément la pertinence des résultats effectivement fournis. Une requête sans résultat vaut 0 % et est également comptée séparément.",  # noqa: E501
            "- **Rappel global@k** rapporte les résultats retrouvés à tous les identifiants annotés attendus. Le rappel conditionnel mesure la recherche uniquement parmi les attendus qui ont survécu aux filtres.",  # noqa: E501
            "- Les résultats non présents dans `expected_ids` sont décrits comme **non annotés pertinents** ; ils ne sont pas automatiquement des faux positifs tant que l'exhaustivité des annotations n'a pas été vérifiée.",  # noqa: E501
            "- Les scores FAISS sont des distances, pas des probabilités. Pour une distance L2, une valeur plus faible indique généralement un résultat plus proche.",  # noqa: E501
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate() -> int:
    """Exécute l'évaluation sur l'index FAISS de test isolé."""
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")

    if not INDEX_PATH.exists():
        print(
            f"Index de test introuvable : {INDEX_PATH}\n"
            "Construis-le d'abord avec scripts/rebuild_test_index.py.",
            file=sys.stderr,
        )
        return 2

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print(
            f"MISTRAL_API_KEY introuvable. Vérifie {ROOT_DIR / '.env'}.",
            file=sys.stderr,
        )
        return 2

    documents_payload = _load_json(FIXTURE_PATH)
    evaluation_payload = _load_json(EVALUATION_PATH)
    documents = documents_payload.get("documents", [])
    questions = evaluation_payload.get("questions", [])
    reference_date = date.fromisoformat(evaluation_payload["reference_date"])
    k = int(evaluation_payload.get("k", 5))
    if k < 1:
        raise ValueError(
            "La valeur k du jeu d'évaluation doit être supérieure ou égale à 1."
        )
    if not questions:
        print("Aucune question à évaluer.", file=sys.stderr)
        return 2

    from langchain_community.vectorstores import FAISS
    from langchain_mistralai import MistralAIEmbeddings

    from app.infrastructure.rag.event_filter import EventFilter
    from app.infrastructure.rag.event_retriever import EventRetriever
    from app.infrastructure.rag.query_parser import QueryParser

    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL, mistral_api_key=api_key)
    vector_store = FAISS.load_local(
        str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )

    parser = QueryParser()
    event_filter = EventFilter()
    retriever = EventRetriever()
    corpus_ids = {
        str((document.get("metadata") or {}).get("uid", "")) for document in documents
    }
    rows: list[dict[str, Any]] = []

    print(
        f"Évaluation de {len(questions)} questions — date de référence : {reference_date}"  # noqa: E501
    )
    print(f"Index : {INDEX_PATH}")
    print(f"Mesures sur le top-{k} ; rapport JSON et Markdown généré.\n")

    for item in questions:
        question = item["question"]
        expected_ids = {str(uid) for uid in item["expected_ids"]}
        criteria = parser.parse(question, today=reference_date)
        eligible = event_filter.filter(documents, criteria, today=reference_date)
        if not eligible:
            retrieved: list[tuple[Any, float]] = []
        elif criteria.semantic_query.strip():
            retrieved = retriever.retrieve(
                question=criteria.semantic_query,
                vector_store=vector_store,
                eligible_documents=eligible,
                limit=k,
            )
        else:
            retrieved = retriever.from_documents(eligible, limit=k)

        retrieved_ids = [str(doc.metadata.get("uid", "")) for doc, _ in retrieved]
        distances = [float(score) for _, score in retrieved]
        eligible_ids = {
            str((doc.get("metadata") or {}).get("uid", "")) for doc in eligible
        }
        expected_in_corpus = expected_ids & corpus_ids
        eligible_expected = expected_ids & eligible_ids
        missing_from_corpus = expected_ids - corpus_ids
        excluded_by_filters = expected_in_corpus - eligible_ids
        relevant_found = set(retrieved_ids) & expected_ids
        missing_after_retrieval = expected_ids - set(retrieved_ids)
        not_annotated = [uid for uid in retrieved_ids if uid not in expected_ids]
        p_at_k = precision_at_k(retrieved_ids, expected_ids, k)
        p_returned = precision_on_returned(retrieved_ids, expected_ids, k)
        r_at_k = recall_at_k(retrieved_ids, expected_ids, k)
        conditional_recall = (
            len(set(retrieved_ids[:k]) & eligible_expected) / len(eligible_expected)
            if eligible_expected
            else None
        )
        rr = reciprocal_rank(retrieved_ids[:k], expected_ids)
        filter_coverage = (
            len(eligible_expected) / len(expected_in_corpus)
            if expected_in_corpus
            else None
        )
        row = {
            "id": item["id"],
            "question": question,
            "expected_ids": sorted(expected_ids),
            "expected_ids_missing_from_corpus": sorted(missing_from_corpus),
            "eligible_count": len(eligible),
            "eligible_expected_ids": sorted(eligible_expected),
            "expected_ids_excluded_by_filters": sorted(excluded_by_filters),
            "filter_coverage": filter_coverage,
            "retrieved_ids": retrieved_ids,
            "retrieved_distances": distances,
            "retrieved_details": [
                {
                    "rank": rank,
                    "uid": uid,
                    "distance": distances[rank - 1],
                    "annotated_relevant": uid in expected_ids,
                }
                for rank, uid in enumerate(retrieved_ids, start=1)
            ],
            "returned_count": len(retrieved_ids),
            "relevant_found_ids": sorted(relevant_found),
            "missing_expected_ids_after_retrieval": sorted(missing_after_retrieval),
            "retrieved_ids_not_annotated_relevant": not_annotated,
            "hit_count": len(relevant_found),
            "precision_at_k": p_at_k,
            "precision_on_returned": p_returned,
            "recall_at_k": r_at_k,
            "conditional_recall_at_k": conditional_recall,
            "reciprocal_rank_at_k": rr,
        }
        rows.append(row)
        print(
            f"{item['id']:<30} P@{k}={p_at_k:.2f}  P(ret)={p_returned:.2f}  "
            f"R@{k}={r_at_k:.2f}  MRR={rr:.2f}  "
            f"hits={len(relevant_found)}/{len(expected_ids)}  "
            f"candidats={len(eligible)}  résultats={','.join(retrieved_ids) or 'aucun'}"
        )
        if missing_from_corpus:
            print(
                "  ERREUR fixture : attendus absents du corpus : "
                + ", ".join(sorted(missing_from_corpus))
            )
        if excluded_by_filters:
            print(
                "  ATTENTION filtres : attendus exclus : "
                + ", ".join(sorted(excluded_by_filters))
            )

    count = len(rows)
    summary: dict[str, Any] = {
        "question_count": count,
        "k": k,
        "reference_date": reference_date.isoformat(),
        "index_path": str(INDEX_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "mean_precision_at_k": mean(row["precision_at_k"] for row in rows),
        "mean_precision_on_returned": mean(
            row["precision_on_returned"] for row in rows
        ),
        "mean_recall_at_k": mean(row["recall_at_k"] for row in rows),
        "mean_conditional_recall_at_k": _average(
            [row["conditional_recall_at_k"] for row in rows]
        ),
        "mrr_at_k": mean(row["reciprocal_rank_at_k"] for row in rows),
        "questions_with_hit": sum(row["hit_count"] > 0 for row in rows),
        "questions_with_zero_results": sum(row["returned_count"] == 0 for row in rows),
        "mean_candidate_count": mean(row["eligible_count"] for row in rows),
        "mean_filter_coverage": _average([row["filter_coverage"] for row in rows]),
        "expected_ids_missing_from_corpus_count": sum(
            len(row["expected_ids_missing_from_corpus"]) for row in rows
        ),
        "expected_ids_excluded_by_filters_count": sum(
            len(row["expected_ids_excluded_by_filters"]) for row in rows
        ),
        "results": rows,
    }
    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown_report(summary, MARKDOWN_REPORT_PATH)

    print("\n--- Résumé ---")
    print(
        f"Précision moyenne@{k}                  : {summary['mean_precision_at_k']:.3f}"
    )
    print(
        f"Précision moyenne sur résultats rendus : {summary['mean_precision_on_returned']:.3f}"  # noqa: E501
    )
    print(f"Rappel moyen@{k}                       : {summary['mean_recall_at_k']:.3f}")
    print(
        f"Rappel conditionnel moyen@{k}          : {_pct(summary['mean_conditional_recall_at_k'])}"  # noqa: E501
    )
    print(f"MRR@{k}                                : {summary['mrr_at_k']:.3f}")
    print(
        f"Questions avec au moins un hit         : {summary['questions_with_hit']}/{count}"  # noqa: E501
    )
    print(
        f"Questions sans résultat                : {summary['questions_with_zero_results']}/{count}"  # noqa: E501
    )
    print(
        f"Attendues absentes du corpus           : {summary['expected_ids_missing_from_corpus_count']}"  # noqa: E501
    )
    print(
        f"Attendues exclues par les filtres      : {summary['expected_ids_excluded_by_filters_count']}"  # noqa: E501
    )
    print(f"Rapport JSON      : {JSON_REPORT_PATH}")
    print(f"Rapport Markdown  : {MARKDOWN_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
