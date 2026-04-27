from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_rag.metrics import (  # noqa: E402
    NO_ANSWER_SIMILARITY_THRESHOLD,
    keyword_recall,
    page_hit,
    reciprocal_rank,
    similarity_stats,
)
from app.agent_rag.run_business_pdf_eval import build_pdf_vector_store  # noqa: E402
from eval.dataset import eval_dataset  # noqa: E402


def retrieved_pdf_pages(docs: list[Any]) -> list[int]:
    pages: list[int] = []
    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            pages.append(page + 1)
    return pages


def run_eval(*, k: int = 5) -> list[dict[str, Any]]:
    load_dotenv(ROOT / ".env")
    vector_store, source_filter = build_pdf_vector_store()
    rows: list[dict[str, Any]] = []

    for item in eval_dataset:
        docs_with_scores = vector_store.similarity_search_with_score(
            item["question"],
            k=k,
            filter=source_filter,
        )
        docs = [doc for doc, _score in docs_with_scores]
        scores = [float(score) for _doc, score in docs_with_scores]
        contexts = [doc.page_content for doc in docs]
        context_text = " ".join(contexts)
        pages = retrieved_pdf_pages(docs)
        stats = similarity_stats(scores)

        rows.append(
            {
                "question": item["question"],
                "negative_question": item["negative_question"],
                "expected_page": item["expected_page"],
                "retrieved_pages": pages,
                "page_hit": page_hit(pages, item["expected_page"]),
                "hit_at_k": float(item["expected_page"] in pages)
                if item["expected_page"] is not None
                else None,
                "mrr": reciprocal_rank(item["expected_page"], pages),
                "keyword_recall": keyword_recall(context_text, item["expected_keywords"]),
                "max_similarity_at_k": stats["max_similarity_at_k"],
                "avg_similarity_at_k": stats["avg_similarity_at_k"],
                "similarity_gap": stats["similarity_gap"],
                "negative_low_similarity": (
                    stats["max_similarity_at_k"] < NO_ANSWER_SIMILARITY_THRESHOLD
                    if item["negative_question"]
                    else None
                ),
            }
        )

    return rows


def main() -> None:
    k = int(os.getenv("RAG_EVAL_TOP_K") or "5")
    rows = run_eval(k=k)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
