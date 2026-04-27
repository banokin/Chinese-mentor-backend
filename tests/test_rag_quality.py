from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agent_rag.metrics import keyword_recall, page_hit
from app.agent_rag.run_business_pdf_eval import build_pdf_vector_store
from eval.dataset import eval_dataset
from eval.run_eval import retrieved_pdf_pages


load_dotenv(BACKEND_ROOT / ".env")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("QDRANT_URL"),
    reason="RAG quality test requires OPENAI_API_KEY and QDRANT_URL",
)
def test_retriever_quality() -> None:
    vector_store, source_filter = build_pdf_vector_store()
    passed = 0
    total = 0

    for item in eval_dataset:
        if item["negative_question"]:
            continue

        docs = vector_store.similarity_search(
            item["question"],
            k=5,
            filter=source_filter,
        )
        contexts = [doc.page_content for doc in docs]
        context_text = " ".join(contexts)
        pages = retrieved_pdf_pages(docs)

        total += 1
        if page_hit(pages, item["expected_page"]):
            passed += 1

        assert keyword_recall(context_text, item["expected_keywords"]) >= 0.6

    hit_rate = passed / total
    assert hit_rate >= 0.8
