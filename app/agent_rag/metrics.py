from __future__ import annotations

import re
from collections import Counter
from typing import Any


QUESTION_STOPWORDS = {
    "как",
    "по",
    "китайски",
    "сказать",
    "или",
    "это",
    "по-китайски",
}
NO_ANSWER_SIMILARITY_THRESHOLD = 0.35


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def contains_keyword(text: str, keyword: str) -> bool:
    return compact_text(keyword.casefold()) in compact_text(text.casefold())


def extract_chinese_terms(text: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]+", text)
    return list(dict.fromkeys(terms))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[^\W_]+", text.lower(), flags=re.UNICODE)


def keyword_recall(context_text: str, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None

    hits = [keyword for keyword in expected_keywords if contains_keyword(context_text, keyword)]
    return len(hits) / len(expected_keywords)


def keyword_coverage(question: str, contexts: list[str]) -> float:
    question_tokens = {
        token
        for token in tokenize(question)
        if token not in QUESTION_STOPWORDS and len(token) > 1
    }
    if not question_tokens:
        return 0.0

    context_tokens = set(tokenize(" ".join(contexts)))
    return len(question_tokens & context_tokens) / len(question_tokens)


def page_hit(retrieved_pages: list[int], expected_page: int | None) -> bool | None:
    if expected_page is None:
        return None
    return expected_page in retrieved_pages


def reciprocal_rank(expected_page: int | None, retrieved_pages: list[int]) -> float | None:
    if expected_page is None:
        return None
    try:
        return 1.0 / (retrieved_pages.index(expected_page) + 1)
    except ValueError:
        return 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    true_tokens = tokenize(ground_truth)
    if not pred_tokens or not true_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    return 2 * precision * recall / (precision + recall)


def similarity_stats(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {
            "max_similarity_at_k": 0.0,
            "avg_similarity_at_k": 0.0,
            "similarity_gap": 0.0,
        }

    sorted_scores = sorted(scores, reverse=True)
    return {
        "max_similarity_at_k": float(sorted_scores[0]),
        "avg_similarity_at_k": float(sum(scores) / len(scores)),
        "similarity_gap": float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0,
    }


def rag_confidence_score(
    *,
    max_context_similarity: float,
    keyword_coverage_score: float,
    answer_f1_score: float,
) -> float:
    similarity = max(0.0, min(1.0, max_context_similarity))
    return 0.5 * similarity + 0.2 * keyword_coverage_score + 0.3 * answer_f1_score


def answer_refuses(answer: str) -> bool:
    normalized = answer.casefold()
    refusal_markers = [
        "в контексте нет",
        "в документе нет",
        "нет информации",
        "не найден",
        "не могу ответить",
    ]
    return any(marker in normalized for marker in refusal_markers)


def average_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), int | float)]
    return sum(values) / len(values) if values else None
