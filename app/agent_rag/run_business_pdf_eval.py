from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.agent_rag.business_pdf_eval_dataset import SOURCE_FILE, TEST_QUESTIONS


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


def build_pdf_vector_store():
    collection = (os.getenv("QDRANT_COLLECTION") or "chinese_lexicon").strip()
    qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise EnvironmentError("QDRANT_URL is not set")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    embeddings = OpenAIEmbeddings(
        model=(os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    )
    client = QdrantClient(url=qdrant_url, prefer_grpc=False, check_compatibility=False)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )
    source_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.source_file",
                match=MatchValue(value=SOURCE_FILE),
            )
        ]
    )
    return vector_store, source_filter


def build_pdf_retriever(*, top_k: int):
    vector_store, source_filter = build_pdf_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": top_k, "filter": source_filter})


def build_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    return f"""
Ответь на вопрос только на основе контекста.
Если в контексте нет ответа, скажи: "В контексте нет ответа".
Отвечай кратко: русский перевод, китайское слово и pinyin, если он есть в контексте.

Контекст:
{context_text}

Вопрос:
{question}
""".strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_chinese_terms(text: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]+", text)
    return list(dict.fromkeys(terms))


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def contains_term(text: str, term: str) -> bool:
    return compact_text(term) in compact_text(text)


def contains_keyword(text: str, keyword: str) -> bool:
    return compact_text(keyword.casefold()) in compact_text(text.casefold())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[^\W_]+", text.lower(), flags=re.UNICODE)


def keyword_coverage(question: str, contexts: list[str]) -> float:
    question_tokens = {
        token
        for token in tokenize(question)
        if token not in QUESTION_STOPWORDS and len(token) > 1
    }
    if not question_tokens:
        return 0.0

    context_tokens = set(tokenize(" ".join(contexts)))
    matched = question_tokens & context_tokens
    return len(matched) / len(question_tokens)


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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def rag_confidence_score(
    *,
    max_context_similarity: float,
    keyword_coverage_score: float,
    answer_f1_score: float,
) -> float:
    return (
        0.5 * clamp01(max_context_similarity)
        + 0.2 * keyword_coverage_score
        + 0.3 * answer_f1_score
    )


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


def reciprocal_rank(expected_page: int | None, retrieved_pages: list[int]) -> float | None:
    if expected_page is None:
        return None
    try:
        return 1.0 / (retrieved_pages.index(expected_page) + 1)
    except ValueError:
        return 0.0


def score_cheap_row(row: dict[str, Any]) -> dict[str, Any]:
    expected_keywords = row.get("expected_keywords") or []
    expected_terms = expected_keywords or extract_chinese_terms(row["ground_truth"])
    contexts = row["contexts"]
    answer = row["answer"]
    joined_context = "\n".join(contexts)
    retrieval_scores = [float(score) for score in row.get("retrieval_scores", [])]
    similarity = similarity_stats(retrieval_scores)
    retrieved_pages = row.get("retrieved_pages") or []
    expected_page = row.get("expected_page")
    negative_question = bool(row.get("negative_question", False))

    if not expected_terms:
        context_hit_any = 0.0
        context_hit_all = 0.0
        answer_hit_any = 0.0
        answer_hit_all = 0.0
    else:
        context_hits = [contains_term(joined_context, term) for term in expected_terms]
        answer_hits = [contains_term(answer, term) for term in expected_terms]
        context_hit_any = float(any(context_hits))
        context_hit_all = float(all(context_hits))
        answer_hit_any = float(any(answer_hits))
        answer_hit_all = float(all(answer_hits))

    context_precision_proxy = (
        sum(any(contains_term(context, term) for term in expected_terms) for context in contexts)
        / len(contexts)
        if contexts and expected_terms
        else 0.0
    )
    answer_terms = extract_chinese_terms(answer)
    answer_context_support = (
        sum(contains_term(joined_context, term) for term in answer_terms) / len(answer_terms)
        if answer_terms
        else 0.0
    )
    keyword_hits = [
        keyword for keyword in expected_keywords if contains_keyword(joined_context, keyword)
    ]
    keyword_recall = (
        len(keyword_hits) / len(expected_keywords) if expected_keywords else None
    )
    page_hit = expected_page in retrieved_pages if expected_page is not None else None
    hit_at_k = float(page_hit) if page_hit is not None else None
    mrr = reciprocal_rank(expected_page, retrieved_pages)
    negative_low_similarity = (
        similarity["max_similarity_at_k"] < NO_ANSWER_SIMILARITY_THRESHOLD
        if negative_question
        else None
    )
    negative_answer_refusal = answer_refuses(answer) if negative_question else None
    keyword_coverage_score = keyword_coverage(row["question"], contexts)
    answer_f1_score = token_f1(answer, row["ground_truth"])
    confidence = rag_confidence_score(
        max_context_similarity=similarity["max_similarity_at_k"],
        keyword_coverage_score=keyword_coverage_score,
        answer_f1_score=answer_f1_score,
    )

    return {
        "question": row["question"],
        "expected_terms": expected_terms,
        "expected_keywords": expected_keywords,
        "expected_page": expected_page,
        "retrieved_pages": retrieved_pages,
        "page_hit": page_hit,
        "hit_at_k": hit_at_k,
        "mrr": mrr,
        "negative_question": negative_question,
        "negative_low_similarity": negative_low_similarity,
        "negative_answer_refusal": negative_answer_refusal,
        "retrieval_scores": retrieval_scores,
        **similarity,
        "keyword_coverage": keyword_coverage_score,
        "keyword_hits": keyword_hits,
        "keyword_recall": keyword_recall,
        "context_hit_any": context_hit_any,
        "context_hit_all": context_hit_all,
        "context_precision_proxy": context_precision_proxy,
        "answer_hit_any": answer_hit_any,
        "answer_hit_all": answer_hit_all,
        "answer_token_f1": answer_f1_score,
        "answer_context_support": answer_context_support,
        "rag_confidence_score": confidence,
    }


def run_cheap_evaluation(rows: list[dict[str, Any]], output_path: Path) -> None:
    scores = [score_cheap_row(row) for row in rows]
    if not scores:
        return

    summary = {
        "rows": len(scores),
        "max_similarity_at_k": sum(row["max_similarity_at_k"] for row in scores) / len(scores),
        "avg_similarity_at_k": sum(row["avg_similarity_at_k"] for row in scores) / len(scores),
        "similarity_gap": sum(row["similarity_gap"] for row in scores) / len(scores),
        "keyword_coverage": sum(row["keyword_coverage"] for row in scores) / len(scores),
        "page_hit": average_metric(scores, "page_hit"),
        "hit_at_k": average_metric(scores, "hit_at_k"),
        "mrr": average_metric(scores, "mrr"),
        "keyword_recall": average_metric(scores, "keyword_recall"),
        "negative_low_similarity": average_metric(scores, "negative_low_similarity"),
        "negative_answer_refusal": average_metric(scores, "negative_answer_refusal"),
        "context_hit_any": sum(row["context_hit_any"] for row in scores) / len(scores),
        "context_hit_all": sum(row["context_hit_all"] for row in scores) / len(scores),
        "context_precision_proxy": sum(row["context_precision_proxy"] for row in scores) / len(scores),
        "answer_hit_any": sum(row["answer_hit_any"] for row in scores) / len(scores),
        "answer_hit_all": sum(row["answer_hit_all"] for row in scores) / len(scores),
        "answer_token_f1": sum(row["answer_token_f1"] for row in scores) / len(scores),
        "answer_context_support": sum(row["answer_context_support"] for row in scores) / len(scores),
        "rag_confidence_score": sum(row["rag_confidence_score"] for row in scores) / len(scores),
    }

    json_path = output_path.with_name(f"{output_path.stem}_cheap_metrics.json")
    csv_path = output_path.with_name(f"{output_path.stem}_cheap_metrics.csv")
    payload = {"summary": summary, "rows": scores}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0].keys()))
        writer.writeheader()
        writer.writerows(scores)

    print("\nCheap metrics result:")
    for metric, value in summary.items():
        print(f"{metric}: {value:.3f}" if isinstance(value, float) else f"{metric}: {value}")
    print(f"Saved cheap metrics CSV:  {csv_path}")
    print(f"Saved cheap metrics JSON: {json_path}")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return json_safe(value.dict())
    return repr(value)


def run_ragas_evaluation(rows: list[dict[str, Any]], output_path: Path) -> None:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ragas_llm = ChatOpenAI(
        model=(os.getenv("RAGAS_LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip(),
        temperature=0,
    )
    ragas_embeddings = OpenAIEmbeddings(
        model=(os.getenv("RAGAS_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    )
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = result.to_pandas()
    csv_path = output_path.with_name(f"{output_path.stem}_ragas.csv")
    json_path = output_path.with_name(f"{output_path.stem}_ragas.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print("\nRAGAS result:")
    print(result)
    print(f"Saved RAGAS CSV:  {csv_path}")
    print(f"Saved RAGAS JSON: {json_path}")


def run_deepeval_evaluation(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    threshold: float,
) -> None:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge_model = (
        os.getenv("DEEPEVAL_LLM_MODEL")
        or os.getenv("RAGAS_LLM_MODEL")
        or os.getenv("OPENAI_CHAT_MODEL")
        or "gpt-4o-mini"
    ).strip()
    test_cases = [
        LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            retrieval_context=row["contexts"],
            expected_output=row["ground_truth"],
        )
        for row in rows
    ]
    metrics = [
        AnswerRelevancyMetric(threshold=threshold, model=judge_model),
        FaithfulnessMetric(threshold=threshold, model=judge_model),
        ContextualPrecisionMetric(threshold=threshold, model=judge_model),
    ]

    result = deepeval_evaluate(test_cases=test_cases, metrics=metrics)
    result_path = output_path.with_name(f"{output_path.stem}_deepeval.json")
    result_path.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDeepEval result:")
    print(result)
    print(f"Saved DeepEval JSON: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG eval data for business Chinese PDF.")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions for a quick smoke run.")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_TOP_K") or "4"))
    parser.add_argument(
        "--skip-cheap-metrics",
        action="store_true",
        help="Do not run deterministic no-LLM metrics.",
    )
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS metrics after building answers.")
    parser.add_argument(
        "--deepeval",
        action="store_true",
        help="Run DeepEval judge metrics after building answers.",
    )
    parser.add_argument(
        "--deepeval-threshold",
        type=float,
        default=0.7,
        help="Pass threshold for DeepEval metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_outputs/business_pdf_eval.jsonl"),
        help="Output JSONL path relative to backend/ by default.",
    )
    args = parser.parse_args()

    backend_root = Path(__file__).resolve().parents[2]
    load_dotenv(backend_root / ".env")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = backend_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vector_store, source_filter = build_pdf_vector_store()
    top_k = max(1, args.top_k)
    llm = ChatOpenAI(
        model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip(),
        temperature=0,
    )

    questions = TEST_QUESTIONS[: args.limit] if args.limit > 0 else TEST_QUESTIONS
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        docs_with_scores = vector_store.similarity_search_with_score(
            question,
            k=top_k,
            filter=source_filter,
        )
        docs = [doc for doc, _score in docs_with_scores]
        contexts = [doc.page_content for doc in docs]
        retrieval_scores = [float(score) for _doc, score in docs_with_scores]
        retrieved_pages = [
            page + 1
            for doc in docs
            if isinstance((page := doc.metadata.get("page")), int)
        ]
        answer = llm.invoke(build_prompt(question, contexts)).content
        expected_answer = item["expected_answer"]
        row = {
            "question": question,
            "answer": str(answer).strip(),
            "contexts": contexts,
            "retrieval_scores": retrieval_scores,
            "retrieved_pages": retrieved_pages,
            "ground_truth": expected_answer,
            "expected_answer": expected_answer,
            "expected_keywords": item["expected_keywords"],
            "expected_page": item["expected_page"],
            "negative_question": item["negative_question"],
        }
        rows.append(row)
        print(f"[{index}/{len(questions)}] {question} -> {row['answer']}")

    write_jsonl(output_path, rows)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved JSONL: {output_path}")
    print(f"Saved JSON:  {json_path}")
    if not args.skip_cheap_metrics:
        run_cheap_evaluation(rows, output_path)
    if args.ragas:
        run_ragas_evaluation(rows, output_path)
    if args.deepeval:
        run_deepeval_evaluation(rows, output_path, threshold=args.deepeval_threshold)


if __name__ == "__main__":
    main()
