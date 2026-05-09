

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import sacrebleu
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer

from app.agent_rag.ingest import ingest_documents_to_qdrant
from app.agent_rag.metrics import tokenize
from app.agent_rag.qdrant_factory import (
    default_embedding_model,
    load_agent_rag_env,
    make_qdrant_client,
    make_vector_store,
    require_openai_key,
    resolve_qdrant_url,
)


def _load_backend_env() -> None:
    backend_root = Path(__file__).resolve().parents[2]
    load_dotenv(backend_root / ".env")
    load_agent_rag_env()


def _demo_collection_name() -> str:
    return (os.getenv("RAG_DEMO_QDRANT_COLLECTION") or "pronunciation_trainer_rag_demo").strip()


def _demo_top_k() -> int:
    return max(1, int(os.getenv("RAG_DEMO_TOP_K") or "2"))


# 1. Мини-база знаний: китайское произношение и контекст приложения
docs: list[Document] = [
    Document(
        page_content=(
            "Стандартное произношение китайского (普通话) строится на слогах: "
            "начальная согласная (声母), финаль гласная (韵母) и тон (声调). "
            "Пиньинь записывает произношение; без тона слог может означать разные слова."
        )
    ),
    Document(
        page_content=(
            "В мандаринском четыре основных тона плюс нейтральный: "
            "первый высокий ровный, второй восходящий, третий падающе-восходящий, "
            "четвёртый резко падающий. Ошибки в тоне часто меняют смысл фразы."
        )
    ),
    Document(
        page_content=(
            "Русскоязычные ученики часто путают аспирированные и неаспирированные "
            "согласные (например p / t / k в пиньини сопоставляют с русскими «мягкими» звуками "
            "неправильно). Полезно тренировать различение через минимальные пары слогов и запись."
        )
    ),
    Document(
        page_content=(
            "RAG в этом тренажёре использует релевантные фрагменты из материалов "
            "(например разговорника), чтобы ответы опирались на заданный контент, "
            "а не только на общие знания модели."
        )
    ),
]


def _collection_needs_seed(collection: str) -> bool:
    client = make_qdrant_client()
    if not client.collection_exists(collection):
        return True
    return client.count(collection_name=collection).count == 0


def _ensure_demo_index() -> None:
    collection = _demo_collection_name()
    if not _collection_needs_seed(collection):
        return
    ingest_documents_to_qdrant(
        docs,
        collection_name=collection,
        qdrant_url=resolve_qdrant_url(),
        embedding_model=default_embedding_model(),
    )


_load_backend_env()
require_openai_key()
_ensure_demo_index()

_vector_store = make_vector_store(
    _demo_collection_name(),
    qdrant_url=resolve_qdrant_url(),
)
retriever = _vector_store.as_retriever(search_kwargs={"k": _demo_top_k()})


@tool
def search_knowledge_base(query: str) -> str:
    """Ищет информацию в базе знаний о китайском произношении и приложении."""
    found_docs = retriever.invoke(query)
    if not found_docs:
        return "NO_CONTEXT"
    return "\n\n".join(doc.page_content for doc in found_docs)


_llm = ChatOpenAI(
    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
    temperature=0,
)

agent = create_agent(
    model=_llm,
    tools=[search_knowledge_base],
    system_prompt=(
        "Ты RAG-ассистент тренажёра китайского произношения. "
        "Всегда сначала вызывай инструмент search_knowledge_base. "
        "Отвечай только по найденному контексту, кратко и по делу. "
        "Если контекста нет (NO_CONTEXT или пусто), скажи: "
        "'В базе знаний нет ответа.'"
    ),
)


def token_precision_recall(prediction: str, reference: str) -> tuple[float, float]:
    """
    Токеновые precision/recall (мультимножества токенов, как в token_f1 в metrics.py).
    Подходит для русско-китайского текста за счёт общего tokenize().
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0
    if not pred_tokens:
        return 0.0, 0.0
    if not ref_tokens:
        return 0.0, 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return float(precision), float(recall)


def ask_agent(question: str) -> str:
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question},
            ]
        }
    )
    messages = result.get("messages") if isinstance(result, dict) else None
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is not None:
            return str(content)
    return str(result)


def compute_metrics(prediction: str, reference: str) -> dict[str, float]:
    bleu = sacrebleu.sentence_bleu(
        hypothesis=prediction,
        references=[reference],
    ).score

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    rouge = scorer.score(reference, prediction)

    precision_tok, recall_tok = token_precision_recall(prediction, reference)

    return {
        "bleu": float(bleu),
        "rouge1_f1": rouge["rouge1"].fmeasure,
        "rouge2_f1": rouge["rouge2"].fmeasure,
        "rougeL_f1": rouge["rougeL"].fmeasure,
        "token_precision": precision_tok,
        "token_recall": recall_tok,
    }


eval_dataset: list[dict[str, str]] = [
    {
        "question": "Сколько основных тонов в мандаринском и какие они?",
        "reference": (
            "В мандаринском четыре основных тона плюс нейтральный: первый высокий ровный, "
            "второй восходящий, третий падающе-восходящий, четвёртый резко падающий."
        ),
    },
    {
        "question": "Из чего состоит китайский слог в пиньине?",
        "reference": (
            "Начальная согласная (声母), финаль гласная (韵母) и тон (声调)."
        ),
    },
    {
        "question": "Для чего в тренажёре используют RAG?",
        "reference": (
            "Чтобы ответы опирались на фрагменты из материалов, например разговорника."
        ),
    },
]


if __name__ == "__main__":
    print("Collection:", _demo_collection_name())
    print("Qdrant:", resolve_qdrant_url())

    all_metrics: list[dict[str, float]] = []

    for item in eval_dataset:
        question = item["question"]
        reference = item["reference"]
        prediction = ask_agent(question)
        metrics = compute_metrics(prediction, reference)
        all_metrics.append(metrics)

        print("=" * 80)
        print("QUESTION:", question)
        print("PREDICTION:", prediction)
        print("REFERENCE:", reference)
        print("METRICS:", metrics)

    n = len(all_metrics)
    print("\nAVERAGE METRICS:")
    print(
        {
            "bleu": sum(m["bleu"] for m in all_metrics) / n,
            "rouge1_f1": sum(m["rouge1_f1"] for m in all_metrics) / n,
            "rouge2_f1": sum(m["rouge2_f1"] for m in all_metrics) / n,
            "rougeL_f1": sum(m["rougeL_f1"] for m in all_metrics) / n,
            "token_precision": sum(m["token_precision"] for m in all_metrics) / n,
            "token_recall": sum(m["token_recall"] for m in all_metrics) / n,
        }
    )
