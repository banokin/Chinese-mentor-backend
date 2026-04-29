

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.agent_rag.observability import get_agent_run_config
from app.agent_rag.retriever import build_retriever

logger = logging.getLogger(__name__)


def _require_openai_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("Задайте OPENAI_API_KEY в окружении")
    return key


def get_retriever():
    """
    Retriever поверх коллекции Qdrant (dense, OpenAI embeddings).
    Коллекция должна уже существовать и содержать документы с тем же размером эмбеддингов.
    """
    k = max(1, int(os.getenv("RAG_TOP_K") or "4"))
    collection = (os.getenv("QDRANT_COLLECTION") or "chinese_lexicon").strip()
    qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise ValueError("Задайте QDRANT_URL в окружении")
    return build_retriever(collection_name=collection, qdrant_url=qdrant_url, top_k=k)


@tool
def search_knowledge_base(query: str) -> str:
    """Поиск по базе знаний. Используй для фактов из документов."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "Ничего не найдено в базе знаний."
    snippets = []
    for i, doc in enumerate(docs, start=1):
        text = " ".join(doc.page_content.split())
        snippets.append(f"[{i}] {text[:500]}")
    return "\n\n".join(snippets)


@lru_cache
def create_rag_agent_executor(verbose: bool = False):
    _require_openai_key()
    return create_agent(
        model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip(),
        tools=[search_knowledge_base],
        system_prompt=(
            "Ты помощник с доступом к базе знаний через инструмент search_knowledge_base. "
            "Сначала ищи релевантный контекст, если вопрос опирается на факты из документов. "
            "Отвечай чётко; при необходимости цитируй суть найденного."
        ),
        debug=verbose,
    )


async def run_agent_query(
    question: str,
    *,
    chat_history: list[Any] | None = None,
    verbose: bool = False,
) -> str:
    """
    Один асинхронный вызов агента (удобно из FastAPI).
    """
    executor = create_rag_agent_executor(verbose=verbose)
    history = chat_history or []
    input_messages = [*history, {"role": "user", "content": question.strip()}]
    out = await executor.ainvoke({"messages": input_messages}, config=get_agent_run_config())

    messages = out.get("messages") if isinstance(out, dict) else None
    if messages:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return str(message.content).strip()
    return str(out).strip()
