"""Общая сборка RAG-агента (Qdrant + create_agent), без привязки к языку."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.tools import tool

from app.agent_rag.language_agents import LanguageCode, qdrant_collection_for_language
from app.agent_rag.observability import get_agent_run_config
from app.agent_rag.retriever import build_retriever

logger = logging.getLogger(__name__)


def trim_chat_turns(raw: list[Any], max_messages: int) -> list[Any]:
    if max_messages <= 0 or len(raw) <= max_messages:
        return raw
    trimmed = raw[-max_messages:]
    logger.debug("История чата обрезана: %s → %s сообщений", len(raw), len(trimmed))
    return trimmed


def require_openai_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("Задайте OPENAI_API_KEY в окружении")
    return key


def rag_top_k() -> int:
    return max(1, int(os.getenv("RAG_TOP_K") or "4"))


def get_zh_retriever():
    """Retriever коллекции китайского (скрипты, совместимость)."""
    k = rag_top_k()
    collection = qdrant_collection_for_language("zh")
    qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise ValueError("Задайте QDRANT_URL в окружении")
    return build_retriever(collection_name=collection, qdrant_url=qdrant_url, top_k=k)


def make_search_knowledge_tool(collection_name: str):
    k = rag_top_k()

    @tool
    def search_knowledge_base(query: str) -> str:
        """Поиск по базе знаний. Используй для фактов из документов."""
        qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
        if not qdrant_url:
            return "База знаний недоступна: не задан QDRANT_URL."
        try:
            retriever = build_retriever(
                collection_name=collection_name,
                qdrant_url=qdrant_url,
                top_k=k,
            )
            docs = retriever.invoke(query)
        except Exception as e:
            logger.warning("Ошибка поиска в RAG: %s", e)
            return f"Ошибка поиска в базе знаний: {e}"
        if not docs:
            return "Ничего не найдено в базе знаний."
        snippets = []
        for i, doc in enumerate(docs, start=1):
            text = " ".join(doc.page_content.split())
            snippets.append(f"[{i}] {text[:500]}")
        return "\n\n".join(snippets)

    return search_knowledge_base


def build_rag_agent_executor(
    language: LanguageCode,
    system_prompt: str,
    *,
    verbose: bool = False,
):
    """Создаёт исполнителя агента для языка и системного промпта (без кэша)."""
    require_openai_key()
    collection = qdrant_collection_for_language(language)
    search_tool = make_search_knowledge_tool(collection)
    return create_agent(
        model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip(),
        tools=[search_tool],
        system_prompt=system_prompt,
        debug=verbose,
    )


async def ainvoke_agent(
    executor: Any,
    *,
    question: str,
    chat_history: list[Any] | None,
    verbose: bool,
) -> str:
    raw_history = list(chat_history or [])
    max_turns = max(0, int(os.getenv("AGENT_CHAT_HISTORY_MAX_MESSAGES") or "80"))
    raw_history = trim_chat_turns(raw_history, max_turns)
    payload: list[Any] = [*raw_history, {"role": "user", "content": question.strip()}]
    input_messages = convert_to_messages(payload)
    out = await executor.ainvoke({"messages": input_messages}, config=get_agent_run_config())

    messages = out.get("messages") if isinstance(out, dict) else None
    if messages:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return str(message.content).strip()
    return str(out).strip()
