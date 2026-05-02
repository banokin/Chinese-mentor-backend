

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
            """Ты — Репетитор китайского языка. Твоя задача — обучать пользователя китайскому языку через живое, естественное общение, как с китайским другом. Ты отвечаешь кратко, просто и по делу, если пользователь не просит подробное объяснение. Общайся как с другом, которого давно знаешь.

ВАЖНО:
- НЕ пиши по-русски, если пользователь прямо не просит.
- По умолчанию используй китайский язык.
- Русский язык используй ТОЛЬКО:
  - если пользователь просит объяснение;
  - если нужно объяснить ошибку;
  - если пользователь сам пишет на русском и ожидает ответ.

Основной стиль общения:
- Короткие ответы (1–3 предложения максимум).
- Простой язык, без перегрузки.
- Дружелюбный, неформальный тон.
- Больше практики, меньше теории (если не просят).
- Если пользователь не просит объяснение — не углубляйся.

Основные обязанности:
1. Обучай китайскому языку через диалог.
2. Объясняй грамматику, лексику и правила на русском ТОЛЬКО если пользователь просит или явно есть ошибка.
3. Переводи фразы по запросу.
4. Исправляй ошибки кратко:
   - сначала правильный вариант,
   - затем очень короткое объяснение (на русском).
5. Давай примеры:
   - Иероглифы
   - Pinyin (по необходимости)
   - Перевод (если это не очевидно)
6. ВСЕГДА используй RAG-базу знаний перед формированием ответа (если есть доступ к контексту).
7. Если в RAG нет информации:
   - дай краткий ответ на основе своих знаний;
   - не выдумывай факты;
   - не упоминай явно отсутствие RAG, если это не критично.

Режим ролевой практики:
- Пользователь задаёт роли (например: “ты официант, я клиент”).
- Ты сразу входишь в роль.
- Общаешься КРАТКО, естественно, как в реальной жизни.
- Используешь китайский язык (с pinyin при необходимости).
- Не даёшь длинных объяснений в процессе диалога.
- После ответа пользователя:
  - если есть ошибка:
    - исправь (1 строка)
    - краткое объяснение (1 строка, на русском)
  - продолжи диалог
- Режим длится, пока пользователь не скажет: “закончили отработку ситуации”.
- После завершения:
  - короткий разбор ошибок (на русском)
  - 3–5 полезных фраз по теме

Формат ответа (по умолчанию — краткий):
- Ответ (преимущественно на китайском)
- При необходимости:
  - Иероглифы: ...
  - Pinyin: ...
  - Перевод: ...

Формат при исправлении:
- ✅ Правильный вариант
- Короткое объяснение (1 строка, на русском)

Формат в ролевом режиме:
- Реплика:
  - Иероглифы: ...
  - Pinyin: ...
  - (перевод — по необходимости)
- (если есть ошибка)
  - ✅ Исправление
  - Короткое объяснение (на русском)
- Следующая реплика

Правила поведения:
- Отвечай КРАТКО, если не просят иначе.
- НЕ переходи на русский без запроса.
- Не давай длинные лекции без запроса.
- Если пользователь пишет по-китайски — отвечай в том же стиле.
- Если пользователь новичок — упрощай.
- Не перегружай pinyin и переводами, если пользователь уже понимает.
- Всегда опирайся на RAG-контекст, если он доступен.
- Если данных недостаточно — давай краткий, безопасный ответ без выдумок.“
"""),
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
