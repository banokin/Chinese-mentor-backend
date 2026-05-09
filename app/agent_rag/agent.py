from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.agent_rag.agent_common import (
    ainvoke_agent,
    build_rag_agent_executor,
    get_zh_retriever,
)
from app.agent_rag.language_agents import normalize_language

# Совместимость: старый код вызывает get_retriever()
get_retriever = get_zh_retriever

ZH_SYSTEM_PROMPT = """Ты — Репетитор китайского языка. Твоя задача — обучать пользователя китайскому языку через живое, естественное общение, как с китайским другом. Ты отвечаешь кратко, просто и по делу, если пользователь не просит подробное объяснение. Общайся как с другом, которого давно знаешь.

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

Основные обязанности:
1. Обучай китайскому языку через диалог.
2. Объясняй грамматику, лексику и правила на русском ТОЛЬКО если пользователь просит или явно есть ошибка.
3. Переводи фразы по запросу.
4. Исправляй ошибки кратко: сначала правильный вариант, затем очень короткое объяснение на русском.
5. Давай примеры: иероглифы, pinyin по необходимости, перевод если неочевидно.
6. ВСЕГДА используй инструмент поиска по базе знаний перед ответом, если нужны факты из материалов.
7. Если в базе мало данных — краткий ответ из общих знаний, без выдумок.

Режим ролевой практики: входишь в роль по запросу, отвечаешь кратко по-китайски, после ошибки — исправление + одна строка на русском.

Учитывай историю чата. Отвечай кратко, если не просят иначе."""


@lru_cache(maxsize=16)
def create_rag_agent_executor(verbose: bool = False):
    """Китайский RAG-агент (основной)."""
    return build_rag_agent_executor("zh", ZH_SYSTEM_PROMPT, verbose=verbose)


async def run_agent_query(
    question: str,
    *,
    chat_history: list[Any] | None = None,
    verbose: bool = False,
    language: str = "zh",
) -> str:
    """
    Вызов агента по языку: zh — этот модуль; fr, es, en — см. agent_fr / agent_es / agent_en.
    """
    key = normalize_language(language)
    if key == "zh":
        executor = create_rag_agent_executor(verbose=verbose)
        return await ainvoke_agent(
            executor,
            question=question,
            chat_history=chat_history,
            verbose=verbose,
        )
    if key == "fr":
        from app.agent_rag.agent_fr import run_fr_agent_query

        return await run_fr_agent_query(question, chat_history=chat_history, verbose=verbose)
    if key == "es":
        from app.agent_rag.agent_es import run_es_agent_query

        return await run_es_agent_query(question, chat_history=chat_history, verbose=verbose)
    if key == "en":
        from app.agent_rag.agent_en import run_en_agent_query

        return await run_en_agent_query(question, chat_history=chat_history, verbose=verbose)
    raise AssertionError("normalize_language вернул неожиданный код языка")
