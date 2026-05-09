"""RAG-агент — репетитор английского (отдельный модуль)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.agent_rag.agent_common import ainvoke_agent, build_rag_agent_executor

EN_SYSTEM_PROMPT = """You are an English tutor for a Russian-speaking learner. Reply in English by default, naturally and briefly (1–3 sentences), like a friendly conversation partner.

Rules:
- Use Russian ONLY if the user explicitly asks, to fix a meaningful mistake, or if they write in Russian and clearly expect it.
- Grammar explanations in Russian only on request or after an error (correction + one short line).
- Always use the knowledge-base search tool for facts from uploaded materials.
- If retrieval is empty, answer briefly without inventing details.

Role-play: stay in character; short natural English; after mistakes — quick fix + optional one-line Russian note.

Follow the conversation history."""


@lru_cache(maxsize=16)
def create_en_rag_agent_executor(verbose: bool = False):
    return build_rag_agent_executor("en", EN_SYSTEM_PROMPT, verbose=verbose)


async def run_en_agent_query(
    question: str,
    *,
    chat_history: list[Any] | None = None,
    verbose: bool = False,
) -> str:
    executor = create_en_rag_agent_executor(verbose=verbose)
    return await ainvoke_agent(
        executor,
        question=question,
        chat_history=chat_history,
        verbose=verbose,
    )
