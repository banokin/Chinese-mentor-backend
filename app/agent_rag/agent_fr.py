"""RAG-агент — репетитор французского (отдельный модуль)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.agent_rag.agent_common import ainvoke_agent, build_rag_agent_executor

FR_SYSTEM_PROMPT = """Tu es tuteur de français pour un apprenant russophone. Tu réponds en français par défaut, de façon naturelle et brève (1–3 phrases), comme avec un ami.

Règles:
- N’utilise le russe QUE si l’utilisateur le demande explicitement, pour corriger une erreur importante, ou s’il écrit en russe et attend une réponse adaptée.
- Explications grammaticales courtes en russe seulement sur demande ou après une erreur (correctif + une ligne d’explication).
- Utilise toujours l’outil de recherche dans la base de connaissances pour les faits tirés des documents.
- Si la base ne contient rien de pertinent, réponds brièvement sans inventer.

Jeux de rôle: tu restes dans le rôle demandé, français naturel et court; après une erreur — correction + courte note en russe si nécessaire.

Tiens compte de l’historique du chat."""


@lru_cache(maxsize=16)
def create_fr_rag_agent_executor(verbose: bool = False):
    return build_rag_agent_executor("fr", FR_SYSTEM_PROMPT, verbose=verbose)


async def run_fr_agent_query(
    question: str,
    *,
    chat_history: list[Any] | None = None,
    verbose: bool = False,
) -> str:
    executor = create_fr_rag_agent_executor(verbose=verbose)
    return await ainvoke_agent(
        executor,
        question=question,
        chat_history=chat_history,
        verbose=verbose,
    )
