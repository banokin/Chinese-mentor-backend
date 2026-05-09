"""RAG-агент — репетитор испанского (отдельный модуль)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.agent_rag.agent_common import ainvoke_agent, build_rag_agent_executor

ES_SYSTEM_PROMPT = """Eres tutor de español para un estudiante de habla rusa. Respondes en español por defecto, de forma natural y breve (1–3 frases), como con un amigo.

Reglas:
- Usa el ruso SOLO si el usuario lo pide, para corregir un error clave, o si escribe en ruso y espera esa respuesta.
- Explicaciones gramaticales cortas en ruso solo si las piden o tras un error (corrección + una línea).
- Usa siempre la herramienta de búsqueda en la base de conocimiento para hechos de los documentos.
- Si no hay contexto útil, responde de forma breve sin inventar.

Práctica de roles: mantén el rol, español natural y corto; tras un error — corrección + nota breve en ruso si hace falta.

Ten en cuenta el historial del chat."""


@lru_cache(maxsize=16)
def create_es_rag_agent_executor(verbose: bool = False):
    return build_rag_agent_executor("es", ES_SYSTEM_PROMPT, verbose=verbose)


async def run_es_agent_query(
    question: str,
    *,
    chat_history: list[Any] | None = None,
    verbose: bool = False,
) -> str:
    executor = create_es_rag_agent_executor(verbose=verbose)
    return await ainvoke_agent(
        executor,
        question=question,
        chat_history=chat_history,
        verbose=verbose,
    )
