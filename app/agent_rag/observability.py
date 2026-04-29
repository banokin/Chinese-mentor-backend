from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def _env_is_enabled(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def configure_langsmith_env() -> None:
    """
    Normalize LangSmith env vars for current LangChain versions while keeping
    compatibility with the older LANGCHAIN_* names already used in the project.
    """
    if os.getenv("LANGCHAIN_TRACING_V2") and not os.getenv("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = os.getenv("LANGCHAIN_TRACING_V2", "")
    if os.getenv("LANGSMITH_TRACING") and not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "")
    if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    if os.getenv("LANGCHAIN_PROJECT") and not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")
    if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")


@lru_cache
def _langfuse_callback_handler() -> Any | None:
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    if not public_key or not secret_key:
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning("Langfuse keys are set, but langfuse is not installed")
        return None

    return CallbackHandler()


def get_agent_run_config() -> dict[str, Any]:
    configure_langsmith_env()

    callbacks = []
    langfuse_handler = _langfuse_callback_handler()
    if langfuse_handler is not None:
        callbacks.append(langfuse_handler)

    config: dict[str, Any] = {
        "run_name": "rag_agent_query",
        "tags": ["rag-agent"],
        "metadata": {"component": "agent_rag"},
    }
    if callbacks:
        config["callbacks"] = callbacks
    return config


def log_observability_status() -> None:
    configure_langsmith_env()
    langfuse_enabled = bool(
        (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
        and (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    )
    logger.info(
        "Agent observability: langsmith=%s langfuse=%s",
        _env_is_enabled("LANGSMITH_TRACING"),
        langfuse_enabled,
    )
