"""LangChain агент с RAG (Qdrant + OpenAI)."""

__all__ = [
    "create_rag_agent_executor",
    "get_retriever",
    "run_agent_query",
]


def __getattr__(name: str):
    if name in __all__:
        from app.agent_rag import agent as _agent

        return getattr(_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
