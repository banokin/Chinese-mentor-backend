"""Общая сборка Qdrant + OpenAI embeddings для agent_rag."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def backend_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def load_agent_rag_env() -> None:
    load_dotenv(backend_env_path())


def default_embedding_model(explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    return (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()


def require_openai_key() -> None:
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise EnvironmentError("OPENAI_API_KEY is not set")


def resolve_qdrant_url(explicit: str | None = None) -> str:
    url = (explicit or os.getenv("QDRANT_URL") or "").strip()
    if not url:
        raise EnvironmentError("QDRANT_URL is not set")
    return url


def make_qdrant_client(*, qdrant_url: str | None = None) -> QdrantClient:
    return QdrantClient(
        url=resolve_qdrant_url(qdrant_url),
        prefer_grpc=False,
        check_compatibility=False,
    )


def make_openai_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=default_embedding_model(model))


def make_vector_store(
    collection_name: str,
    *,
    client: QdrantClient | None = None,
    qdrant_url: str | None = None,
    embeddings: OpenAIEmbeddings | None = None,
    embedding_model: str | None = None,
) -> QdrantVectorStore:
    if embeddings is None:
        embeddings = make_openai_embeddings(embedding_model)
    if client is None:
        client = make_qdrant_client(qdrant_url=qdrant_url)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
