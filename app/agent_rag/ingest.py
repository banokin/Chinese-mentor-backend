from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.agent_rag.loader_and_splitter import load_pdf, split_documents


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}


def _ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if client.collection_exists(name):
        collection_info = client.get_collection(collection_name=name)
        vectors_config = collection_info.config.params.vectors
        existing_size = (
            vectors_config.size
            if hasattr(vectors_config, "size")
            else list(vectors_config.values())[0].size
        )
        if existing_size != vector_size:
            raise ValueError(
                f"Collection '{name}' has vector size {existing_size}, expected {vector_size}"
            )
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _load_text_document(filename: str, content: bytes) -> list[Document]:
    text = content.decode("utf-8-sig")
    return [
        Document(
            page_content=text,
            metadata={
                "source_file": filename,
                "file_type": Path(filename).suffix.lower().lstrip(".") or "text",
            },
        )
    ]


def load_uploaded_documents(filename: str, content: bytes) -> list[Document]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            documents = load_pdf(tmp.name)
        for doc in documents:
            doc.metadata["source_file"] = filename
        return documents

    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return _load_text_document(filename, content)

    raise ValueError("Поддерживаются только PDF, TXT и MD файлы")


def ingest_documents_to_qdrant(
    documents: Sequence[Document],
    *,
    collection_name: str,
    qdrant_url: str,
    embedding_model: str = "text-embedding-3-small",
) -> dict[str, int | str]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")
    if not documents:
        raise ValueError("Документы для загрузки не найдены")

    chunks = split_documents(list(documents), chunk_size=1000, chunk_overlap=200)
    if not chunks:
        raise ValueError("Не удалось получить текстовые чанки из файла")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_size = len(embeddings.embed_query("проверка эмбеддинга"))
    client = QdrantClient(url=qdrant_url, prefer_grpc=False, check_compatibility=False)
    _ensure_collection(client, collection_name, vector_size)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    points_count = client.count(collection_name=collection_name).count

    return {
        "collection": collection_name,
        "documents": len(documents),
        "chunks": len(chunks),
        "vector_size": vector_size,
        "points_count": points_count,
    }
