import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from app.agent_rag.loader_and_splitter import load_pdf, split_documents
from app.agent_rag.qdrant_factory import (
    default_embedding_model,
    make_openai_embeddings,
    make_qdrant_client,
    require_openai_key,
    resolve_qdrant_url,
)


def ingest_pdf_to_qdrant(
    pdf_path: str,
    collection_name: str = "chinese_lexicon",
    qdrant_url: str | None = None,
    embedding_model: str | None = None,
) -> None:
    """Load PDF, split into chunks, embed and store in configured Qdrant."""
    require_openai_key()
    qdrant_url = resolve_qdrant_url(qdrant_url)

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
    embeddings = make_openai_embeddings(embedding_model)
    vector_size = len(embeddings.embed_query("проверка эмбеддинга"))

    client = make_qdrant_client(qdrant_url=qdrant_url)

    if client.collection_exists(collection_name):
        collection_info = client.get_collection(collection_name=collection_name)
        existing_vectors_config = collection_info.config.params.vectors
        existing_size = (
            existing_vectors_config.size
            if hasattr(existing_vectors_config, "size")
            else list(existing_vectors_config.values())[0].size
        )
        if existing_size != vector_size:
            print(
                f"Recreating collection '{collection_name}' "
                f"because vector size changed {existing_size} -> {vector_size}"
            )
            client.delete_collection(collection_name=collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)

    points_count = client.count(collection_name=collection_name).count
    print(f"Loaded pages: {len(documents)}")
    print(f"Split chunks: {len(chunks)}")
    print(f"Embedding model: {default_embedding_model(embedding_model)}")
    print(f"Vector size: {vector_size}")
    print(f"Indexed vectors in '{collection_name}': {points_count}")


if __name__ == "__main__":
    backend_root = Path(__file__).resolve().parents[2]
    load_dotenv(backend_root / ".env")
    project_root = Path(__file__).resolve().parents[3]
    pdf_file = project_root / "russko-kitajskij_razgovornik.pdf"
    ingest_pdf_to_qdrant(
        str(pdf_file),
        collection_name=(os.getenv("QDRANT_COLLECTION") or "chinese_lexicon").strip(),
        qdrant_url=os.getenv("QDRANT_URL"),
    )
