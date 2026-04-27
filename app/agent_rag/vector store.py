import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from loader_and_splitter import load_pdf, split_documents


def ingest_pdf_to_qdrant(
    pdf_path: str,
    collection_name: str = "chinese_lexicon",
    qdrant_url: str | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> None:
    """Load PDF, split into chunks, embed and store in configured Qdrant."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")
    qdrant_url = (qdrant_url or os.getenv("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise EnvironmentError("QDRANT_URL is not set")

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_size = len(embeddings.embed_query("проверка эмбеддинга"))

    client = QdrantClient(url=qdrant_url, prefer_grpc=False, check_compatibility=False)

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
    print(f"Embedding model: {embedding_model}")
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