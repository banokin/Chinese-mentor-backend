import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def build_retriever(
    collection_name: str = "chinese_lexicon",
    qdrant_url: str = "http://localhost:6333",
    top_k: int = 3,
):
    """Create retriever over Qdrant collection with OpenAI embeddings."""
    backend_root = Path(__file__).resolve().parents[2]
    load_dotenv(backend_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = QdrantClient(url=qdrant_url)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": max(1, int(top_k))})


if __name__ == "__main__":
    retriever = build_retriever()
    query = "Как по-китайски сказать спасибо?"
    docs = retriever.invoke(query)
    print(f"Query: {query}")
    print(f"Retrieved documents: {len(docs)}")
    for i, doc in enumerate(docs, start=1):
        preview = " ".join(doc.page_content.split())[:220]
        print(f"{i}. {preview}")