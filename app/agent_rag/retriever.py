from app.agent_rag.qdrant_factory import (
    load_agent_rag_env,
    make_vector_store,
    require_openai_key,
)


def build_retriever(
    collection_name: str = "chinese_lexicon",
    qdrant_url: str | None = None,
    top_k: int = 3,
):
    """Create retriever over Qdrant collection with OpenAI embeddings."""
    load_agent_rag_env()
    require_openai_key()
    vectorstore = make_vector_store(
        collection_name,
        qdrant_url=qdrant_url,
    )
    return vectorstore.as_retriever(search_kwargs={"k": max(1, int(top_k))})


if __name__ == "__main__":
    load_agent_rag_env()
    retriever = build_retriever()
    query = "Как по-китайски сказать спасибо?"
    docs = retriever.invoke(query)
    print(f"Query: {query}")
    print(f"Retrieved documents: {len(docs)}")
    for i, doc in enumerate(docs, start=1):
        preview = " ".join(doc.page_content.split())[:220]
        print(f"{i}. {preview}")
