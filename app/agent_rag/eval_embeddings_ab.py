import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from loader_and_splitter import load_pdf, split_documents


@dataclass
class EvalCase:
    query: str
    expected_fragments: Sequence[str]


EVAL_CASES: List[EvalCase] = [
    EvalCase("Как по-китайски сказать «Здравствуйте»?", ["你好"]),
    EvalCase("Как сказать «Спасибо» на китайском?", ["谢谢"]),
    EvalCase("Как сказать «Пожалуйста»?", ["不客气", "不用谢"]),
    EvalCase("Как сказать «До свидания» по-китайски?", ["再见"]),
    EvalCase("Как попросить прощения: «Извините»?", ["对不起"]),
    EvalCase("Как спросить «Сколько это стоит?»", ["多少钱"]),
    EvalCase("Как сказать «Я не понимаю» на китайском?", ["我不明白", "我听不懂"]),
    EvalCase("Как сказать «Где туалет?»", ["厕所", "洗手间"]),
]


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def contains_any(text: str, fragments: Sequence[str]) -> bool:
    compact_text = text.replace(" ", "").replace("\u00a0", "")
    return any(fragment in compact_text for fragment in fragments)


def evaluate_store(vector_store: QdrantVectorStore, cases: Sequence[EvalCase]) -> dict:
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    precision_sum_at_1 = 0.0
    precision_sum_at_3 = 0.0
    precision_sum_at_5 = 0.0
    reciprocal_rank_sum = 0.0

    for case in cases:
        docs = vector_store.similarity_search(case.query, k=5)
        top_texts = [doc.page_content for doc in docs]
        relevance = [contains_any(text, case.expected_fragments) for text in top_texts]

        top1_hit = any(relevance[:1])
        top3_hit = any(relevance[:3])
        top5_hit = any(relevance[:5])

        hit_at_1 += int(top1_hit)
        hit_at_3 += int(top3_hit)
        hit_at_5 += int(top5_hit)
        precision_sum_at_1 += sum(relevance[:1]) / 1
        precision_sum_at_3 += sum(relevance[:3]) / 3
        precision_sum_at_5 += sum(relevance[:5]) / 5

        first_relevant_rank = next((idx + 1 for idx, is_rel in enumerate(relevance) if is_rel), None)
        reciprocal_rank_sum += 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

        print(
            f"Query: {case.query}\n"
            f"  expected: {list(case.expected_fragments)}\n"
            f"  hit@1={top1_hit}, hit@3={top3_hit}, hit@5={top5_hit}\n"
            f"  precision@1={sum(relevance[:1]) / 1:.3f}, "
            f"precision@3={sum(relevance[:3]) / 3:.3f}, "
            f"precision@5={sum(relevance[:5]) / 5:.3f}, "
            f"rr={(0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank):.3f}"
        )

    total = len(cases)
    return {
        "recall@1": hit_at_1 / total,
        "recall@3": hit_at_3 / total,
        "recall@5": hit_at_5 / total,
        "precision@1": precision_sum_at_1 / total,
        "precision@3": precision_sum_at_3 / total,
        "precision@5": precision_sum_at_5 / total,
        "mrr@5": reciprocal_rank_sum / total,
        "cases": total,
    }


def build_store(
    client: QdrantClient,
    collection_name: str,
    embeddings,
    chunks,
) -> QdrantVectorStore:
    vector_size = len(embeddings.embed_query("контрольный запрос"))
    ensure_collection(client, collection_name, vector_size)

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    store.add_documents(chunks)
    return store


def main() -> None:
    backend_root = Path(__file__).resolve().parents[2]
    load_dotenv(backend_root / ".env")
    hf_cache_dir = backend_root / ".hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    project_root = Path(__file__).resolve().parents[3]
    pdf_file = project_root / "russko-kitajskij_razgovornik.pdf"

    qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise EnvironmentError("QDRANT_URL is not set")
    client = QdrantClient(url=qdrant_url, prefer_grpc=False, check_compatibility=False)

    docs = load_pdf(str(pdf_file))
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
    print(f"Loaded pages: {len(docs)}, chunks: {len(chunks)}")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\n=== OpenAI: text-embedding-3-small ===")
    openai_store = build_store(
        client=client,
        collection_name="eval_phrasebook_openai",
        embeddings=openai_embeddings,
        chunks=chunks,
    )
    openai_metrics = evaluate_store(openai_store, EVAL_CASES)
    print(f"OpenAI metrics: {openai_metrics}")

    print("\n=== HuggingFace: BAAI/bge-m3 ===")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        cache_folder=str(hf_cache_dir),
        encode_kwargs={"normalize_embeddings": True},
    )
    hf_store = build_store(
        client=client,
        collection_name="eval_phrasebook_hf_bge_m3",
        embeddings=hf_embeddings,
        chunks=chunks,
    )
    hf_metrics = evaluate_store(hf_store, EVAL_CASES)
    print(f"HF metrics: {hf_metrics}")


if __name__ == "__main__":
    main()
