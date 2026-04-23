from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_path: str) -> List[Document]:
    """Load one PDF file into LangChain documents."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = "pdf"

    return documents


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks suitable for RAG indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    # Project root: .../тренажер по произношению по китайскому языку
    project_root = Path(__file__).resolve().parents[3]
    pdf_file = project_root / "russko-kitajskij_razgovornik.pdf"

    docs = load_pdf(str(pdf_file))
    chunks = split_documents(docs)
    print(f"Loaded pages: {len(docs)}")
    print(f"Split chunks: {len(chunks)}")
    if chunks:
        preview = chunks[0].page_content[:300].replace("\n", " ")
        print(f"First chunk preview: {preview}")