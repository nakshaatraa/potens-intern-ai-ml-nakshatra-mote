"""
Document ingestion pipeline for the multilingual RAG system.
Handles PDF loading, text chunking, embedding generation, and vector store persistence.
"""

import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


def load_documents(docs_dir: str = DOCS_DIR) -> list:
    """
    Load all PDF documents from the specified directory.
    Each page becomes a separate Document object with source and page metadata.
    """
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    all_documents = []
    for pdf_path in sorted(pdf_files):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        filename = os.path.basename(pdf_path)
        for doc in documents:
            doc.metadata["source"] = filename
        all_documents.extend(documents)
        print(f"  Loaded {len(documents)} pages from {filename}")

    print(f"\nTotal: {len(all_documents)} pages from {len(pdf_files)} documents")
    return all_documents


def chunk_documents(documents: list, chunk_size: int = None, chunk_overlap: int = None) -> list:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Strategy rationale:
    - RecursiveCharacterTextSplitter tries to split on natural boundaries (paragraphs,
      sentences, words) before resorting to character-level splits. This preserves
      semantic coherence within chunks better than naive fixed-size splitting.
    - chunk_size=500 balances granularity (too small loses context) with precision
      (too large dilutes relevant information during retrieval).
    - chunk_overlap=100 ensures continuity across chunk boundaries, reducing the risk
      of splitting critical information between two chunks.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 500))
    _chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 100))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Assign a unique chunk_id to each chunk for citation tracking
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(f"Split {len(documents)} pages into {len(chunks)} chunks "
          f"(size={_chunk_size}, overlap={_chunk_overlap})")
    return chunks


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "rag_documents"


def create_vectorstore(chunks: list, persist_dir: str = CHROMA_DIR) -> None:
    """
    Generate embeddings for all chunks and store them in a persistent ChromaDB instance.
    Uses sentence-transformers/all-MiniLM-L6-v2 — a lightweight model that produces
    384-dimensional embeddings optimized for semantic similarity tasks.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading embedding model: {embedding_model}")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Clear existing collection to ensure idempotent re-ingestion
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        print("Cleared existing vector store")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at {persist_dir}")
    return vectorstore


def ingest_pipeline():
    """Run the complete ingestion pipeline: load → chunk → embed → store."""
    print("=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    print("\n[1/3] Loading documents...")
    docs = load_documents()

    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(docs)

    print("\n[3/3] Generating embeddings and storing in ChromaDB...")
    create_vectorstore(chunks)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    return chunks


if __name__ == "__main__":
    ingest_pipeline()
