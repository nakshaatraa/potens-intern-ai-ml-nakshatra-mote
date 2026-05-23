"""
Modified ingestion pipeline using local TF-IDF embeddings (no internet needed).
"""
import os, glob, pickle
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHUNKS_PKL = os.path.join(CHROMA_DIR, "chunks.pkl")
VECTORS_NPY = os.path.join(CHROMA_DIR, "vectors.npy")


def load_documents(docs_dir=DOCS_DIR):
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in {docs_dir}")
    all_docs = []
    for path in sorted(pdf_files):
        loader = PyPDFLoader(path)
        docs = loader.load()
        fname = os.path.basename(path)
        for doc in docs:
            doc.metadata["source"] = fname
        all_docs.extend(docs)
        print(f"  Loaded {len(docs)} pages from {fname}")
    print(f"Total: {len(all_docs)} pages from {len(pdf_files)} docs")
    return all_docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    from embeddings_local import LocalTFIDFEmbeddings
    emb = LocalTFIDFEmbeddings()
    texts = [c.page_content for c in chunks]
    print("Fitting TF-IDF and generating embeddings...")
    vectors = np.array(emb.embed_documents(texts), dtype=np.float32)
    np.save(VECTORS_NPY, vectors)
    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Stored {len(chunks)} chunks + vectors in {CHROMA_DIR}")


if __name__ == "__main__":
    print("=" * 50)
    print("INGESTION PIPELINE (Local TF-IDF)")
    print("=" * 50)
    docs = load_documents()
    chunks = chunk_documents(docs)
    create_vectorstore(chunks)
    print("INGESTION COMPLETE")
