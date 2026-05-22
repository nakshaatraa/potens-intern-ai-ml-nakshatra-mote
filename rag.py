"""
RAG engine for the multilingual citation-based system.
Handles document retrieval, answer generation, and citation extraction.
"""

import os
import math
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "rag_documents"

# ── Global model cache ──────────────────────────────────────────────────────
_embedding_model = None
_cross_encoder = None


def get_embedding_model():
    """Initialize the embedding model (cached globally)."""
    global _embedding_model
    if _embedding_model is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def get_cross_encoder():
    """Initialize the cross-encoder reranker (cached globally)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


import pickle
import numpy as np

class SimpleVectorStore:
    """
    A pure-Python persistent vector store that implements similarity search.
    Provides identical API to langchain's Chroma for the needs of this application.
    """
    def __init__(self, persist_directory, embedding_function):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.db_path = os.path.join(persist_directory, "vector_store.pkl")
        self.documents = []
        self.embeddings = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    self.embeddings = data.get("embeddings", [])
                    print(f"Loaded {len(self.documents)} documents from {self.db_path}")
            except Exception as e:
                print(f"Error loading vector store from {self.db_path}: {e}")
                self.documents = []
                self.embeddings = []
        else:
            self.documents = []
            self.embeddings = []

    def save(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "embeddings": self.embeddings
                }, f)
            print(f"Saved {len(self.documents)} documents to {self.db_path}")
        except Exception as e:
            print(f"Error saving vector store to {self.db_path}: {e}")

    @classmethod
    def from_documents(cls, documents, embedding, persist_directory, collection_name=None):
        instance = cls(persist_directory, embedding)
        instance.documents = documents
        
        # Embed all documents
        print(f"Generating embeddings for {len(documents)} documents using {embedding.__class__.__name__}...")
        texts = [doc.page_content for doc in documents]
        instance.embeddings = embedding.embed_documents(texts)
        instance.save()
        return instance

    def similarity_search_with_relevance_scores(self, query, k=10, filter=None):
        if not self.documents:
            return []
            
        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Calculate cosine similarities
        q_vec = np.array(query_embedding)
        db_vecs = np.array(self.embeddings)
        
        if len(db_vecs) == 0:
            return []
            
        # Normalization
        q_norm = np.linalg.norm(q_vec)
        db_norms = np.linalg.norm(db_vecs, axis=1)
        
        # Prevent division by zero
        db_norms = np.where(db_norms == 0, 1e-10, db_norms)
        if q_norm == 0:
            q_norm = 1e-10
            
        scores = np.dot(db_vecs, q_vec) / (db_norms * q_norm)
        
        # Pair documents with their scores and apply metadata filtering
        results = []
        for doc, score in zip(self.documents, scores):
            if filter:
                matched = True
                for key, val in filter.items():
                    if doc.metadata.get(key) != val:
                        matched = False
                        break
                if not matched:
                    continue
            # Cosine similarity is in [-1, 1], map or clip to relevance score [0, 1]
            relevance_score = float((score + 1.0) / 2.0) if score >= -1.0 else 0.0
            results.append((doc, relevance_score))
            
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


def get_vectorstore(persist_dir: str = CHROMA_DIR):
    """Load the persisted SimpleVectorStore."""
    db_file = os.path.join(persist_dir, "vector_store.pkl")
    if not os.path.exists(db_file):
        return None
    embeddings = get_embedding_model()
    return SimpleVectorStore(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def is_ingested() -> bool:
    """Check whether documents have been ingested into the vector store."""
    db_file = os.path.join(CHROMA_DIR, "vector_store.pkl")
    return os.path.exists(db_file) and os.path.getsize(db_file) > 0


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = None, source_filter: str = None) -> list:
    """
    Retrieve top-k most relevant chunks for a given query.
    Optionally filter by source document filename.
    """
    top_k = k or int(os.getenv("TOP_K_RETRIEVAL", 10))
    vectorstore = get_vectorstore()

    if vectorstore is None:
        return []

    search_kwargs = {"k": top_k}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query, **search_kwargs
        )
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

    retrieved = []
    for doc, score in results:
        doc.metadata["relevance_score"] = round(float(score), 4)
        retrieved.append(doc)

    return retrieved


def format_context(chunks: list) -> str:
    """Format retrieved chunks into a numbered context block for the LLM prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        chunk_id = chunk.metadata.get("chunk_id", "?")
        context_parts.append(
            f"[Source {i}] (file: {source}, page: {page}, chunk: {chunk_id})\n"
            f"{chunk.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


# ── LLM ──────────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM and return the response text."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text

    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini' or 'groq'.")


def build_qa_prompt(question: str, context: str) -> str:
    """Construct the QA prompt with context and citation instructions."""
    return f"""You are a precise document question-answering assistant. Your task is to answer the user's question using ONLY the information provided in the context below.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question based ONLY on the provided context
- Reference specific sources using [Source N] notation in your answer
- If the answer cannot be found in the context, respond with: "I could not find this information in the provided documents."
- Be concise but thorough
- Do not make up or infer information beyond what is explicitly stated

ANSWER:"""


# ── Reranking & Confidence ───────────────────────────────────────────────────

def format_citations(chunks: list) -> list[dict]:
    """Convert retrieved chunks into structured citation objects."""
    citations = []
    for chunk in chunks:
        citations.append({
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", 0),
            "chunk_id": chunk.metadata.get("chunk_id", 0),
            "snippet": chunk.page_content[:300],
            "relevance_score": chunk.metadata.get("relevance_score", 0.0),
        })
    return citations


def rerank(query: str, chunks: list, top_n: int = None) -> list:
    """
    Rerank retrieved chunks using a cross-encoder model for improved precision.
    Falls back to original ordering if reranking fails.
    """
    top_n = top_n or int(os.getenv("TOP_N_RERANK", 5))

    try:
        encoder = get_cross_encoder()
        pairs = [[query, chunk.page_content] for chunk in chunks]
        scores = encoder.predict(pairs)

        for chunk, score in zip(chunks, scores):
            normalized_score = 1 / (1 + math.exp(-score))
            chunk.metadata["relevance_score"] = round(normalized_score, 4)

        chunks.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)
        return chunks[:top_n]
    except Exception as e:
        print(f"Reranking failed: {e}. Returning original chunks.")
        return chunks[:top_n]


def compute_confidence(chunks: list) -> float:
    """
    Compute a heuristic confidence score based on the top reranked chunks.
    Takes the average relevance score of the top 3 chunks.
    """
    if not chunks:
        return 0.0
    scores = [c.metadata.get("relevance_score", 0.0) for c in chunks[:3]]
    return round(sum(scores) / len(scores), 2)


# ── Main Pipeline ────────────────────────────────────────────────────────────

from utils import detect_language, translate_to_english, translate_from_english


def generate_answer(question: str) -> dict:
    """
    Full RAG pipeline: detect language → translate to EN → retrieve → rerank →
    build prompt → generate answer → translate back → return structured response.
    """
    if not is_ingested():
        return {
            "answer": "No documents have been ingested yet. Please upload and ingest documents first.",
            "citations": [],
            "language": "en",
            "confidence": 0.0,
        }

    # 1. Language Detection & Translation
    source_lang = detect_language(question)
    search_query = question

    if source_lang != "en":
        search_query = translate_to_english(question, source_lang)
        print(f"Translated query ({source_lang} -> en): {search_query}")

    # 2. Retrieve relevant chunks
    retrieved_chunks = retrieve(search_query, k=10)

    if not retrieved_chunks:
        fallback_msg = "I could not find this information in the provided documents."
        if source_lang != "en":
            fallback_msg = translate_from_english(fallback_msg, source_lang)
        return {
            "answer": fallback_msg,
            "citations": [],
            "language": source_lang,
            "confidence": 0.0,
        }

    # 3. Rerank and compute confidence
    chunks = rerank(search_query, retrieved_chunks)
    confidence = compute_confidence(chunks)

    # 4. Build context and prompt (always in English)
    context = format_context(chunks)
    prompt = build_qa_prompt(search_query, context)

    # 5. Generate answer via LLM
    answer_en = call_llm(prompt)

    # 6. Translate answer back to source language if needed
    final_answer = answer_en
    if source_lang != "en":
        final_answer = translate_from_english(answer_en, source_lang)

    # 7. Build citation list
    citations = format_citations(chunks)

    return {
        "answer": final_answer,
        "citations": citations,
        "language": source_lang,
        "confidence": confidence,
    }


def detect_contradictions(doc1: str, doc2: str, topic: str) -> dict:
    """
    Check for contradictions between two documents on a specific topic.
    Returns whether a conflict exists and the reasoning.
    """
    chunks1 = retrieve(topic, k=3, source_filter=doc1)
    chunks2 = retrieve(topic, k=3, source_filter=doc2)

    if not chunks1 and not chunks2:
        return {"conflict": False, "reasoning": "Topic not found in either document."}

    context1 = format_context(chunks1) if chunks1 else "No information found in Document 1."
    context2 = format_context(chunks2) if chunks2 else "No information found in Document 2."

    prompt = f"""You are an expert fact-checker analyzing two different documents for contradictions on a specific topic.

TOPIC: {topic}

DOCUMENT 1 ({doc1}):
{context1}

DOCUMENT 2 ({doc2}):
{context2}

INSTRUCTIONS:
1. Compare the claims made in Document 1 against Document 2 regarding the topic.
2. Determine if there is a direct contradiction or conflict in facts, statistics, or conclusions.
3. Output your response as a JSON-like structure (but plain text):
   CONFLICT: true/false
   REASONING: Detailed explanation of the conflict or lack thereof.

ANALYSIS:"""

    response_text = call_llm(prompt)

    conflict = False
    reasoning = response_text

    if "conflict: true" in response_text.lower():
        conflict = True

    reasoning_split = response_text.split("REASONING:", 1)
    if len(reasoning_split) > 1:
        reasoning = reasoning_split[1].strip()

    return {
        "conflict": conflict,
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    query = "What is machine learning?"
    print(f"Query: {query}\n")
    result = generate_answer(query)
    print(f"Answer: {result['answer']}\n")
    print(f"Citations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  - {c['source']} (page {c['page']}, chunk {c['chunk_id']})")
