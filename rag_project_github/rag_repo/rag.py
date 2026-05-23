"""
RAG engine using local TF-IDF retrieval + Gemini LLM.
No internet required for embeddings.
"""
import os, pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHUNKS_PKL = os.path.join(CHROMA_DIR, "chunks.pkl")
VECTORS_NPY = os.path.join(CHROMA_DIR, "vectors.npy")

_chunks = None
_vectors = None
_emb = None

def _load_index():
    global _chunks, _vectors, _emb
    if _chunks is None:
        if not os.path.exists(CHUNKS_PKL):
            raise FileNotFoundError("No index found. Run ingest_local.py first.")
        with open(CHUNKS_PKL, "rb") as f:
            _chunks = pickle.load(f)
        _vectors = np.load(VECTORS_NPY)
    if _emb is None:
        from embeddings_local import LocalTFIDFEmbeddings
        _emb = LocalTFIDFEmbeddings()
    return _chunks, _vectors, _emb


def retrieve(query: str, k: int = 10, source_filter: str = None) -> list:
    chunks, vectors, emb = _load_index()
    q_vec = np.array(emb.embed_query(query), dtype=np.float32)
    # Cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) * np.linalg.norm(q_vec)
    norms = np.where(norms == 0, 1e-9, norms)
    scores = (vectors @ q_vec) / norms.flatten()

    results = []
    for i, score in enumerate(scores):
        c = chunks[i]
        if source_filter and c.metadata.get("source") != source_filter:
            continue
        c.metadata["relevance_score"] = round(float(score), 4)
        results.append((float(score), c))

    results.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in results[:k]]


def format_context(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        src = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        cid = chunk.metadata.get("chunk_id", "?")
        parts.append(f"[Source {i}] (file: {src}, page: {page}, chunk: {cid})\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def call_llm(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        return model.generate_content(prompt).text
    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2048,
        )
        return resp.choices[0].message.content


def build_qa_prompt(question: str, context: str) -> str:
    return f"""You are a precise document question-answering assistant. Answer using ONLY the context below.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Use [Source N] notation to reference sources
- If not found: "I could not find this information in the provided documents."
- Be concise but thorough

ANSWER:"""


def rerank(query: str, chunks: list, top_n: int = 5) -> list:
    # Simple rerank: re-score using TF-IDF and return top_n
    # (cross-encoder skipped — no internet for model download)
    return sorted(chunks, key=lambda c: c.metadata.get("relevance_score", 0), reverse=True)[:top_n]


def compute_confidence(chunks: list) -> float:
    if not chunks:
        return 0.0
    scores = [c.metadata.get("relevance_score", 0.0) for c in chunks[:3]]
    return round(sum(scores) / len(scores), 2)


def format_citations(chunks: list) -> list:
    return [{
        "source": c.metadata.get("source", "unknown"),
        "page": c.metadata.get("page", 0),
        "chunk_id": c.metadata.get("chunk_id", 0),
        "snippet": c.page_content[:300],
        "relevance_score": c.metadata.get("relevance_score", 0.0),
    } for c in chunks]


def generate_answer(question: str) -> dict:
    from utils import detect_language, translate_to_english, translate_from_english
    
    source_lang = detect_language(question)
    search_query = question
    if source_lang != "en":
        search_query = translate_to_english(question, source_lang)

    retrieved = retrieve(search_query, k=10)
    if not retrieved:
        msg = "I could not find this information in the provided documents."
        if source_lang != "en":
            msg = translate_from_english(msg, source_lang)
        return {"answer": msg, "citations": [], "language": source_lang, "confidence": 0.0}

    chunks = rerank(search_query, retrieved)
    confidence = compute_confidence(chunks)
    context = format_context(chunks)
    prompt = build_qa_prompt(search_query, context)
    answer_en = call_llm(prompt)

    final_answer = answer_en
    if source_lang != "en":
        from utils import translate_from_english
        final_answer = translate_from_english(answer_en, source_lang)

    return {
        "answer": final_answer,
        "citations": format_citations(chunks),
        "language": source_lang,
        "confidence": confidence,
    }


def detect_contradictions(doc1: str, doc2: str, topic: str) -> dict:
    chunks1 = retrieve(topic, k=3, source_filter=doc1)
    chunks2 = retrieve(topic, k=3, source_filter=doc2)

    if not chunks1 and not chunks2:
        return {"conflict": False, "reasoning": "Topic not found in either document."}

    context1 = format_context(chunks1) if chunks1 else "No information found in Document 1."
    context2 = format_context(chunks2) if chunks2 else "No information found in Document 2."

    prompt = f"""You are an expert fact-checker analyzing two documents for contradictions on a topic.

TOPIC: {topic}

DOCUMENT 1 ({doc1}):
{context1}

DOCUMENT 2 ({doc2}):
{context2}

INSTRUCTIONS:
1. Compare claims in Document 1 vs Document 2 on this topic.
2. Determine if there is a direct contradiction or conflict.
3. Respond using this format:
   CONFLICT: true/false
   REASONING: Your detailed explanation.

ANALYSIS:"""

    response_text = call_llm(prompt)
    conflict = "conflict: true" in response_text.lower()
    parts = response_text.split("REASONING:", 1)
    reasoning = parts[1].strip() if len(parts) > 1 else response_text

    return {"conflict": conflict, "reasoning": reasoning}
