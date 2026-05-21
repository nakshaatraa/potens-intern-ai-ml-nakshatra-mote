"""
RAG engine for the multilingual citation-based system.
Handles document retrieval, answer generation, and citation extraction.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "rag_documents"


def get_embedding_model():
    """Initialize the embedding model used for query encoding."""
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(persist_dir: str = CHROMA_DIR):
    """Load the persisted ChromaDB vector store."""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run ingest.py first."
        )
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve(query: str, k: int = None, source_filter: str = None) -> list:
    """
    Retrieve top-k most relevant chunks for a given query.
    Optionally filter by source document filename.
    """
    top_k = k or int(os.getenv("TOP_K_RETRIEVAL", 10))
    vectorstore = get_vectorstore()

    search_kwargs = {"k": top_k}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    results = vectorstore.similarity_search_with_relevance_scores(
        query, **search_kwargs
    )

    # Each result is (Document, score) — attach score to metadata for downstream use
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


def get_llm_client():
    """
    Initialize the LLM client based on the configured provider.
    Supports Gemini (Google) and Groq as backends.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_name)
        return ("gemini", model)

    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return ("groq", client, model_name)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini' or 'groq'.")


def call_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM and return the response text."""
    llm = get_llm_client()

    if llm[0] == "gemini":
        _, model = llm
        response = model.generate_content(prompt)
        return response.text

    elif llm[0] == "groq":
        _, client, model_name = llm
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content


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
    """
    top_n = top_n or int(os.getenv("TOP_N_RERANK", 5))
    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    try:
        from sentence_transformers import CrossEncoder
        # In a real app, this should be loaded once globally
        encoder = CrossEncoder(model_name)
        
        # Prepare pairs for scoring
        pairs = [[query, chunk.page_content] for chunk in chunks]
        scores = encoder.predict(pairs)
        
        # Attach new scores and sort
        for chunk, score in zip(chunks, scores):
            # Normalize score loosely to 0-1 range for confidence estimation
            import math
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


from utils import detect_language, translate_to_english, translate_from_english

def generate_answer(question: str) -> dict:
    """
    Full RAG pipeline: detect language → translate to EN → retrieve → rerank → 
    build prompt → generate answer → translate back → return structured response.
    """
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
            "confidence": 0.0
        }

    # 3. Rerank and compute confidence
    chunks = rerank(search_query, retrieved_chunks)
    confidence = compute_confidence(chunks)

    # 4. Build context and prompt (Always in English)
    context = format_context(chunks)
    prompt = build_qa_prompt(search_query, context)

    # 5. Generate answer via LLM
    answer_en = call_llm(prompt)
    
    # 6. Translate answer back to source language if needed
    final_answer = answer_en
    if source_lang != "en":
        final_answer = translate_from_english(answer_en, source_lang)

    # 7. Build citation list from top chunks used
    citations = format_citations(chunks)

    return {
        "answer": final_answer,
        "citations": citations,
        "language": source_lang,
        "confidence": confidence
    }


def detect_contradictions(doc1: str, doc2: str, topic: str) -> dict:
    """
    Check for contradictions between two documents on a specific topic.
    Returns whether a conflict exists and the reasoning.
    """
    # Retrieve top chunks for the topic from each document
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
    
    # Simple parser for the expected output format
    conflict = False
    reasoning = response_text
    
    if "CONFLICT: true" in response_text.lower():
        conflict = True
        
    reasoning_split = response_text.split("REASONING:", 1)
    if len(reasoning_split) > 1:
        reasoning = reasoning_split[1].strip()

    return {
        "conflict": conflict,
        "reasoning": reasoning
    }


if __name__ == "__main__":
    query = "What is machine learning?"
    print(f"Query: {query}\n")
    result = generate_answer(query)
    print(f"Answer: {result['answer']}\n")
    print(f"Citations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  - {c['source']} (page {c['page']}, chunk {c['chunk_id']})")

