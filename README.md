# Multilingual Citation-Based RAG System

A production-style Retrieval-Augmented Generation (RAG) system built to answer questions strictly from ingested PDF documents. Features include citation tracking, cross-document contradiction detection, confidence scoring, reranking, and multilingual support.

![Architecture Diagram](./architecture.png)

## Features

1. **Multilingual Query Processing**: Ask questions in English, Hindi, or Marathi. Non-English queries are seamlessly translated for optimal English-document retrieval, then translated back for the final answer.
2. **Hallucination Prevention**: The system strictly anchors LLM responses to retrieved context. It clearly states "I could not find this information" if the answer isn't present in the source PDFs.
3. **Citation & Source Tracking**: Every generated answer includes exact file, page, and chunk-level metadata.
4. **Cross-Encoder Reranking**: Initial retrieval via ChromaDB is re-scored by `ms-marco-MiniLM-L-6-v2` for high-precision context selection.
5. **Contradiction Detection**: A dedicated API endpoint and UI page to compare claims between two different documents on a specific topic.

## System Architecture

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: ChromaDB (persistent local storage)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Provider**: Gemini 2.0 Flash or Groq (Llama-3.1)
- **Orchestration**: LangChain

## Chunking Strategy

Documents are split using `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=100`. 
- **Why?** Recursive splitting tries to break on natural linguistic boundaries (paragraphs, sentences, words) before resorting to hard character cuts. This prevents chopping sentences in half, maintaining semantic coherence.
- **Size**: 500 characters provides enough context for the LLM without diluting the retrieval signal.
- **Overlap**: 100 characters ensures no critical information is lost at the boundary of two chunks.

## Setup Instructions

**Prerequisites:** Python 3.10+

1. **Clone and Setup Environment**
   ```bash
   git clone <repo-url>
   cd multilingual-rag
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env to add your GEMINI_API_KEY or GROQ_API_KEY
   ```

3. **Generate Sample Documents & Ingest**
   ```bash
   python generate_docs.py
   python ingest.py
   ```

4. **Run the Application**
   ```bash
   # Start the FastAPI backend (Terminal 1)
   uvicorn api:app --reload
   
   # Start the Streamlit frontend (Terminal 2)
   streamlit run app.py
   ```

## Limitations & Edge Cases

- **Translation Errors**: The system uses `deep-translator` (Google Translate wrapper) which may occasionally misinterpret highly technical jargon in Hindi or Marathi.
- **Chunk Boundary Drops**: Even with a 100-character overlap, highly complex multi-paragraph concepts might be split across chunks, occasionally causing incomplete retrieval.
- **Heuristic Confidence**: The confidence score is a simple heuristic average of the top 3 cross-encoder reranker scores. It represents retrieval confidence, not necessarily factual confidence of the generated text.
- **Probabilistic Contradictions**: The contradiction detection relies on LLM analysis. Edge cases in nuanced language might result in false positive/negative conflict detection.

## Evaluation

An evaluation set is provided in `evaluation/eval_set.json`. Run the automated evaluation suite:
```bash
python evaluation/evaluate.py
```

## AI Usage Disclosure
*This project was developed with the assistance of AI coding tools for scaffolding, boilerplate generation, and initial logic structuring.*

## Screenshots

*(Placeholder for UI screenshots: Ask Question page, Citation Panel expanded, Contradiction Check results)*
