# 🔍 Multilingual RAG System

A production-style **Retrieval-Augmented Generation** pipeline with a Streamlit UI that answers questions from PDF documents with exact citations, supports English/Hindi/Marathi, and detects cross-document contradictions.

## ✨ Features

- **Multilingual**: Detects and handles English, Hindi, Marathi queries automatically
- **Citation tracking**: Every answer comes with source document, page number, and relevance score
- **Contradiction detection**: Compare two documents on any topic to find conflicts
- **Confidence scoring**: Heuristic confidence score with human-review flag for low-confidence answers
- **Modular LLM backend**: Supports Google Gemini and Groq (Llama) out of the box

## 🏗️ Architecture

```
PDF docs → PyPDF loader → RecursiveCharacterTextSplitter (500 chars, 100 overlap)
         → TF-IDF embeddings → Local vector index (numpy + sklearn)
         → TF-IDF retrieval (cosine similarity, top-10)
         → Reranking (top-5 by score)
         → LLM prompt (Gemini 2.0 Flash / Groq Llama)
         → Language translation back if needed
         → Streamlit UI with citations
```

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/multilingual-rag
cd multilingual-rag
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY or GROQ_API_KEY
```

### 3. Add your PDF documents
```bash
mkdir -p docs
# Drop your PDF files into the docs/ folder
# Or generate sample docs:
python create_sample_docs.py
```

### 4. Ingest documents
```bash
python ingest.py
```

### 5. Start the API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 6. Start the Streamlit UI (in a new terminal)
```bash
streamlit run app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

## 🐳 Docker Compose

```bash
cp .env.example .env  # fill in your API keys
mkdir docs && python create_sample_docs.py  # or add your own PDFs
python ingest.py                            # build the index
docker-compose up --build
```

- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

## 🌐 Deploy to Streamlit Community Cloud (Free Demo Link)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, set `app.py` as the main file
4. Add secrets in the Streamlit dashboard:
   ```toml
   GEMINI_API_KEY = "your_key"
   LLM_PROVIDER = "gemini"
   API_URL = "https://your-api-backend-url"
   ```
5. Click **Deploy** — you get a `https://your-app.streamlit.app` link instantly

> **Note**: For Streamlit Cloud you'll need to host the FastAPI backend separately (e.g. on Railway, Render, or Fly.io) and set `API_URL` to point to it.

## 📁 Project Structure

```
├── app.py                  # Streamlit frontend
├── api.py                  # FastAPI backend
├── rag.py                  # RAG engine (retrieval + generation)
├── ingest.py               # PDF ingestion pipeline
├── utils.py                # Language detection & translation
├── embeddings_local.py     # Local TF-IDF embeddings (no HuggingFace needed)
├── create_sample_docs.py   # Generate sample PDFs for demo
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── docs/                   # Place your PDFs here
```

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `groq` |
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `GROQ_API_KEY` | — | Groq API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `10` | Chunks retrieved |
| `TOP_N_RERANK` | `5` | Chunks after reranking |
| `API_URL` | `http://localhost:8000` | FastAPI backend URL |

## 📝 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/documents` | List ingested PDFs |
| POST | `/ask` | Ask a question (RAG pipeline) |
| POST | `/contradict` | Check contradictions between two docs |

Interactive docs: `http://localhost:8000/docs`
