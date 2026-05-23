<div align="center">

<img src="./architecture.png" alt="Architecture" width="100%"/>

# 🌐 Multilingual Citation-Based RAG System

### Ask questions in **English**, **Hindi**, or **Marathi** — get answers grounded strictly in your documents.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://potens-intern-ai-ml-nakshatra-mote-pxqbwccqhtrwfbut98mykf.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **"No hallucinations. No guesswork. Just answers — backed by your documents."**

</div>

---

## ✨ What Makes This Different

Most RAG systems give you an answer. This one gives you an answer **you can trust** — with citations, confidence scores, contradiction flags, and support for regional Indian languages built-in.

| Feature | Description |
|---|---|
| 🗣️ **Multilingual** | Ask in English, Hindi, or Marathi |
| 📄 **Citation Tracking** | File + page + chunk-level source attribution |
| 🚫 **Hallucination Prevention** | Strictly anchored to retrieved context — says "I don't know" when needed |
| 🔁 **Cross-Encoder Reranking** | Re-scores retrieval results with `ms-marco-MiniLM-L-6-v2` |
| ⚡ **Contradiction Detection** | Compare claims across two documents via a dedicated endpoint |
| 📊 **Confidence Scoring** | Heuristic confidence from top reranker scores |

---

## 🚀 Live Demo

> Try it now — no setup required.

**👉 [https://potens-intern-ai-ml-nakshatra-mote-pxqbwccqhtrwfbut98mykf.streamlit.app/](https://potens-intern-ai-ml-nakshatra-mote-pxqbwccqhtrwfbut98mykf.streamlit.app/)**

Upload a PDF, ask a question in any supported language, and get a cited, grounded answer in seconds.

---

## 🏗️ System Architecture

```
User Query (EN / HI / MR)
        │
        ▼
 ┌─────────────────┐
 │  Translation    │  ← deep-translator (Google Translate)
 │  (→ English)   │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐     ┌──────────────────────┐
 │  ChromaDB       │────▶│  Cross-Encoder        │
 │  Vector Search  │     │  Reranker             │
 │  (MiniLM-L6-v2) │     │  (ms-marco-MiniLM)   │
 └─────────────────┘     └──────────┬───────────┘
                                    │
                          Top-K Reranked Chunks
                          + Citation Metadata
                                    │
                                    ▼
                         ┌──────────────────┐
                         │  LLM Generation  │
                         │  Gemini 2.0 Flash│
                         │     or Groq      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Back-Translation│  ← if query was non-English
                         │  (→ HI / MR)    │
                         └──────────────────┘
                                  │
                                  ▼
                    Final Answer + Citations + Confidence
```

### Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector DB** | ChromaDB (persistent local) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Gemini 2.0 Flash / Groq (Llama-3.1) |
| **Orchestration** | LangChain |
| **Translation** | `deep-translator` |

---

## 📦 Chunking Strategy

Documents are split using `RecursiveCharacterTextSplitter`:

```python
chunk_size    = 500   # characters
chunk_overlap = 100   # characters
```

**Why this configuration?**

- **Recursive splitting** tries natural linguistic boundaries (paragraphs → sentences → words) before hard cuts — prevents mid-sentence breaks.
- **500 characters** gives the LLM enough context per chunk without diluting the retrieval signal.
- **100-char overlap** ensures no critical information is silently dropped at chunk boundaries.

---

## ⚙️ Setup & Installation

**Prerequisites:** Python 3.10+

### 1. Clone and create environment

```bash
git clone https://github.com/nakshaatraa/potens-intern-ai-ml-nakshatra-mote
cd potens-intern-ai-ml-nakshatra-mote
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY or GROQ_API_KEY
```

### 3. Generate sample docs and ingest

```bash
python generate_docs.py
python ingest.py
```

### 4. Run the application

```bash
# Terminal 1 — FastAPI backend
uvicorn api:app --reload

# Terminal 2 — Streamlit frontend
streamlit run app.py
```

> 🌐 Frontend will open at `http://localhost:8501`  
> 📡 API docs available at `http://localhost:8000/docs`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Ask a question; returns answer + citations + confidence |
| `POST` | `/contradict` | Compare claims between two documents on a topic |
| `POST` | `/ingest` | Ingest a new PDF into the vector store |
| `GET` | `/health` | Health check |

---

## 🧪 Evaluation

An evaluation set is provided in `evaluation/eval_set.json`. Run the automated suite:

```bash
python evaluation/evaluate.py
```

Metrics tracked: retrieval precision, answer faithfulness, citation accuracy.

---

## ⚠️ Known Limitations

| Limitation | Details |
|---|---|
| **Translation Errors** | `deep-translator` may misinterpret highly technical jargon in Hindi/Marathi |
| **Chunk Boundary Drops** | Multi-paragraph concepts may occasionally be split across chunks |
| **Heuristic Confidence** | Confidence = average of top-3 reranker scores; reflects *retrieval* quality, not factual accuracy |
| **Probabilistic Contradictions** | LLM-based — nuanced edge cases may produce false positives/negatives |

---

## 🗺️ Roadmap

- [ ] Add support for more Indian languages (Tamil, Telugu, Gujarati)
- [ ] Persistent user sessions and conversation history
- [ ] Table and chart extraction from PDFs
- [ ] Fine-tuned confidence calibration
- [ ] Docker / cloud deployment guide

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 🙏 AI Usage Disclosure

This project was developed with the assistance of AI coding tools for scaffolding, boilerplate generation, and initial logic structuring.

---

<div align="center">

Made with ❤️ by [Nakshatra Mote](https://github.com/nakshaatraa)

[![Live Demo](https://img.shields.io/badge/🚀_Try_Live_Demo-Click_Here-FF4B4B?style=for-the-badge)](https://potens-intern-ai-ml-nakshatra-mote-pxqbwccqhtrwfbut98mykf.streamlit.app/)

</div>
