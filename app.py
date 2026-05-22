"""
Streamlit frontend for the Multilingual Citation-Based RAG System.
Calls rag.py directly — no separate backend server needed.
"""

import streamlit as st
import os
import glob

# Must be the first Streamlit command
st.set_page_config(
    page_title="Multilingual RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium dark theme ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 30px 35px;
        margin-bottom: 30px;
        color: white;
    }
    .hero-header h1 {
        margin: 0 0 8px 0;
        font-size: 2em;
        font-weight: 700;
    }
    .hero-header p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.05em;
    }

    /* Citation card */
    .citation-card {
        background: linear-gradient(145deg, #1a1f2e, #1e2538);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .citation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1f2e, #1e2538);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-card h3 {
        margin: 0 0 5px 0;
        font-size: 0.85em;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 1.8em;
        font-weight: 700;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-lang {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .badge-warn {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
    }
    .badge-ok {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: #0a0a0a;
    }

    /* Conflict indicators */
    .conflict-true {
        color: #f5576c;
        font-weight: 700;
    }
    .conflict-false {
        color: #00f2fe;
        font-weight: 700;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(145deg, #1a1f2e, #1e2538);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 14px;
        padding: 25px;
        margin: 15px 0;
        line-height: 1.7;
        font-size: 1.02em;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95em;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95em;
    }

    /* Hide Streamlit menu and footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


def get_ingested_documents() -> list[str]:
    """List PDF files in the docs directory."""
    if not os.path.exists(DOCS_DIR):
        return []
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(DOCS_DIR, "*.pdf"))])


# ── Pages ────────────────────────────────────────────────────────────────────

def ask_question_page():
    st.markdown("""
    <div class="hero-header">
        <h1>🔍 Ask a Question</h1>
        <p>Ask in English, Hindi, or Marathi — get a cited, grounded answer in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    from rag import is_ingested
    if not is_ingested():
        st.warning("⚠️ No documents ingested yet. Go to **📥 Setup & Ingest** to get started.")
        return

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is machine learning?  /  मशीन लर्निंग क्या है?",
        label_visibility="collapsed",
    )

    if st.button("🔎 Search & Answer", type="primary") and question:
        with st.spinner("Searching documents and generating answer..."):
            try:
                from rag import generate_answer
                data = generate_answer(question)

                answer = data["answer"]
                citations = data["citations"]
                lang = data.get("language", "en")
                confidence = data.get("confidence", 0.0)

                # ── Answer ───────────────────────────────
                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                # ── Metadata row ─────────────────────────
                col1, col2, col3 = st.columns(3)
                with col1:
                    lang_names = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
                    lang_label = lang_names.get(lang, lang.upper())
                    st.markdown(
                        f'<div class="metric-card"><h3>Language</h3>'
                        f'<span class="badge badge-lang">{lang_label}</span></div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if confidence >= 0.7:
                        color = "#00f2fe"
                    elif confidence >= 0.4:
                        color = "#ffd700"
                    else:
                        color = "#f5576c"
                    st.markdown(
                        f'<div class="metric-card"><h3>Confidence</h3>'
                        f'<span class="value" style="color:{color}">{confidence:.0%}</span></div>',
                        unsafe_allow_html=True,
                    )
                with col3:
                    src_count = len(set(c["source"] for c in citations)) if citations else 0
                    st.markdown(
                        f'<div class="metric-card"><h3>Sources Used</h3>'
                        f'<span class="value" style="color:#667eea">{src_count}</span></div>',
                        unsafe_allow_html=True,
                    )

                if confidence < 0.6 and citations:
                    st.markdown(
                        '<span class="badge badge-warn">⚠️ Low confidence — human review recommended</span>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # ── Citations ────────────────────────────
                st.markdown("### 📑 Sources & Citations")
                if citations:
                    for i, cit in enumerate(citations, 1):
                        score = cit.get("relevance_score", 0)
                        with st.expander(f"Source {i}: {cit['source']}  —  Page {cit['page']}  |  Score: {score:.4f}"):
                            st.markdown(f"**Chunk ID:** `{cit['chunk_id']}`")
                            st.info(cit["snippet"])
                else:
                    st.info("No specific citations were used to generate this answer.")

            except Exception as e:
                st.error(f"❌ Error: {e}")


def contradiction_page():
    st.markdown("""
    <div class="hero-header">
        <h1>⚖️ Contradiction Detection</h1>
        <p>Compare two documents to find conflicting claims on a specific topic.</p>
    </div>
    """, unsafe_allow_html=True)

    from rag import is_ingested
    if not is_ingested():
        st.warning("⚠️ No documents ingested yet. Go to **📥 Setup & Ingest** to get started.")
        return

    docs = get_ingested_documents()
    if len(docs) < 2:
        st.warning("Need at least 2 documents for contradiction detection. Upload more documents in the Setup page.")
        return

    col1, col2 = st.columns(2)
    with col1:
        doc1 = st.selectbox("📄 Document 1", options=docs, key="doc1")
    with col2:
        doc2 = st.selectbox("📄 Document 2", options=docs, key="doc2")

    topic = st.text_input("Topic to compare:", placeholder="e.g., renewable energy emission reduction projections")

    if st.button("🔬 Analyze for Contradictions") and topic:
        if doc1 == doc2:
            st.error("Please select two different documents.")
            return

        with st.spinner("Analyzing documents for contradictions..."):
            try:
                from rag import detect_contradictions
                data = detect_contradictions(doc1, doc2, topic)

                conflict = data["conflict"]
                reasoning = data["reasoning"]

                st.markdown("---")
                if conflict:
                    st.error("🚨 **Contradiction Detected!**")
                else:
                    st.success("✅ **No Contradictions Found**")

                st.markdown(f"**Analysis:**\n\n{reasoning}")

            except Exception as e:
                st.error(f"❌ Error: {e}")


def setup_page():
    st.markdown("""
    <div class="hero-header">
        <h1>📥 Setup & Ingest</h1>
        <p>Generate sample documents, upload your own PDFs, and build the search index.</p>
    </div>
    """, unsafe_allow_html=True)

    from rag import is_ingested

    # ── Status ───────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        docs = get_ingested_documents()
        status_color = "#00f2fe" if docs else "#f5576c"
        st.markdown(
            f'<div class="metric-card"><h3>Documents</h3>'
            f'<span class="value" style="color:{status_color}">{len(docs)}</span></div>',
            unsafe_allow_html=True,
        )
    with col2:
        idx_status = "✅ Ready" if is_ingested() else "❌ Not built"
        idx_color = "#00f2fe" if is_ingested() else "#f5576c"
        st.markdown(
            f'<div class="metric-card"><h3>Search Index</h3>'
            f'<span style="color:{idx_color}; font-weight:700; font-size:1.2em">{idx_status}</span></div>',
            unsafe_allow_html=True,
        )

    if docs:
        st.markdown("**Current documents:**")
        for d in docs:
            st.markdown(f"- 📄 `{d}`")

    st.markdown("---")

    # ── Step 1: Generate Sample Docs ─────────
    st.markdown("### 1️⃣ Generate Sample Documents")
    st.caption("Creates 5 sample PDFs covering AI, Climate, Python, Energy, and Healthcare.")

    if st.button("📝 Generate Sample PDFs"):
        with st.spinner("Generating sample PDF documents..."):
            try:
                from generate_docs import main as generate_main
                generate_main()
                st.success(f"✅ Generated 5 sample PDFs in `docs/`")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating docs: {e}")

    st.markdown("---")

    # ── Step 2: Upload Your Own ──────────────
    st.markdown("### 2️⃣ Upload Your Own PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        os.makedirs(DOCS_DIR, exist_ok=True)
        for uploaded_file in uploaded_files:
            filepath = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Saved `{uploaded_file.name}`")

    st.markdown("---")

    # ── Step 3: Build Index ──────────────────
    st.markdown("### 3️⃣ Build Search Index")
    st.caption("This embeds all documents and builds the vector search index. Takes 1-2 minutes.")

    docs_available = get_ingested_documents()
    if not docs_available:
        st.info("No PDFs found. Generate sample docs or upload your own first.")
    else:
        if st.button("🚀 Build / Rebuild Index", type="primary"):
            with st.spinner("Building search index... this may take a minute."):
                try:
                    from ingest import ingest_pipeline
                    ingest_pipeline()
                    st.success("✅ Search index built successfully! Go to **🔍 Ask Question** to try it out.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error during ingestion: {e}")


def about_page():
    st.markdown("""
    <div class="hero-header">
        <h1>ℹ️ About the System</h1>
        <p>Architecture, tech stack, and design decisions behind this RAG system.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Multilingual Citation-Based RAG

    **"No hallucinations. No guesswork. Just answers — backed by your documents."**

    This system implements a production-style Retrieval-Augmented Generation pipeline with:

    - 🌐 **Multilingual Support** — English, Hindi, and Marathi query processing
    - 📑 **Citation Tracking** — Strict hallucination prevention with exact source mapping
    - ⚖️ **Contradiction Detection** — Cross-document fact-checking via LLM analysis
    - 📊 **Confidence Scoring** — Heuristic confidence based on reranked relevance scores

    ---

    ### 🏗️ Architecture

    ```
    User Query (EN / HI / MR)
            │
            ▼
    ┌─────────────────┐
    │   Translation    │ ← deep-translator (Google Translate)
    │   (→ English)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     ┌──────────────────────┐
    │    ChromaDB      │────▶│   Cross-Encoder       │
    │  Vector Search   │     │     Reranker           │
    │ (MiniLM-L6-v2)  │     │ (ms-marco-MiniLM)     │
    └─────────────────┘     └──────────┬───────────┘
                                       │
                          Top-K Reranked Chunks + Citations
                                       │
                                       ▼
                            ┌──────────────────┐
                            │  LLM Generation   │
                            │  Gemini 2.0 Flash │
                            │     or Groq       │
                            └────────┬─────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │ Back-Translation  │ ← if query was non-English
                            │   (→ HI / MR)     │
                            └──────────────────┘
                                     │
                                     ▼
                        Final Answer + Citations + Confidence
    ```

    ---

    ### 📦 Tech Stack

    | Component | Technology |
    |-----------|-----------|
    | Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
    | Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
    | Vector DB | ChromaDB |
    | LLM | Gemini 2.0 Flash / Groq |
    | Translation | `deep-translator` (Google Translate) |
    | Chunking | `RecursiveCharacterTextSplitter` (500 chars, 100 overlap) |
    | Frontend | Streamlit |

    ---

    ### 🙏 AI Usage Disclosure
    This project was developed with the assistance of AI coding tools for scaffolding,
    boilerplate generation, and initial logic structuring.

    Made with ❤️ by [Nakshatra Mote](https://github.com/nakshaatraa)
    """)


# ── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 15px 0;">
    <span style="font-size: 2em;">🔍</span>
    <h2 style="margin: 5px 0 0 0; font-weight: 700;">RAG System</h2>
    <p style="opacity: 0.6; font-size: 0.85em; margin: 0;">Multilingual • Citation-Based</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Ask Question", "⚖️ Contradiction Check", "📥 Setup & Ingest", "ℹ️ About"],
    label_visibility="collapsed",
)

# Show quick status in sidebar
from rag import is_ingested
if is_ingested():
    st.sidebar.success("✅ Index ready")
else:
    st.sidebar.warning("⚠️ No index — run Setup first")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Gemini + ChromaDB + Streamlit")

# ── Route to selected page ──────────────────────────────────────────────────
if page == "🔍 Ask Question":
    ask_question_page()
elif page == "⚖️ Contradiction Check":
    contradiction_page()
elif page == "📥 Setup & Ingest":
    setup_page()
else:
    about_page()
