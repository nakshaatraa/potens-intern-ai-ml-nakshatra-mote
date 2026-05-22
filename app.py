import streamlit as st
import httpx
import pandas as pd

# Must be the first Streamlit command
st.set_page_config(
    page_title="Multilingual RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# Custom CSS for a premium look
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .citation-card {
        background-color: #1E2329;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #4CAF50;
    }
    .metric-card {
        background-color: #1E2329;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .conflict-true {
        color: #FF5252;
        font-weight: bold;
    }
    .conflict-false {
        color: #4CAF50;
        font-weight: bold;
    }
    .language-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #2196F3;
        color: white;
        font-size: 0.8em;
        margin-left: 10px;
    }
    .review-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #FF9800;
        color: white;
        font-size: 0.8em;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)


def get_documents():
    try:
        response = httpx.get(f"{API_URL}/documents", timeout=5.0)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception as e:
        st.sidebar.error(f"Failed to connect to backend: {e}")
    return []


def ask_question_page():
    st.title("🔍 Ask a Question")
    st.write("Ask a question in English, Hindi, or Marathi. The system will retrieve relevant information and provide a cited answer.")
    
    question = st.text_input("Enter your question:", placeholder="e.g., What is machine learning?")
    
    if st.button("Submit", type="primary") and question:
        with st.spinner("Searching and generating answer..."):
            try:
                response = httpx.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    citations = data["citations"]
                    lang = data.get("language", "en")
                    confidence = data.get("confidence", 0.0)
                    
                    # Display Answer
                    st.markdown("### Answer")
                    st.markdown(f'<div style="background-color: #1E2329; padding: 20px; border-radius: 10px;">{answer}</div>', unsafe_allow_html=True)
                    
                    # Metadata row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Detected Language:** <span class='language-badge'>{lang.upper()}</span>", unsafe_allow_html=True)
                    with col2:
                        color = "#4CAF50" if confidence >= 0.7 else "#FFC107" if confidence >= 0.4 else "#FF5252"
                        st.markdown(f"**Confidence:** <span style='color: {color}; font-weight: bold;'>{confidence:.0%}</span>", unsafe_allow_html=True)
                    with col3:
                        if confidence < 0.6 and citations:
                            st.markdown("<span class='review-badge'>⚠️ Human Review Recommended</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Citations
                    st.markdown("### 📑 Sources & Citations")
                    if citations:
                        for i, cit in enumerate(citations, 1):
                            with st.expander(f"Source {i}: {cit['source']} (Page {cit['page']})"):
                                st.markdown(f"**Relevance Score:** `{cit.get('relevance_score', 0):.4f}`")
                                st.markdown("**Snippet:**")
                                st.info(cit['snippet'])
                    else:
                        st.warning("No specific citations were used to generate this answer.")
                        
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to process request: {e}")


def contradiction_page():
    st.title("⚖️ Contradiction Detection")
    st.write("Compare two documents to check if they contradict each other on a specific topic.")
    
    docs = get_documents()
    if not docs:
        st.warning("No documents found. Please ingest documents first.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        doc1 = st.selectbox("Select Document 1", options=docs, key="doc1")
    with col2:
        doc2 = st.selectbox("Select Document 2", options=docs, key="doc2")
        
    topic = st.text_input("Topic to compare:", placeholder="e.g., Carbon emission statistics")
    
    if st.button("Check for Contradictions") and topic:
        if doc1 == doc2:
            st.error("Please select two different documents.")
            return
            
        with st.spinner("Analyzing documents..."):
            try:
                response = httpx.post(
                    f"{API_URL}/contradict",
                    json={"doc1": doc1, "doc2": doc2, "topic": topic},
                    timeout=45.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    conflict = data["conflict"]
                    reasoning = data["reasoning"]
                    
                    st.markdown("---")
                    if conflict:
                        st.error("🚨 Contradiction Detected!")
                        st.markdown(f"**Reasoning:**\n\n{reasoning}")
                    else:
                        st.success("✅ No Contradictions Found.")
                        st.markdown(f"**Reasoning:**\n\n{reasoning}")
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to process request: {e}")


def about_page():
    st.title("ℹ️ About the System")
    st.markdown("""
    ### Multilingual Citation-Based RAG
    
    This system implements a production-style Retrieval-Augmented Generation pipeline with:
    
    - **Multilingual Support:** English, Hindi, and Marathi query processing.
    - **Citation Tracking:** Strict hallucination prevention with exact source mapping.
    - **Contradiction Detection:** Cross-document fact-checking via LLM analysis.
    
    #### Architecture
    1. Documents are loaded and split using RecursiveCharacterTextSplitter.
    2. Embeddings are generated via `sentence-transformers` and stored in ChromaDB.
    3. Queries are translated (if needed), retrieved, and optionally reranked.
    4. Answers are generated using Gemini/Groq with strict grounding prompts.
    """)


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Ask Question", "Contradiction Check", "About"])

if page == "Ask Question":
    ask_question_page()
elif page == "Contradiction Check":
    contradiction_page()
else:
    about_page()
