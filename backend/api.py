"""
FastAPI backend for the multilingual RAG system.
Provides REST endpoints for question answering and contradiction detection.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from rag import generate_answer, detect_contradictions
import os
import glob

app = FastAPI(
    title="Multilingual RAG API",
    description="API for document-based question answering with citations.",
    version="1.0.0",
)

# Enable CORS for the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., description="The question to ask based on the documents")


class Citation(BaseModel):
    source: str
    page: int
    chunk_id: int
    snippet: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    language: str
    confidence: float


class ContradictRequest(BaseModel):
    doc1: str = Field(..., description="Filename of the first document")
    doc2: str = Field(..., description="Filename of the second document")
    topic: str = Field(..., description="The topic to compare between the documents")


class ContradictResponse(BaseModel):
    conflict: bool
    reasoning: str


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@app.get("/documents")
def list_documents():
    """List all ingested PDF documents."""
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    if not os.path.exists(docs_dir):
        return {"documents": []}
    
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    return {"documents": [os.path.basename(p) for p in sorted(pdf_files)]}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Process a question through the RAG pipeline.
    Returns the answer, citations, detected language, and confidence score.
    """
    try:
        result = generate_answer(request.question)
        return AskResponse(
            answer=result["answer"],
            citations=result["citations"],
            language=result.get("language", "en"),
            confidence=result.get("confidence", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/contradict", response_model=ContradictResponse)
def check_contradiction(request: ContradictRequest):
    """
    Check for contradictions between two documents on a specific topic.
    """
    try:
        result = detect_contradictions(request.doc1, request.doc2, request.topic)
        return ContradictResponse(
            conflict=result["conflict"],
            reasoning=result["reasoning"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
