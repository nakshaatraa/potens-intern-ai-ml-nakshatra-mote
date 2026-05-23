"""FastAPI backend using local TF-IDF RAG engine."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os, glob

app = FastAPI(title="Multilingual RAG API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

from rag_local import generate_answer, detect_contradictions

class AskRequest(BaseModel):
    question: str = Field(..., description="The question to ask")

class Citation(BaseModel):
    source: str; page: int; chunk_id: int; snippet: str; relevance_score: float

class AskResponse(BaseModel):
    answer: str; citations: list[Citation]; language: str; confidence: float

class ContradictRequest(BaseModel):
    doc1: str; doc2: str; topic: str

class ContradictResponse(BaseModel):
    conflict: bool; reasoning: str

@app.get("/health")
def health(): return {"status": "healthy"}

@app.get("/documents")
def list_docs():
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    if not os.path.exists(docs_dir): return {"documents": []}
    pdfs = glob.glob(os.path.join(docs_dir, "*.pdf"))
    return {"documents": [os.path.basename(p) for p in sorted(pdfs)]}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        r = generate_answer(request.question)
        return AskResponse(answer=r["answer"], citations=r["citations"],
                           language=r.get("language","en"), confidence=r.get("confidence",0.0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/contradict", response_model=ContradictResponse)
def contradict(request: ContradictRequest):
    try:
        r = detect_contradictions(request.doc1, request.doc2, request.topic)
        return ContradictResponse(conflict=r["conflict"], reasoning=r["reasoning"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
