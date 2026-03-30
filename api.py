from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retrieval import get_available_models, setup_hybrid_retriever, retrieve
from generation import generate_answer
from augmentation import build_sources

app = FastAPI(title="DriveAssist-AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache retrievers so we don't rebuild indexes on every request
_retriever_cache: dict = {}


def get_retriever(model: Optional[str]):
    key = model or "__all__"
    if key not in _retriever_cache:
        _retriever_cache[key] = setup_hybrid_retriever(model)
    return _retriever_cache[key]


class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None


@app.get("/models")
def list_models():
    """Return all vehicle manuals currently stored in the database."""
    return {"models": get_available_models()}


@app.post("/chat")
def chat(req: ChatRequest):
    """Run a RAG query and return the answer with source citations."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    retriever = get_retriever(req.model)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Database not ready. Run ingest.py first.")

    docs = retrieve(req.query, retriever)
    answer = generate_answer(req.query, docs)
    sources = build_sources(docs)

    return {"answer": answer, "sources": sources}
