import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from augmentation import build_sources
from generation import generate_answer
from ingest import ingest_manual
from retrieval import get_available_models, retrieve, setup_hybrid_retriever

app = FastAPI(title="DriveAssist-AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MANUALS_DIR = Path("data/owners_manual")
MANUALS_DIR.mkdir(parents=True, exist_ok=True)

_retriever_cache: dict = {}
_ingest_jobs: dict = {}  # job_id -> queue.Queue


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


@app.post("/upload")
async def upload_manual(file: UploadFile = File(...)):
    """Accept a PDF upload and save it to the manuals directory."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = MANUALS_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)
    return {"filename": file.filename}


@app.post("/ingest/{filename}")
def start_ingest(filename: str):
    """Start background ingestion for an uploaded manual. Returns a job_id for SSE progress."""
    filepath = MANUALS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found. Upload it first.")

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    _ingest_jobs[job_id] = q

    def run():
        try:
            def on_progress(step, message, percent):
                q.put({"type": "progress", "step": step, "message": message, "percent": percent})

            ingest_manual(str(filepath), on_progress)
            _retriever_cache.clear()
            q.put({"type": "done", "success": True, "message": "Manual is ready. You can now select it and chat."})
        except Exception as e:
            q.put({"type": "done", "success": False, "message": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/ingest/progress/{job_id}")
def ingest_progress(job_id: str):
    """SSE stream of progress updates for a running ingest job."""
    if job_id not in _ingest_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    q = _ingest_jobs[job_id]

    def event_stream():
        while True:
            try:
                msg = q.get(timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") == "done":
                    _ingest_jobs.pop(job_id, None)
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/models/{filename}")
def delete_model(filename: str):
    """Remove a manual's PDF file and all its ChromaDB entries."""
    filepath = MANUALS_DIR / filename

    # Delete embeddings from ChromaDB
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="documind_data")
        existing = collection.get(where={"source": str(filepath)})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Delete the PDF file
    if filepath.exists():
        filepath.unlink()
    elif not filepath.exists():
        raise HTTPException(status_code=404, detail="Manual not found.")

    _retriever_cache.clear()
    return {"deleted": filename}
