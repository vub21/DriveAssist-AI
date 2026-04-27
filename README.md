# DriveAssist AI

A RAG-powered chatbot that lets you ask plain-English questions about any vehicle and get accurate, cited answers directly from the owner's manual.

---

## Problem

Owner's manuals are hundreds of pages long. Finding the right section — especially under stress (warning light just came on, roadside situation) — takes too long. Generic web searches return generic answers, not the answer for your specific vehicle trim and year.

## What It Does

DriveAssist AI ingests one or more PDF owner's manuals, indexes them, and exposes a chat interface. Ask "What does the tire pressure warning light mean?" and the app retrieves the exact relevant passages from your manual, sends them to GPT-4o, and returns a precise answer with page-number citations.

---

## Architecture

```
PDF Manuals
    │
    ▼
ingest.py          — Chunks PDFs, generates embeddings, persists to ChromaDB
    │
    ▼
retrieval.py       — Hybrid retrieval: BM25 (keyword) + vector similarity (EnsembleRetriever)
    │
    ▼
augmentation.py    — Builds structured prompt with retrieved context
    │
    ▼
generation.py      — GPT-4o answers the question, cites sources
    │
    ▼
api.py (FastAPI)   — /models and /chat endpoints, retriever caching
    │
    ▼
frontend/ (React)  — Chat UI with model selector, typing indicator, source chips
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent) |
| Keyword Search | BM25 via `rank-bm25` |
| Retrieval | LangChain `EnsembleRetriever` (hybrid) |
| LLM | OpenAI GPT-4o |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite |

---

## Setup

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/vub21/DriveAssist-AI.git
cd DriveAssist-AI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### 3. Add owner's manuals

Drop any vehicle owner's manual PDFs into `data/owners_manual/`.

### 4. Ingest the manuals

```bash
python ingest.py
```

This chunks the PDFs, creates embeddings, and saves them to a local ChromaDB database. Run once per new manual.

### 5. Start the backend

```bash
uvicorn api:app --reload
```

API runs at `http://localhost:8000`. Endpoints:
- `GET /models` — lists all ingested manuals
- `POST /chat` — `{ "query": "...", "model": "filename.pdf" }`

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

App runs at `http://localhost:5173`.

---

## How It Works

1. **Hybrid Retrieval** — Every query hits both a BM25 keyword index and a vector similarity index. Results are merged with equal weighting (`EnsembleRetriever`). This outperforms pure vector search on exact terminology (part names, warning codes) while still handling paraphrased questions.

2. **Per-Manual Filtering** — When a specific manual is selected, the retriever scopes both BM25 and vector search to that document only, preventing cross-manual contamination.

3. **Source Citation** — The prompt instructs GPT-4o to cite sources using `[1]`, `[2]` notation. The API returns source metadata (filename + page number) alongside the answer, which the frontend renders as chips below each response.

4. **Retriever Caching** — The FastAPI layer caches initialized retrievers by model key, so the BM25 index is only built once per session rather than on every request.
