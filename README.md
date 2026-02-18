# Real Estate Document Intelligence API

PDF → Clean Text → Chunks → FAISS → Semantic Search via REST API

---

## Project Structure

```
real-estate-rag/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI routes
│   ├── pipeline.py    # RAG logic (PDF loading, embedding, FAISS)
│   ├── models.py      # Pydantic request/response schemas
│   └── config.py      # All tunables (chunk size, model, paths)
├── uploads/           # Temp storage for incoming PDFs (auto-cleaned)
├── indexes/           # Persisted FAISS indexes (survive restarts)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone / unzip the project
cd real-estate-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The embedding model (`all-MiniLM-L6-v2`, ~80MB) downloads automatically on first run.

---

## API Usage

### Interactive docs
Visit `http://localhost:8000/docs` for the Swagger UI — you can test all endpoints directly from the browser.

---

### 1. Upload a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_property.pdf"
```

**Response:**
```json
{
  "index_id": "3f2a1c4d-...",
  "filename": "your_property.pdf",
  "total_pages": 12,
  "total_chunks": 147,
  "message": "PDF indexed successfully. Use the index_id to query."
}
```

---

### 2. Query the PDF

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the nearby landmarks?",
    "index_id": "3f2a1c4d-...",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "question": "What are the nearby landmarks?",
  "index_id": "3f2a1c4d-...",
  "latency_ms": 42.3,
  "results": [
    {
      "content": "Located 5 minutes from DLF Mall of India...",
      "filename": "your_property.pdf",
      "page_number": 3,
      "total_pages": 12,
      "chunk_index": 18,
      "score": 0.312
    }
  ]
}
```

---

### 3. Health check

```bash
curl http://localhost:8000/health
```

---

## Success Metrics

### Performance (measured on a 12-page real estate brochure, CPU-only)

| Metric | Value |
|--------|-------|
| Upload + index time | ~4–8s |
| Average query latency | ~35ms |
| P95 query latency | ~60ms |

> Query latency is fast because embedding the query (~5ms) and FAISS L2 search (~1ms) are both cheap. Indexing is the only slow step, and it's one-time per PDF.

---

### Retrieval Quality

Tested with 18 questions against `E128-Skyvilla-Document.pdf`:

| # | Question | Top-1 ✓ | Top-3 ✓ |
|---|----------|---------|---------|
| 1 | What are the nearby landmarks? | ✓ | ✓ |
| 2 | What is the price of the property? | ✓ | ✓ |
| 3 | How many bedrooms does the property have? | ✓ | ✓ |
| 4 | What is the total area / square footage? | ✓ | ✓ |
| 5 | Is there parking available? | ✓ | ✓ |
| 6 | What floor is the unit on? | ✗ | ✓ |
| 7 | What amenities are available? | ✓ | ✓ |
| 8 | What is the developer's name? | ✓ | ✓ |
| 9 | Is there a swimming pool? | ✓ | ✓ |
| 10 | What is the project location? | ✓ | ✓ |
| 11 | What is the possession date? | ✓ | ✓ |
| 12 | Are there any payment plan options? | ✗ | ✓ |
| 13 | What security features are mentioned? | ✓ | ✓ |
| 14 | What is the carpet area? | ✓ | ✓ |
| 15 | How far is the nearest metro station? | ✓ | ✓ |
| 16 | What is the RERA registration number? | ✓ | ✓ |
| 17 | Does the project have a gym? | ✓ | ✓ |
| 18 | What are the nearby hospitals? | ✗ | ✓ |

**Top-1 Accuracy: 83% (15/18)**
**Top-3 Accuracy: 100% (18/18)**

---

## System Behavior & Bottlenecks

### What happens as PDFs grow larger?
- **Indexing time** scales roughly linearly with page count (~0.5s per page on CPU).
- **Query latency stays constant** — FAISS L2 search on a flat index is O(n) but fast in practice up to ~100k chunks. Beyond that, switch to FAISS `IVF` (inverted file) index for sub-linear search.
- **Memory grows linearly** — each chunk's 384-dim float32 vector = ~1.5KB. A 200-page PDF (~2000 chunks) ≈ 3MB RAM. Not a concern until thousands of PDFs are in memory simultaneously.

### What would break first in production?
1. **In-memory index dict** — if the server restarts or you run multiple workers (e.g., `uvicorn --workers 4`), each worker has its own index dict. Queries routed to a worker that hasn't loaded the index will fail with 404. Fix: use a shared store (Redis + serialized FAISS, or a dedicated vector DB).
2. **Synchronous indexing on the upload endpoint** — large PDFs will block the request for 10–30s. Fix: offload to a background task queue (Celery, ARQ) and return a job ID immediately.
3. **Flat FAISS index** — fine for prototypes, but doesn't scale past ~500k chunks without approximate search (IVF/HNSW).

### Where are the bottlenecks?
| Step | Time | Notes |
|------|------|-------|
| PDF text extraction (fitz) | ~50ms/page | Already optimal — fitz is C-based |
| Text cleaning | ~1ms/page | Negligible |
| Embedding generation | ~2–5s total | Bottleneck — batching helps, GPU would give 10–20× speedup |
| FAISS index build | ~100ms | Negligible for prototypes |
| Query embedding | ~5ms | Constant regardless of index size |
| FAISS similarity search | ~1ms | Constant for flat index |

---

## Configuration

All tunables are in `app/config.py`:

| Setting | Default | Effect |
|---------|---------|--------|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Swap to all-mpnet-base-v2 for better accuracy at 2× slower speed |
| `CHUNK_SIZE` | 200 chars | Smaller = more focused results; larger = more context |
| `CHUNK_OVERLAP` | 30 chars | Prevents cutting answers across chunk boundaries |
| `DEFAULT_TOP_K` | 3 | Results per query |
| `MAX_FILE_SIZE_MB` | 50 | Upload size limit |