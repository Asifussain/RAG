# Real Estate Document Intelligence API

A lightweight, high-performance RAG (Retrieval-Augmented Generation) system for querying real estate PDFs using natural language. Built with a two-stage retrieval pipeline: FAISS bi-encoder for fast candidate retrieval, followed by a cross-encoder reranker for precision.

---

## Project Structure

```
real-estate-rag/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI routes
│   ├── pipeline.py    # Two-stage RAG logic (FAISS + CrossEncoder)
│   ├── models.py      # Pydantic request/response schemas
│   └── config.py      # All tunables (chunk size, models, thresholds)
├── frontend/
│   └── index.html     # Single-file UI (no framework, pure HTML/CSS/JS)
├── uploads/           # Temp storage for incoming PDFs (auto-cleaned)
├── indexes/           # Persisted FAISS indexes (survive restarts)
│   └── master/        # Master index — all uploaded PDFs merged here
├── eval.py            # Evaluation script (latency + accuracy metrics)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repository
cd real-estate-rag

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On first run, two models download automatically:
- `all-MiniLM-L6-v2` (~80MB) — bi-encoder for FAISS indexing
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB) — reranker for precision

Visit `http://localhost:8000` for the UI or `http://localhost:8000/docs` for the Swagger API.

---

## How It Works — Two-Stage Retrieval

```
PDF Upload
  │
  ├─► PyMuPDF extraction → text cleaning → chunking (400 chars)
  ├─► Bi-encoder embeds chunks → FAISS IndexFlatIP (cosine similarity)
  ├─► Per-document index saved to disk
  └─► Merged into master index (all PDFs searchable together)

Query
  │
  ├─► Stage 1: FAISS fetches top-20 candidates (~8ms)
  │     Bi-encoder embeds query, cosine search, score threshold filter
  │
  └─► Stage 2: CrossEncoder reranks candidates (~185ms)
        Reads (query + chunk) together, understands intent vs. content
        Returns true top-K in correct relevance order
```

**Why two stages?**
A bi-encoder embeds query and document independently, fast but imprecise based on evaluation, so shifted to two stages using cross-encoder. A cross-encoder reads them together and understands semantic intent, improving ranking accuracy for specific factual queries.

## Success Metrics

### Performance
Measured across 20 evaluation queries on CPU (no GPU):

| Metric | Value |
|---|---|
| Average total latency | **192.4ms** |
| P95 latency | **245.2ms** |
| P99 latency | 263.7ms |
| Min / Max | 102.5ms / 268.3ms |
| Avg Stage 1 (FAISS) | 7.7ms |
| Avg Stage 2 (Rerank) | 184.7ms |

All queries comfortably within the 2 second target. Stage 2 (cross-encoder reranking) accounts for ~96% of total latency, the expected tradeoff for significantly improved ranking precision.

---

### Retrieval Quality

Evaluated against 20 questions spanning two real estate PDFs:
- `E128-Skyvilla-Document.pdf` — Estate 128 Sky Villas, Noida
- `222-Rajpur-Document.pdf` — 222 Rajpur, Dehradun

| Metric | Score |
|---|---|
| **Top-1 Accuracy** | **75.0% (15/20)** |
| **Top-3 Accuracy** | **95.0% (19/20)** |

#### Evaluation Set

| ID | Question | Top-1 | Top-3 |
|---|---|:---:|:---:|
| Q01 | Carpet area of Sky Villa 1 vs Sky Villa 2 
| Q02 | Does Estate 128 triplex include pool and gym? 
| Q03 | Separate entrances for guest rooms in Estate 128? 
| Q04 | Architectural features of the double-height living room 
| Q05 | How wide are the wraparound balconies at Estate 128? 
| Q06 | Is Estate 128 RERA registered and what is the number? 
| Q07 | Private elevators in Estate 128 Sky Villas? 
| Q08 | Air conditioning system used in Estate 128?
| Q09 | Built-up area and carpet area of a Townhouse in 222 Rajpur 
| Q10 | Courtyard Villas vs Forest Villas unit count
| Q11 | Plot size range for Townhouses in 222 Rajpur 
| Q12 | Forest Villas connection to landscape 
| Q13 | Sky court or atrium in 222 Rajpur townhouses? 
| Q14 | Service areas separated from private spaces in 222 Rajpur 
| Q15 | Private elevators in Forest Villas Dehradun?
| Q16 | Unique botanical feature at 222 Rajpur 
| Q17 | Primary natural views at 222 Rajpur 
| Q18 | UKRERA registration number for 222 Rajpur 
| Q19 | Seismic zone for 222 Rajpur structural design 
| Q20 | Car recognition security + drive time to Jolly Grant Airport

---

## System Behavior

### What happens as PDFs grow larger?

| Component | Behavior |
| Indexing time | Scales linearly ~0.5s per page on CPU |
| Query latency | Stays constant — FAISS search is O(n) but fast up to ~500k chunks |
| RAM usage | ~1.5KB per chunk (384-dim float32). 200-page PDF ≈ 3MB. Not a concern until thousands of PDFs |
| Reranker latency | Scales with CANDIDATE_K (fixed at 20), not index size — stays constant |

### What would break first in production?

**1. In-memory index dict across multiple workers**
Currently using `uvicorn --workers 4` means each worker has its own `self._indexes` dict. A PDF uploaded via worker 1 is invisible to workers 2, 3, 4. Should probably use a shared vector store (Qdrant, Weaviate) or Redis-backed index registry.

**2. Synchronous indexing blocks the upload request**
Large PDFs hold the HTTP connection open for 10–30s. I am thinking of implementing job queue like Celery and redis to handle the job inorder to be reponsive.

**3. FAISS flat index at scale**
Currently it uses `IndexFlatIP` which does exact search in O(n). But for a very high volume of PDFs, we must switch to approximate search (`IndexIVFFlat` which uses clustering and reduces the search space or `IndexHNSWFlat` which is a graph based searching) for sub-linear query time.

**4. No authentication**
Any client can upload arbitrary PDFs or flood the query endpoint. I can think of some sort of Authentication, maybe Google OAuth to make things simple, rate limiting, validation of content being uploaded.

### Where are the bottlenecks?

|Embedding generation is the bottleneck during indexing. Every upload blocks until all chunks are embedded on CPU. Fix: GPU inference or async background task queue (Celery/ARQ).|
|Cross-encoder reranking dominates query latency at ~185ms avg. Three ways to reduce it: switch to the smaller ms-marco-MiniLM-L-2-v2 (~60ms), reduce CANDIDATE_K from 20 to 10, or run on GPU for ~10× speedup.|
|In-memory Python dict for index storage is the most critical production limitation. All FAISS indexes live in self._indexes: Dict[str, FAISS] — a plain Python dict in the server process. This means indexes are not shared across multiple workers (uvicorn --workers 4 would break query routing), RAM grows unbounded as more PDFs are uploaded, and there is no eviction policy. Fix: migrate to a dedicated vector database (Qdrant, Weaviate, Pinecone) which handles persistence, multi-worker access, and memory management natively.|
|FAISS flat index (IndexFlatIP) does exact exhaustive search — correct but O(n) at query time. Acceptable up to ~500k vectors. Beyond that, approximate search indexes (IndexIVFFlat, IndexHNSWFlat) give sub-linear query time with minimal accuracy loss.|
---

## Configuration

All tunables in `app/config.py`:

| Setting | Default | Effect |
|---|---|---|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2|
| `RERANKER_MODEL` | ms-marco-MiniLM-L-6-v2|
| `CHUNK_SIZE` | 400 chars | Larger = more context per chunk, fewer boundary cuts |
| `CHUNK_OVERLAP` | 60 chars | Prevents splitting sentences across chunk boundaries |
| `CANDIDATE_K` | 20 | Stage 1 fetch size. Higher = better recall, slower reranking |
| `SCORE_THRESHOLD` | 0.3 | Min cosine similarity to pass Stage 1. Raise for stricter filtering |
| `DEFAULT_TOP_K` | 5 | Results returned per query |
| `MAX_FILE_SIZE_MB` | 50 | Upload size limit |

---

## Running the Evaluation

```bash

# Upload PDFs and run evaluation
python eval.py --upload

# Run evaluation only (PDFs already indexed)
python eval.py

# Verbose mode — shows retrieved chunks per question
python eval.py --verbose
```

Results are saved to `eval_results.json`.

---

## Challenges Addressed

**Handling large PDFs**
As page count grows, indexing time scales linearly, each page must be extracted, cleaned, chunked, and embedded. On CPU, this means a 50-page document takes approximately 25–30 seconds to index.

Beyond indexing time, large PDFs introduce memory pressure since all chunk vectors must fit in RAM alongside the FAISS index. A 100-page document generates roughly 1,000 chunks at 400-char size, adding about 6MB to the in-memory index — manageable individually but cumulative across many uploads. So the current practical limit for this prototype is 50 pages per document, beyond which a chunked upload strategy or dedicated vector database with disk-backed storage would be more appropriate.

**Balancing chunk size for accuracy and speed**
Initial chunk size of 200 chars caused fragmentation — specific facts like "Carpet area: 5789 sq. ft." were split across chunks, degrading retrieval. Increasing to 400 chars with 60-char overlap kept related sentences together and improved Top-1 accuracy significantly.

**Retrieval precision — semantic mismatch**
Pure bi-encoder retrieval returned semantically similar but factually wrong chunks (e.g., commercial property descriptions for residential bedroom queries). Solved by switching from L2 to cosine similarity (IndexFlatIP) and adding cross-encoder reranking, which reads query and chunk together to understand intent.

**Duplicate chunks**
Real estate brochures repeat layout elements across consecutive pages. Added content-hash deduplication before reranking to prevent the same text appearing multiple times in results.

### Things that i would wish to include in this
**LLM Response Generation**
The current system returns raw retrieved chunks — the user has to read and interpret them manually. The natural next step is adding an LLM generation layer on top of retrieval:

This converts the system from a **search engine** into a true **document Q&A assistant**. The retrieval pipeline stays identical, the LLM only sees the top-K chunks as context, not the entire document. This keeps latency low and prevents hallucination by grounding every answer in retrieved evidence.
