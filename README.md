# Real Estate Document Intelligence API

A high-performance RAG (Retrieval-Augmented Generation) system for querying real estate PDFs using natural language. Built with a two-stage retrieval pipeline: FAISS bi-encoder for fast candidate retrieval, followed by a cross-encoder reranker for precision. Backed by Supabase PostgreSQL for metadata storage and Upstash Redis for query caching.

---

## Demo

[Watch the demo video](https://drive.google.com/file/d/18JSkCblbfuOBub2ffc8HIBFCBAp5ju-Y/view?usp=drivesdk)

---

## Project Structure

```
real-estate-rag/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI routes
│   ├── pipeline.py    # Two-stage RAG logic (FAISS + CrossEncoder)
│   ├── models.py      # Pydantic request/response schemas
│   ├── config.py      # All tunables (chunk size, models, thresholds)
│   ├── db.py          # Supabase PostgreSQL metadata registry
│   └── cache.py       # Upstash Redis query cache
├── frontend/
│   └── index.html     # Single-file UI (no framework, pure HTML/CSS/JS)
├── uploads/           # Temp storage for incoming PDFs (auto-cleaned)
├── indexes/
│   └── master/        # Single master FAISS index — all PDFs merged here
├── eval.py            # Comprehensive evaluation script
├── eval_results.json  # Latest evaluation results
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone and enter the repository
cd real-estate-rag

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Fill in your Supabase and Upstash credentials (optional — falls back gracefully)

# 5. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On first run, two models download automatically:
- `all-MiniLM-L6-v2` (~80MB) — bi-encoder for FAISS indexing
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB) — reranker for precision

Visit `http://localhost:8000` for the UI or `http://localhost:8000/docs` for the interactive API docs.

---

## Environment Variables

```bash
# Supabase PostgreSQL — document metadata registry
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-region.pooler.supabase.com:5432/postgres

# Upstash Redis — query result cache
UPSTASH_REDIS_REST_URL=https://xxxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-token

# Cache TTL in seconds (default: 1 hour)
CACHE_TTL_SECONDS=3600
```

Both are optional — the system falls back to `registry.json` and disables caching if credentials are not set.

---

## Architecture

```
PDF Upload
  │
  ├─► PyMuPDF extraction → text cleaning → chunking (400 chars, 60 overlap)
  ├─► Bi-encoder (all-MiniLM-L6-v2) embeds chunks
  ├─► FAISS IndexFlatIP (cosine similarity) index built
  ├─► Merged into single master index (all PDFs searchable together)
  └─► Metadata saved to Supabase PostgreSQL

Query
  │
  ├─► Check Upstash Redis cache → return instantly if hit (~41ms)
  │
  ├─► Stage 1 — FAISS bi-encoder search (~9ms)
  │     Embed query, cosine search, score threshold filter (≥ 0.3)
  │     Fetch top-20 candidates
  │
  └─► Stage 2 — CrossEncoder reranking (~209ms)
        Reads (query + chunk) together, understands full intent
        Returns true top-K sorted by relevance score
        Result cached in Redis for future identical queries
```

**Why two stages?**
A bi-encoder embeds query and document independently — fast but imprecise. In early testing, pure bi-encoder retrieval returned semantically similar but factually wrong chunks (commercial property descriptions for residential bedroom queries). A cross-encoder reads query and chunk together, understanding intent vs. content. This improved Top-1 accuracy from ~55% to 80% and Top-3 from ~70% to 100%.

**Why a single master index?**
Rather than maintaining per-document indexes and requiring users to track `index_id` values, every uploaded PDF is merged into one master FAISS index. Queries always search across all documents automatically, with source attribution (filename + page) on every result.

---

## API

### Upload a PDF
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@property.pdf"
```
```json
{
  "filename": "E128-Skyvilla-Document.pdf",
  "total_pages": 10,
  "total_chunks": 147,
  "message": "PDF indexed. Use /query/all to search across all documents."
}
```

### Upload multiple PDFs
```bash
curl -X POST http://localhost:8000/collection/create \
  -F "files=@property1.pdf" \
  -F "files=@property2.pdf"
```

### Query all documents
```bash
curl -X POST http://localhost:8000/query/all \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the carpet area of Sky Villa 1?", "top_k": 5}'
```
```json
{
  "question": "What is the carpet area of Sky Villa 1?",
  "results": [
    {
      "content": "Sky Villa 1 Level 1 Carpet area: 5789 sq. ft./ 537.81 sq. m...",
      "filename": "E128-Skyvilla-Document.pdf",
      "page_number": 3,
      "score": 0.4071,
      "rerank_score": 2.7723
    }
  ],
  "stage1_latency_ms": 9.1,
  "stage2_latency_ms": 209.0,
  "total_latency_ms": 218.1,
  "cached": false
}
```

### Health check
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "active_indexes": 1,
  "master_index_ready": true,
  "cache": { "enabled": true, "ttl_seconds": 3600 }
}
```

---

## Success Metrics

### Retrieval Quality

Evaluated against 20 questions spanning two real estate PDFs:
- `E128-Skyvilla-Document.pdf` — Estate 128 Sky Villas, Noida
- `222-Rajpur-Document.pdf` — 222 Rajpur, Dehradun

| Metric | Score | Target | Status |
|---|---|---|---|
| **Recall@1 (Top-1 Accuracy)** | **80.0% (16/20)** | ≥ 75% | ✓ |
| **Recall@3 (Top-3 Accuracy)** | **100.0% (20/20)** | ≥ 90% | ✓ |
| **Recall@5** | **100.0% (20/20)** | — | ✓ |
| **MRR** | **0.90** | closer to 1.0 | ✓ |
| **nDCG@5** | **0.9169** | closer to 1.0 | ✓ |
| **Entity Coverage Score** | **0.525** | — | — |
| **Paraphrase Robustness** | **80.0%** | — | — |
| **Hallucination Rate** | **40.0%** | lower = better | — |
| **False Positive Rate** | **40.0%** | lower = better | — |

**Note on Hallucination / False Positive Rate:** This is measured at the retrieval level, no LLM generation exists in the current system. A "false positive" means the reranker returned a positive score for a question whose answer does not exist in the documents (e.g. "What is the price per sq ft?"). The 40% rate reflects that the reranker is not trained to abstain, it tries to rank the best available chunk, even if no truly relevant chunk exists. This is a known limitation of retrieval-only systems and would be addressed by adding a confidence threshold gate before returning results.

#### Per-question breakdown

| ID | Question | Rank | nDCG | Entity Cov |
|---|---|:---:|:---:|:---:|
| Q01 | Carpet area of Sky Villa 1 vs Sky Villa 2 | 1 | 1.00 | 0.50 |
| Q02 | Does Estate 128 include pool and gym? | 1 | 1.00 | 0.50 |
| Q03 | Separate entrances for guest rooms? | 1 | 1.00 | 1.00 |
| Q04 | Double-height living room features | 1 | 1.00 | 1.00 |
| Q05 | Width of wraparound balconies | 1 | 1.00 | 1.00 |
| Q06 | RERA registration number | 1 | 1.00 | 1.00 |
| Q07 | Private elevators in Estate 128 | 2 | 0.65 | 0.00 |
| Q08 | Air conditioning system | 1 | 0.85 | 0.00 |
| Q09 | Built-up and carpet area of Townhouse | 2 | 0.63 | 0.00 |
| Q10 | Courtyard vs Forest Villas unit count | 1 | 1.00 | 0.50 |
| Q11 | Plot size range for Townhouses | 1 | 1.00 | 0.50 |
| Q12 | Forest Villas landscape connection | 1 | 0.98 | 0.50 |
| Q13 | Sky court or atrium in townhouses | 2 | 0.65 | 0.00 |
| Q14 | Service areas separated from private | 1 | 1.00 | 1.00 |
| Q15 | Private elevators in Forest Villas | 1 | 1.00 | 1.00 |
| Q16 | Unique botanical feature | 1 | 0.98 | 0.00 |
| Q17 | Natural views at 222 Rajpur | 1 | 0.96 | 1.00 |
| Q18 | UKRERA registration number | 1 | 1.00 | 0.00 |
| Q19 | Seismic zone adherence | 2 | 0.63 | 0.00 |
| Q20 | Car recognition security measures | 1 | 1.00 | 1.00 |

---

### Caching Strategy

Query results are cached in Upstash Redis with a 1-hour TTL. The cache key is `result:{hash(question)}:{index_id}` — identical questions served from cache instantly.

| Metric | Value |
|---|---|
| Cold query average | 220.7ms |
| Cached query average | 41.0ms |
| **Latency reduction** | **81.4%** |

This directly exceeds the 50% reduction target. The 41ms warm latency includes HTTP round-trip to Upstash Redis plus response serialization.

**Can we cache embeddings?** Yes, but the benefit during indexing is negligible — chunks are always new documents so cache hit rate would be near zero. At query time, embedding the question takes ~5ms which is already small relative to reranking (~209ms). Result caching at the full response level gives far better ROI.

---

### Stage-wise Latency Breakdown

Measured across 20 evaluation queries on CPU:

| Stage | Avg Time | % of Total |
|---|---|---|
| Stage 1 — FAISS retrieval | 9.1ms | 4.2% |
| Stage 2 — Cross-encoder rerank | 209.0ms | 95.8% |
| **Total (cold)** | **218.2ms** | — |
| **Total (cached)** | **~41ms** | — |

| Percentile | Latency |
|---|---|
| Average | 218.2ms |
| P95 | 269.1ms |
| P99 | 279.8ms |

All queries within the 2 second target. Generation time is not applicable — the current system returns raw retrieved chunks without LLM synthesis.

**Is reranking worth the added latency?**
Yes. Top-3 accuracy of 100% with reranking versus ~70% without it justifies the ~209ms cost. The cross-encoder's ability to understand query intent versus chunk content is what separates relevant from irrelevant results for specific factual queries.

---

## System Behavior

### What happens as PDFs grow larger?

| Component | Behavior |
|---|---|
| Indexing time | Scales linearly ~0.5s per page on CPU |
| Query latency | Constant — FAISS search is O(n), fast up to ~500k chunks. Reranker scales with `CANDIDATE_K` (fixed at 20), not index size |
| RAM usage | ~1.5KB per chunk (384-dim float32). 200-page PDF ≈ 3MB. Not a concern until thousands of PDFs |

Large PDFs are the primary stress test. Beyond indexing time, they introduce memory pressure since all chunk vectors must fit in RAM alongside the FAISS index. The current practical limit is 50 pages per document, beyond which a dedicated vector database with disk-backed storage is more appropriate.

### What would break first in production?

**In-memory Python dict for index storage** is the most critical limitation. All FAISS indexes live in `self._indexes: Dict[str, FAISS]` — a plain dict in the server process. Running `uvicorn --workers 4` would break query routing since each worker has its own isolated dict. Fix: migrate to Qdrant, Weaviate, or Pinecone for shared persistent vector storage.

**Synchronous indexing blocks uploads.** Large PDFs hold the HTTP connection open for 10–30s. Fix: `BackgroundTasks` or a job queue (Celery/ARQ) — return a job ID immediately and poll for completion.

**FAISS flat index at scale.** `IndexFlatIP` does exact exhaustive search. Fine up to ~500k vectors, but beyond that switch to approximate search (`IndexIVFFlat` or `IndexHNSWFlat`) for sub-linear query time.

**No authentication.** Any client can upload arbitrary PDFs or flood the query endpoint. Fix: API key middleware, rate limiting, file content validation.

### Where are the bottlenecks?

| Step | Avg Time | Notes |
|---|---|---|
| PDF extraction (PyMuPDF) | ~50ms/page | C-based, already optimal |
| Embedding generation | ~2–5s total | Main indexing bottleneck. GPU = 10–20× faster |
| FAISS index build | ~100ms | Negligible |
| Stage 1: FAISS query | **9.1ms** | Constant regardless of index size |
| Stage 2: CrossEncoder | **209ms** | Dominates query latency. Scales with `CANDIDATE_K` not index size |

The cross-encoder dominates query latency at ~209ms.

The in-memory Python dict is the most critical production limitation — indexes are not shared across multiple workers, RAM grows unbounded, and there is no eviction policy.

---

## Configuration

All tunables in `app/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Lightweight, fast, good general-purpose baseline |
| `RERANKER_MODEL` | ms-marco-MiniLM-L-6-v2 | Trained on MS MARCO, strong factual Q&A precision |
| `CHUNK_SIZE` | 400 chars | Tuned up from 200 — keeps related sentences together |
| `CHUNK_OVERLAP` | 60 chars | Prevents boundary cuts between chunks |
| `CANDIDATE_K` | 20 | Stage 1 fetch size — wider net for reranker |
| `SCORE_THRESHOLD` | 0.3 | Min cosine similarity to pass Stage 1 |
| `DEFAULT_TOP_K` | 5 | Results returned per query |
| `MAX_FILE_SIZE_MB` | 50 | Upload size limit |

---

## Running the Evaluation

```bash
# Install eval dependencies
pip install numpy requests

# Upload PDFs and run full evaluation
python eval.py --upload

# Evaluate only (PDFs already indexed)
python eval.py

# Verbose — shows retrieved chunks per question
python eval.py --verbose

# Against a remote server
API_BASE_URL=https://your-deployed-api.com python eval.py
```

Results are saved to `eval_results.json`. The script measures:
- Recall@1, Recall@3, Recall@5, MRR, nDCG@5
- Entity Coverage Score
- Paraphrase Robustness Score
- Hallucination Rate and False Positive Rate (negative query test)
- Cold vs cached latency with % improvement
- Stage-wise latency breakdown with P95 and P99

---

## Challenges Addressed

**Retrieval precision — semantic mismatch**
Pure bi-encoder retrieval returned semantically similar but factually wrong chunks. Solved by switching from L2 to cosine similarity (`IndexFlatIP`) and adding cross-encoder reranking, improving Top-1 from ~55% to 80%.

**Chunk size tuning**
Initial 200-char chunks fragmented specific facts like "Carpet area: 5789 sq. ft." across boundaries. Increasing to 400 chars with 60-char overlap kept related sentences together and improved accuracy significantly.

**Large PDF handling**
Indexing time scales linearly with page count on CPU. Pages with less than 50 characters after cleaning (cover pages, image-only pages) are skipped to reduce noise. Practical limit is 50 pages per document for acceptable CPU indexing time.

**Duplicate chunks**
Real estate brochures repeat layout elements across pages. Content-hash deduplication removes identical chunks before reranking.

**Score threshold calibration**
Default L2 distance metrics returned irrelevant results with misleadingly high scores. Switching to inner product with normalized embeddings gave meaningful cosine similarity scores, and a 0.3 threshold filters genuinely irrelevant candidates before reranking.

---

## Roadmap

**LLM Response Generation**
The current system returns raw retrieved chunks — the user interprets them manually. The natural next step is a `/answer` endpoint that passes the top-K reranked chunks as context to an LLM (Claude, GPT-4, or a local model via Ollama), returning a synthesised cited answer. This converts the system from a search engine into a true document Q&A assistant without changing the retrieval pipeline.

**PostgreSQL Metadata Storage**
Currently using Supabase PostgreSQL for document metadata via SQLAlchemy. A production deployment would extend this to store query logs, usage analytics, and per-user document access control.

**Async Indexing**
Replace synchronous upload processing with a background job queue (Celery/ARQ) — return a job ID immediately, poll for completion. Eliminates the 10–30s HTTP connection hold on large PDFs.

**Hybrid Search**
Combine FAISS semantic search with BM25 keyword search, then rerank the merged candidates. Improves recall for exact terms like RERA numbers and specific sq ft values.
