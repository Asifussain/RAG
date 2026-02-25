# Real Estate Document Intelligence API

A production-grade RAG (Retrieval-Augmented Generation) system for querying real estate PDFs using natural language. The pipeline runs two-stage retrieval, FAISS bi-encoder for fast candidate fetch, cross-encoder reranker for precision and synthesises grounded answers via Groq LLM. Backed by Supabase PostgreSQL for metadata and Upstash Redis for query caching.

---

## Demo

[Watch the demo video](https://drive.google.com/file/d/18JSkCblbfuOBub2ffc8HIBFCBAp5ju-Y/view?usp=drivesdk)

---

## Project Structure

```
real-estate-doc-intelligence/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes
│   ├── pipeline.py      # Two-stage RAG logic (FAISS + CrossEncoder)
│   ├── models.py        # Pydantic request/response schemas
│   ├── config.py        # All tunables (chunk size, models, thresholds)
│   ├── db.py            # Supabase PostgreSQL metadata registry
│   ├── cache.py         # Upstash Redis query cache
│   └── llm.py           # Groq LLM generation layer
├── frontend/
│   └── index.html       # Simple UI
├── uploads/             # Temp storage for incoming PDFs
├── indexes/
│   └── master/          # Single master FAISS index — all PDFs merged here
├── eval1.py             # Evaluation — Sections A + B (222 Rajpur + Max Towers)
├── eval2.py             # Evaluation — Sections C + D + E (Max House + Cross-property)
├── eval3.py             # Evaluation — Sections F + G + H (Robustness + Negative + Clarification)
├── merge_eval_results.py # Merges eval1/2/3 results into final report
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
# Fill in your Supabase, Upstash, and Groq credentials

# 5. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On first run, two models download automatically:
- `all-MiniLM-L6-v2` (~80MB) — bi-encoder for FAISS indexing
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB) — reranker for precision

Visit `http://localhost:8000` for the UI or `http://localhost:8000/docs` for the interactive API.

---

## Environment Variables

```bash
# Supabase PostgreSQL — document metadata registry
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-region.pooler.supabase.com:5432/postgres

# Upstash Redis — query result cache
UPSTASH_REDIS_REST_URL=https://xxxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-token

# Groq LLM — answer generation
# Get a free key at console.groq.com
GROQ_API_KEY=your-groq-api-key

# Cache TTL in seconds (default: 1 hour)
CACHE_TTL_SECONDS=3600
```

Supabase and Upstash are optional — the system falls back to `registry.json` and disables caching gracefully. Groq is required for the `/answer` endpoint but `/query/all` retrieval works without it.

---

## Architecture

```
PDF Upload
  │
  ├─► PyMuPDF extraction → text cleaning → chunking (400 chars, 60 overlap)
  ├─► Bi-encoder (all-MiniLM-L6-v2) embeds all chunks in a single batch call
  ├─► FAISS IndexFlatIP (cosine similarity) 
  ├─► Merged into single master index (all PDFs searchable together)
  └─► Metadata saved to Supabase PostgreSQL + local registry.json fallback

Query  (/query/all — retrieval only)
  │
  ├─► Check Upstash Redis cache → return instantly if hit (~35ms)
  ├─► Stage 1 — FAISS bi-encoder search (~13ms)
  │     Embed query, cosine search, score threshold filter (≥ 0.3)
  │     Fetch top-20 candidates
  └─► Stage 2 — CrossEncoder reranking (~198ms)
        Reads (query + chunk) together, understands full semantic intent
        Returns top-K sorted by relevance score, cached in Redis

Answer  (/answer — retrieval + LLM generation)
  │
  ├─► Same Stage 1 + Stage 2 as above
  └─► Stage 3 — Groq LLM synthesis (~727ms avg)
        Top-K chunks passed as grounded context
        llama-3.3-70b-versatile synthesises a cited natural language answer
        Answer cached in Redis separately from raw retrieval results
```

**Why two retrieval stages?** A bi-encoder embeds query and document independently — fast but imprecise. In early testing, pure bi-encoder retrieval returned semantically similar but factually wrong chunks. A cross-encoder reads query and chunk together, understanding intent vs. content, this improved Top-1 accuracy from ~55% to 85% and Top-3 from ~70% to 91%.

**Why a single master index?** I first implemented specific document query as well but this sounded more robust. Every uploaded PDF is merged into one FAISS index. Queries always search across all documents automatically, with source attribution (filename + page) on every result. No `index_id` tracking required from the client.

**Why Groq?** After a bit of research, I decided to go with it as it has fastest inference available via API (~200–700ms), generous free tier (14,400 req/day), and `llama-3.3-70b-versatile` matches GPT-4o-mini quality for factual document Q&A.

---

## API

### Upload a PDF
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@property.pdf"
```
```json
{
  "filename": "MaxTowers-Document.pdf",
  "total_pages": 36,
  "total_chunks": 214,
  "message": "PDF indexed. Use /query/all to search across all documents."
}
```

### Query all documents (retrieval only)
```bash
curl -X POST http://localhost:8000/query/all \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the LEED certification of Max Towers?", "top_k": 5}'
```
```json
{
  "question": "What is the LEED certification of Max Towers?",
  "results": [
    {
      "content": "Max Towers has achieved LEED Platinum Certified...",
      "filename": "MaxTowers-Document.pdf",
      "page_number": 13,
      "score": 0.4812,
      "rerank_score": 3.14
    }
  ],
  "stage1_latency_ms": 11.2,
  "stage2_latency_ms": 196.4,
  "total_latency_ms": 207.6,
  "cached": false
}
```

### Get a natural language answer (retrieval + LLM)
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Which property has LEED Platinum certification?", "top_k": 5}'
```
```json
{
  "question": "Which property has LEED Platinum certification?",
  "answer": "Max Towers has achieved LEED Platinum certification, as stated on page 13 of the Max Towers brochure. In contrast, Max House is LEED Gold certified. Therefore, Max Towers holds the higher certification of the two.",
  "sources": [
    { "filename": "MaxTowers-Document.pdf", "page_number": 13, "rerank_score": 3.14 },
    { "filename": "MaxHouse-Document.pdf",  "page_number": 10, "rerank_score": 1.87 }
  ],
  "retrieval_ms": 207.6,
  "generation_ms": 512.3,
  "total_ms": 719.9,
  "cached": false,
  "model": "llama-3.3-70b-versatile"
}
```

### Clear all documents
```bash
curl -X DELETE http://localhost:8000/documents/all
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

## Evaluation Results

Evaluated against the full 80-question test set across 4 real estate documents:
- `222-rajpur-brochure.pdf` — 222 Rajpur, Dehradun
- `max-towers-brochure.pdf` — Max Towers, Noida
- `max-house-brochure.pdf` — Max House, Okhla

### Retrieval Quality (Sections A–E, 80 questions)

| Metric | Score |
|---|---|
| **Recall@1 (Top-1 Accuracy)** | **85.0%** |
| **Recall@3 (Top-3 Accuracy)** | **91.2%** |
| **Recall@5** | **93.8%** |
| **MRR** | **0.8842** |
| **nDCG@5** | **0.8883** | 
| **Entity Coverage Score** | **0.8063** |

### Robustness & Quality (Sections F–H)

| Metric | Score | Notes |
|---|---|---|
| **Paraphrase Robustness** | **86.7%** | 13/15 paraphrased questions matched the original result |
| **Hallucination Rate** | **10.0%** | 1/10 adversarial questions answered incorrectly |
| **False Positive Rate** | **10.0%** | 1/10 negative queries retrieved a false positive |
| **Clarification Score** | **100.0%** | All 5 ambiguous queries correctly triggered clarification |

The hallucination rate of 10% reflects one edge case where the model attempted an answer despite weak retrieval context. The clarification score of 100% means the system correctly identifies vague questions like "What is the total area?" and asks the user to specify which property rather than guessing.

These scores are conservative by design, a subset of queries hit Groq's free tier rate limits mid-evaluation and were counted as failures rather than excluded. The numbers represent a floor, not a ceiling.

### Caching Strategy

Cache keys are `result:{hash(question)}:{index_id}` with a 1-hour TTL. LLM answers from `/answer` are cached separately so repeated natural language queries also return instantly.

| Metric | Value |
|---|---|
| Cold query average | 229.9ms |
| Cached query average | 35.5ms |
| **Latency reduction** | **84.6%** |

**Can we cache embeddings?** Benchmarked and decided no. During indexing, chunks are always new, cache hit rate is near zero. At query time, embedding takes ~5ms, which is negligible against reranking at ~198ms. Full response caching gives far better ROI.

### Stage-wise Latency Breakdown

| Stage | Avg Time | Share of Total |
|---|---|---|
| Stage 1 — FAISS retrieval | 12.9ms | 1.4% |
| Stage 2 — Cross-encoder rerank | 197.6ms | 21.1% |
| Stage 3 — Groq LLM generation | 727.1ms | 77.5% |
| **End-to-end cold** | **939.8ms** | — |
| **End-to-end cached** | **~35ms** | — |

| Percentile | Retrieval only | Full pipeline (with LLM) |
|---|---|---|
| Average | 210.6ms | 939.8ms |
| P95 | 257.5ms | 1942.7ms |

The P95 spike to ~1.9s is from Groq free tier rate limiting under bulk load (30 RPM). On a paid tier this would flatten to ~600–800ms. Pure retrieval without LLM stays well under 300ms at P95.

**Is reranking worth the added ~198ms?** Yes. Without it, Recall@1 drops to ~70% and Recall@3 to ~85%. The cross-encoder is what catches the difference between chunks that are semantically similar and chunks that actually answer the question.

---

## LLM Integration

The `/answer` endpoint chains two-stage retrieval with Groq LLM synthesis. Retrieved chunks become grounded context — the model is instructed to answer only from what the documents say and cite page numbers.

**System prompt principles:**
- Answer only from retrieved context, never infer or hallucinate beyond it
- For ambiguous queries (no property specified), ask: "Could you please clarify which property — 222 Rajpur, Max Towers, or Max House?"
- For cross-property comparisons, synthesise across all relevant sources explicitly
- If information is not in the documents, respond exactly: "This information is not available in the uploaded documents."

We can add more rules as per the industry needs.

**Model — Groq `llama-3.3-70b-versatile`:** 30 RPM / 12k tokens per query on free tier, ~200–700ms typical generation latency, quality comparable to GPT-4o-mini for factual Q&A. 
Free key from [console.groq.com](https://consolegroq.com).


---

## Running the Evaluation

Split across three files to work within Groq's free tier token limits. Each file covers one batch of questions and saves its own results JSON.

```bash
# No token limit — run everything at once
python eval1.py && python eval2.py && python eval3.py && python merge_eval_results.py

# With TPD limit — run one file per API key
python eval1.py      # Sections A + B 
# swap GROQ_API_KEY in .env, then:
python eval2.py      # Sections C + D + E
# swap again, then:
python eval3.py      # Sections F + G + H
python merge_eval_results.py   # combine into eval_final_results.json

# Flags available on all eval files
--upload      # upload all 3 PDFs first
--retrieval   # skip LLM calls, retrieval metrics only
--verbose     # print answers per question

```
---
All evaluation result files: eval1_results.json, eval2_results.json, eval3_results.json, and the merged eval_final_results.json are committed to the repository so you can inspect the full per-question breakdown without running the eval yourself.

---

## What Would Break First in Production

**In-memory index storage** is the most critical limitation. All FAISS indexes live in a Python dict in a single process. Multiple workers break query routing. Fix: migrate to Qdrant, Weaviate, or Pinecone for shared persistent vector storage.

**Synchronous indexing** blocks uploads for 10–30s on large PDFs. Fix: background job queue (Celery/ARQ) — return job ID immediately, poll for completion.

**Groq free tier limits** cause latency spikes under bulk eval load. Fix: paid Groq tier, or adaptive retry with exponential backoff.

**No authentication** — any client can upload PDFs or flood the query endpoint. Fix: API key middleware, rate limiting, file content validation.

---

## Configuration

All tunables in `app/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Lightweight, fast, strong general-purpose baseline |
| `RERANKER_MODEL` | ms-marco-MiniLM-L-6-v2 | Trained on MS MARCO, strong factual Q&A precision |
| `GROQ_MODEL` | llama-3.3-70b-versatile | Production LLM for answer synthesis |
| `CHUNK_SIZE` | 400 chars | Keeps related sentences together |
| `CHUNK_OVERLAP` | 60 chars | Prevents boundary cuts between chunks |
| `CANDIDATE_K` | 20 | Stage 1 fetch size — wider net for the reranker |
| `SCORE_THRESHOLD` | 0.3 | Min cosine similarity to pass Stage 1 |
| `DEFAULT_TOP_K` | 5 | Results returned per query |
| `MAX_FILE_SIZE_MB` | 50 | Upload size limit |
| `CACHE_TTL_SECONDS` | 3600 | Redis TTL for cached results |

---

## Challenges Addressed

**Retrieval precision.** Pure bi-encoder retrieval returned semantically similar but factually wrong chunks. Adding cross-encoder reranking fixed this, improving Top-1 from ~55% to 85%.

**Chunk size tuning.** Initial 200-char chunks fragmented specific facts like "Carpet area: 5789 sq. ft." across boundaries. Increasing to 400 chars with 60-char overlap kept related sentences together.

**Embedding cache regression.** An early implementation cached chunk embeddings via Upstash during indexing — this added 100+ HTTP round trips per document, making things slower. Removed entirely; only result-level caching at query time remains.

**LLM rate limits under evaluation.** Groq's free tier at 30 RPM caused progressive latency growth from ~400ms to 4s+ when hitting 80 questions back-to-back. Fixed by splitting evaluation into three files and adding a 2.6s sleep between non-cached LLM calls.

**Clarification for ambiguous queries.** The LLM initially returned multi-property answers for vague questions like "How many floors does it have?" Strengthened the system prompt to explicitly name all three properties and require clarification for context-free questions. Result: 100% clarification score.

---

## Future Additions

**Async indexing** — background job queue (celery), immediate job ID response, eliminates long HTTP holds on large PDFs.

**Hybrid search** — combine FAISS semantic search with BM25 keyword search, then rerank merged candidates. Improves recall for exact terms like RERA numbers and specific measurements.

**Shared vector storage** — migrate from in-memory FAISS to Qdrant or Pinecone for multi-worker deployments and persistent storage.

**Per-user document access control** — extend Supabase registry to support multi-user deployments where each organisation queries only their own documents.