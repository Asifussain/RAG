"""
FastAPI Backend — Phase 4
=========================
Endpoints:
  GET  /health  — Health check (index size, model status)
  POST /upload  — Upload & process a PDF
  POST /query   — Natural-language search over indexed chunks
  GET  /pdfs    — List all indexed PDFs and total chunk count
"""

import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB, UPLOADS_DIR
from app.embedder import Embedder
from app.models import (
    HealthResponse,
    PDFListResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
    UploadResponse,
)
from app.pdf_processor import process_pdf
from app.vector_store import VectorStore

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App & Singletons ─────────────────────────────────────────────────────
app = FastAPI(
    title="Real Estate Document Intelligence",
    description="Upload real estate PDFs and query them using natural language.",
    version="1.0.0",
)

# Loaded once at startup — kept in memory for fast inference
embedder: Embedder = None  # type: ignore[assignment]
vector_store: VectorStore = None  # type: ignore[assignment]


@app.on_event("startup")
async def startup():
    """Load the embedding model and FAISS index once at server start."""
    global embedder, vector_store

    logger.info("Starting up — loading model and index ...")
    embedder = Embedder()

    vector_store = VectorStore()
    vector_store.load()  # loads from disk if a saved index exists, else empty

    logger.info("Startup complete — index has %d vectors", vector_store.size)


# ── 1. Health Check ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        index_size=vector_store.size,
        model_loaded=embedder is not None,
    )


# ── 2. PDF Upload ────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    start = time.perf_counter()

    # --- Validate file extension ---
    filename = file.filename or "unknown.pdf"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{suffix}'. Only PDF files are accepted.",
        )

    # --- Validate file size ---
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    # --- Save to uploads/ ---
    save_path = UPLOADS_DIR / filename
    with open(save_path, "wb") as f:
        f.write(contents)

    # --- Process: extract text → chunk → embed → index ---
    try:
        chunks = process_pdf(save_path)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from this PDF.",
            )

        embeddings = embedder.embed_chunks(chunks, show_progress=False)
        vector_store.add(embeddings, chunks)
        vector_store.save()

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Uploaded '%s': %d chunks in %.0f ms", filename, len(chunks), elapsed_ms)

    return UploadResponse(
        message="PDF processed successfully",
        pdf_name=filename,
        chunks_created=len(chunks),
        processing_time_ms=round(elapsed_ms, 1),
    )


# ── 3. Query Search ─────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if vector_store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF first.",
        )

    start = time.perf_counter()

    query_embedding = embedder.embed_query(request.query)
    raw_results = vector_store.search(query_embedding, top_k=request.top_k)

    results = [
        SearchResult(
            text=r["chunk_text"],
            pdf_name=r["pdf_name"],
            page_number=r["page_number"],
            similarity_score=r["similarity_score"],
        )
        for r in raw_results
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Query '%s' → %d results in %.0f ms",
        request.query,
        len(results),
        elapsed_ms,
    )

    return QueryResponse(
        query=request.query,
        results=results,
        latency_ms=round(elapsed_ms, 1),
        total_results=len(results),
    )


# ── 4. List PDFs ────────────────────────────────────────────────────────

@app.get("/pdfs", response_model=PDFListResponse)
async def list_pdfs():
    return PDFListResponse(
        pdfs=vector_store.get_all_pdf_names(),
        total_chunks=vector_store.size,
    )


# ── 5. Frontend ─────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
