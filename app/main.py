"""
main.py — FastAPI application entry point.

Routes:
  POST /upload     → Upload a PDF, get back an index_id
  POST /query      → Query an indexed PDF
  GET  /health     → Health check + active index count
  GET  /indexes    → List all persisted index IDs

Run with:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_MIME_TYPE, EMBEDDING_MODEL, INDEX_DIR
from app.models import QueryRequest, QueryResponse, UploadResponse, HealthResponse, ChunkResult, CollectionResponse, FileSummary
from app.pipeline import rag_pipeline


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Real Estate Document Intelligence API",
    description="Upload real estate PDFs and query them using natural language.",
    version="1.0.0",
)

# Allow local frontend / Postman testing — tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Quick liveness check. Also useful to confirm model is loaded."""
    return HealthResponse(
        status="ok",
        model=EMBEDDING_MODEL,
        active_indexes=rag_pipeline.active_index_count(),
    )


@app.get("/indexes", tags=["System"])
def list_indexes():
    """
    Returns all index IDs that have been persisted to disk.
    Useful for reconnecting to previously uploaded PDFs after a restart.
    """
    saved = [p.name for p in INDEX_DIR.iterdir() if p.is_dir()]
    return {"indexes": saved, "count": len(saved)}


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file. The server will:
    1. Validate the file (type + size)
    2. Extract and clean text page by page
    3. Chunk and embed the content
    4. Build and save a FAISS index

    Returns an `index_id` — pass this in all subsequent /query calls.

    Latency: typically 2–15s depending on PDF size and CPU.
    Large PDFs (>50 pages) may take longer — consider chunking uploads client-side.
    """
    # ── Validation ────────────────────────────────────────────────────────
    if file.content_type != ALLOWED_MIME_TYPE:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Got: {file.content_type}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {size_mb:.1f}MB. Maximum is {MAX_FILE_SIZE_MB}MB.",
        )

    # ── Save to disk temporarily ───────────────────────────────────────────
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(contents)

    # ── Index ──────────────────────────────────────────────────────────────
    try:
        result = rag_pipeline.index_pdf(str(save_path))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indexing failed: {e}")
    finally:
        # Clean up the raw upload — we only need the FAISS index going forward
        save_path.unlink(missing_ok=True)

    return UploadResponse(
        index_id=result["index_id"],
        filename=result["filename"],
        total_pages=result["total_pages"],
        total_chunks=result["total_chunks"],
        message="PDF indexed successfully. Use the index_id to query.",
    )


@app.post("/query", response_model=QueryResponse, tags=["Search"])
def query_index(request: QueryRequest):
    """
    Query a previously indexed PDF using natural language.

    - `index_id`: returned from /upload
    - `question`: plain English question (e.g. "What are the nearby landmarks?")
    - `top_k`: how many results to return (default 3, max 10)

    Results include source metadata (filename, page number) for each chunk.
    Score is L2 distance — lower = more similar.
    """
    if not rag_pipeline.has_index(request.index_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{request.index_id}' not found. Please upload the PDF again.",
        )

    try:
        results, latency_ms = rag_pipeline.query(
            index_id=request.index_id,
            question=request.question,
            top_k=request.top_k or 3,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return QueryResponse(
        question=request.question,
        results=[
            ChunkResult(
                content=r.content,
                filename=r.filename,
                page_number=r.page_number,
                total_pages=r.total_pages,
                chunk_index=r.chunk_index,
                score=r.score,
            )
            for r in results
        ],
        latency_ms=round(latency_ms, 2),
        index_id=request.index_id,
    )


@app.post("/collection/create", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def create_collection(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDFs at once and merge them into a single searchable collection.
    Returns a `collection_id` — use this in /query just like a regular index_id.

    Querying a collection searches across ALL uploaded PDFs simultaneously.
    Each result tells you which filename and page it came from.

    Use this when you want to ask questions like:
    - "Which property has a swimming pool?"
    - "What is the cheapest property?"
    - "Which properties are near a metro station?"
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    saved_paths = []

    for file in files:
        if file.content_type != ALLOWED_MIME_TYPE:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"{file.filename}: Only PDF files are accepted.",
            )
        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{file.filename} is too large: {size_mb:.1f}MB. Max is {MAX_FILE_SIZE_MB}MB.",
            )
        save_path = UPLOAD_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(contents)
        saved_paths.append(save_path)

    try:
        result = rag_pipeline.index_collection([str(p) for p in saved_paths])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indexing failed: {e}")
    finally:
        for p in saved_paths:
            p.unlink(missing_ok=True)

    return CollectionResponse(
        collection_id=result["collection_id"],
        files=[FileSummary(**f) for f in result["files"]],
        total_chunks=result["total_chunks"],
        message=f"Collection of {len(result['files'])} PDFs indexed. Use collection_id to query across all documents.",
    )