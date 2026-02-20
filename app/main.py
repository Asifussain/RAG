"""
main.py — FastAPI application entry point.

Routes:
  POST /upload              -> Upload a single PDF, adds to master index
  POST /collection/create   -> Upload multiple PDFs, adds all to master index
  POST /query               -> Query a specific index_id or collection_id
  POST /query/all           -> Query ALL ever-uploaded PDFs (master index)
  GET  /health              -> Health check
  GET  /documents           -> List all persisted indexes

Run with:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import (
    UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_MIME_TYPE,
    EMBEDDING_MODEL, RERANKER_MODEL, INDEX_DIR,
)
from app.models import (
    QueryRequest, QueryAllRequest, QueryResponse,
    UploadResponse, HealthResponse, ChunkResult,
    CollectionResponse, FileSummary,
)
from app.pipeline import rag_pipeline, MASTER_INDEX_ID


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Real Estate Document Intelligence API",
    description=(
        "Two-stage RAG: FAISS bi-encoder retrieval + cross-encoder reranking. "
        "Upload real estate PDFs and query them using natural language."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

async def _save_upload(file: UploadFile) -> Path:
    if file.content_type != ALLOWED_MIME_TYPE:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{file.filename}: only PDF files are accepted.",
        )
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{file.filename} is {size_mb:.1f}MB — max allowed is {MAX_FILE_SIZE_MB}MB.",
        )
    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(contents)
    return save_path


def _build_response(question: str, index_id: str, pipeline_result: tuple) -> QueryResponse:
    """Convert pipeline output tuple into QueryResponse."""
    results, stage1_ms, stage2_ms = pipeline_result
    return QueryResponse(
        question=question,
        results=[
            ChunkResult(
                content=r.content,
                filename=r.filename,
                page_number=r.page_number,
                total_pages=r.total_pages,
                chunk_index=r.chunk_index,
                score=r.score,
                rerank_score=r.rerank_score,
            )
            for r in results
        ],
        stage1_latency_ms=round(stage1_ms, 2),
        stage2_latency_ms=round(stage2_ms, 2),
        total_latency_ms=round(stage1_ms + stage2_ms, 2),
        index_id=index_id,
    )


# ─────────────────────────────────────────────
# SYSTEM ROUTES
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="ok",
        model=EMBEDDING_MODEL,
        reranker=RERANKER_MODEL,
        active_indexes=rag_pipeline.active_index_count(),
        master_index_ready=rag_pipeline.has_master(),
    )


@app.get("/documents", tags=["System"])
def list_documents():
    """List all indexed documents with real filenames. Reads from registry.json."""
    registry  = rag_pipeline.get_registry()
    documents = [
        {"index_id": idx, **meta}
        for idx, meta in registry.items()
    ]
    documents.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
    return {
        "documents":          documents,
        "count":              len(documents),
        "master_index_ready": rag_pipeline.has_master(),
    }


# ─────────────────────────────────────────────
# UPLOAD ROUTES
# ─────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a single PDF. Adds to its own index AND the master index."""
    save_path = await _save_upload(file)
    try:
        result = rag_pipeline.index_pdf(str(save_path))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    finally:
        save_path.unlink(missing_ok=True)

    return UploadResponse(
        index_id=result["index_id"],
        filename=result["filename"],
        total_pages=result["total_pages"],
        total_chunks=result["total_chunks"],
        message="PDF indexed. Use /query/all to search across all documents.",
    )


@app.post("/collection/create", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def create_collection(files: List[UploadFile] = File(...)):
    """Upload multiple PDFs as a named collection. Also adds all to master index."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_paths = []
    for file in files:
        saved_paths.append(await _save_upload(file))

    try:
        result = rag_pipeline.index_collection([str(p) for p in saved_paths])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    finally:
        for p in saved_paths:
            p.unlink(missing_ok=True)

    return CollectionResponse(
        collection_id=result["collection_id"],
        files=[FileSummary(**f) for f in result["files"]],
        total_chunks=result["total_chunks"],
        message=f"{len(result['files'])} PDFs indexed. Use /query/all to search everything.",
    )


# ─────────────────────────────────────────────
# QUERY ROUTES
# ─────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Search"])
def query_index(request: QueryRequest):
    """
    Two-stage query on a specific index.
    Stage 1: FAISS fetches candidates. Stage 2: cross-encoder reranks them.
    Response includes both stage latencies and rerank_score per result.
    """
    if not rag_pipeline.has_index(request.index_id):
        raise HTTPException(status_code=404, detail=f"Index '{request.index_id}' not found.")
    try:
        result = rag_pipeline.query(request.index_id, request.question, request.top_k or 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _build_response(request.question, request.index_id, result)


@app.post("/query/all", response_model=QueryResponse, tags=["Search"])
def query_all(request: QueryAllRequest):
    """
    Two-stage query across ALL ever-uploaded PDFs via master index.
    No index_id needed. Each result shows which file and page answered.
    """
    if not rag_pipeline.has_master():
        raise HTTPException(status_code=404, detail="No documents uploaded yet.")
    try:
        result = rag_pipeline.query_all(request.question, request.top_k or 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _build_response(request.question, MASTER_INDEX_ID, result)


# ─────────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))