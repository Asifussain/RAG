"""
main.py — FastAPI application entry point.

Routes:
  POST /upload              -> Upload a single PDF, merged into master index
  POST /collection/create   -> Upload multiple PDFs, merged into master index
  POST /query/all           -> Query ALL ever-uploaded PDFs (master index)
  GET  /health              -> Health check + cache stats
  GET  /documents           -> List all indexed documents from Supabase

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
    QueryAllRequest, QueryResponse,
    UploadResponse, HealthResponse, ChunkResult,
    CollectionResponse, FileSummary,
    AnswerResponse, SourceRef,
)
from app.llm import generate_answer
from app.config import GROQ_MODEL as GEMINI_MODEL, LLM_ENABLED
from app.pipeline import rag_pipeline, MASTER_INDEX_ID
from app.cache import get_cached_results, set_cached_results, cache_stats


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


def _build_response(question: str, index_id: str,
                    pipeline_result: tuple, cached: bool = False) -> QueryResponse:
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
        cached=cached,
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
        cache=cache_stats(),
    )


@app.get("/documents", tags=["System"])
def list_documents():
    """List all indexed documents with real filenames. Reads from registry.json."""
    documents = rag_pipeline.get_registry()   # already sorted newest first
    return {
        "documents":          documents,
        "count":              len(documents),
        "master_index_ready": rag_pipeline.has_master(),
    }


@app.post("/answer", response_model=AnswerResponse, tags=["Search"])
def answer_question(request: QueryAllRequest):
    """
    Full RAG pipeline: retrieval + reranking + LLM generation.

    Stage 1 : FAISS fetches top-20 candidates
    Stage 2 : CrossEncoder reranks them
    Stage 3 : Gemini Flash synthesises a grounded natural language answer

    Returns the answer, sources cited, raw chunks, and stage-wise latency.
    Cached answers are served from Redis when available.
    """
    if not rag_pipeline.has_master():
        raise HTTPException(status_code=404, detail="No documents uploaded yet.")

    if not LLM_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="LLM generation not enabled. Set GEMINI_API_KEY in .env"
        )

    # Check answer cache
    cached = get_cached_results(request.question, f"answer:{MASTER_INDEX_ID}")
    if cached:
        return AnswerResponse(
            question      = request.question,
            answer        = cached["answer"],
            sources       = [SourceRef(**s) for s in cached["sources"]],
            chunks        = [ChunkResult(**c) for c in cached["chunks"]],
            retrieval_ms  = 0.0,
            generation_ms = 0.0,
            total_ms      = 0.0,
            cached        = True,
            model         = GEMINI_MODEL,
        )

    # Stage 1 + 2: retrieval
    try:
        results, s1_ms, s2_ms = rag_pipeline.query_all(
            request.question, request.top_k or 5
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    retrieval_ms = round(s1_ms + s2_ms, 2)

    # Stage 3: LLM generation
    result_dicts = [
        {
            "content":      r.content,
            "filename":     r.filename,
            "page_number":  r.page_number,
            "rerank_score": r.rerank_score,
        }
        for r in results
    ]

    llm_out       = generate_answer(request.question, result_dicts)
    generation_ms = llm_out["generation_ms"]
    total_ms      = round(retrieval_ms + generation_ms, 2)

    chunk_results = [
        ChunkResult(
            content      = r.content,
            filename     = r.filename,
            page_number  = r.page_number,
            total_pages  = r.total_pages,
            chunk_index  = r.chunk_index,
            score        = r.score,
            rerank_score = r.rerank_score,
        )
        for r in results
    ]

    source_refs = [SourceRef(**s) for s in llm_out["sources_used"]]

    # Cache the answer
    set_cached_results(request.question, f"answer:{MASTER_INDEX_ID}", {
        "answer":  llm_out["answer"],
        "sources": [s.dict() for s in source_refs],
        "chunks":  [c.dict() for c in chunk_results],
    })

    return AnswerResponse(
        question      = request.question,
        answer        = llm_out["answer"],
        sources       = source_refs,
        chunks        = chunk_results,
        retrieval_ms  = retrieval_ms,
        generation_ms = generation_ms,
        total_ms      = total_ms,
        cached        = False,
        model         = GEMINI_MODEL,
    )


# ─────────────────────────────────────────────
# UPLOAD ROUTES
# ─────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a single PDF. Merged into the master index — searchable immediately via /query/all."""
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

@app.post("/query/all", response_model=QueryResponse, tags=["Search"])
def query_all(request: QueryAllRequest):
    """
    Two-stage query across ALL ever-uploaded PDFs via master index.
    No index_id needed. Each result shows which file and page answered.
    Results are cached in Upstash Redis — repeated queries return instantly.
    """
    if not rag_pipeline.has_master():
        raise HTTPException(status_code=404, detail="No documents uploaded yet.")

    # Check result cache first
    cached = get_cached_results(request.question, MASTER_INDEX_ID)
    if cached:
        from app.pipeline import SearchResult
        results = [SearchResult(**r) for r in cached["results"]]
        return _build_response(
            request.question, MASTER_INDEX_ID,
            (results, 0.0, 0.0), cached=True
        )

    try:
        result = rag_pipeline.query_all(request.question, request.top_k or 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Cache for next time
    results, s1, s2 = result
    set_cached_results(request.question, MASTER_INDEX_ID, {
        "results": [vars(r) for r in results]
    })

    return _build_response(request.question, MASTER_INDEX_ID, result)


@app.delete("/documents/all", tags=["System"])
def clear_all_documents():
    """
    Wipe all indexed documents — clears FAISS master index,
    Supabase registry, and Redis cache.
    Re-upload PDFs after calling this.
    """
    import shutil

    # 1. Clear FAISS master index from disk
    master_path = INDEX_DIR / MASTER_INDEX_ID
    if master_path.exists():
        shutil.rmtree(master_path)

    # 2. Reset in-memory pipeline state
    rag_pipeline._indexes.clear()

    # 3. Clear Supabase registry
    from app.db import get_session, DocumentRegistry
    session = get_session()
    if session:
        try:
            session.query(DocumentRegistry).delete()
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()

    # 4. Fallback: clear local registry.json
    registry_path = INDEX_DIR / "registry.json"
    if registry_path.exists():
        registry_path.unlink()

    return {"message": "All documents cleared. Re-upload PDFs to continue."}


# ─────────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))