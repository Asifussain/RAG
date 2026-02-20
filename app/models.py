"""
models.py — Pydantic schemas for all API request/response contracts.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Inbound ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, example="What is the carpet area of Sky Villa 1?")
    index_id: str = Field(..., description="index_id or collection_id from /upload or /collection/create")
    top_k: Optional[int] = Field(default=5, ge=1, le=10)


class QueryAllRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, example="Which property has a swimming pool?")
    top_k: Optional[int] = Field(default=5, ge=1, le=10)


# ── Outbound ───────────────────────────────────────────────────────────────

class ChunkResult(BaseModel):
    content: str
    filename: str
    page_number: int
    total_pages: int
    chunk_index: int
    score: float = Field(description="Stage 1 cosine similarity — higher = more similar")
    rerank_score: Optional[float] = Field(default=None, description="Stage 2 cross-encoder relevance score — higher = more relevant")


class QueryResponse(BaseModel):
    question: str
    results: List[ChunkResult]
    stage1_latency_ms: float = Field(description="FAISS bi-encoder search time")
    stage2_latency_ms: float = Field(description="Cross-encoder reranking time")
    total_latency_ms: float
    index_id: str


class UploadResponse(BaseModel):
    index_id: str
    filename: str
    total_pages: int
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    reranker: str
    active_indexes: int
    master_index_ready: bool


class FileSummary(BaseModel):
    filename: str
    total_pages: Optional[int] = None
    chunks: Optional[int] = None
    error: Optional[str] = None


class CollectionResponse(BaseModel):
    collection_id: str
    files: List[FileSummary]
    total_chunks: int
    message: str