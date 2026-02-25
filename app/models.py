"""
models.py — Pydantic schemas for all API request/response contracts.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# Inbound

class QueryAllRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, example="Which property has a swimming pool?")
    top_k: Optional[int] = Field(default=5, ge=1, le=10)


# Outbound

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
    cached: bool = Field(default=False, description="True if result was served from Redis cache")


class UploadResponse(BaseModel):
    filename: str
    total_pages: int
    total_chunks: int
    message: str


class SourceRef(BaseModel):
    filename:    str
    page_number: int
    rerank_score: float = 0.0


class AnswerResponse(BaseModel):
    question:         str
    answer:           str
    sources:          List[SourceRef]
    chunks:           List[ChunkResult]        # raw chunks for inspection
    retrieval_ms:     float
    generation_ms:    float
    total_ms:         float
    cached:           bool = False
    model:            str  = "llama-3.3-70b-versatile"


class HealthResponse(BaseModel):
    status: str
    model: str
    reranker: str
    active_indexes: int
    master_index_ready: bool
    cache: Optional[dict] = None


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