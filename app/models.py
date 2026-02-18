"""
models.py — Pydantic schemas for all API request/response contracts.
Keeping these separate makes it easy to version the API later.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Inbound ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, example="What are the nearby landmarks?")
    index_id: str = Field(..., description="Index ID returned after uploading a PDF")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of results to return")


# ── Outbound ───────────────────────────────────────────────────────────────

class ChunkResult(BaseModel):
    content: str
    filename: str
    page_number: int
    total_pages: int
    chunk_index: int
    score: float = Field(description="L2 distance — lower = more similar")


class QueryResponse(BaseModel):
    question: str
    results: List[ChunkResult]
    latency_ms: float
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
    active_indexes: int


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