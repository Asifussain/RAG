from pydantic import BaseModel, Field
from typing import List


# ── PDF Processing Models ─────────────────────────────────────────────────

class PageContent(BaseModel):
    """Represents extracted text from a single PDF page."""
    page_number: int
    text: str


class PDFContent(BaseModel):
    """Represents the full extracted content of a PDF."""
    pdf_name: str
    total_pages: int
    pages: List[PageContent]


class ChunkMetadata(BaseModel):
    """Metadata for a single text chunk produced by the chunking pipeline."""
    chunk_id: str
    pdf_name: str
    page_number: int
    chunk_text: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int


# ── API Request / Response Models ─────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    text: str
    pdf_name: str
    page_number: int
    similarity_score: float


class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    latency_ms: float
    total_results: int


class UploadResponse(BaseModel):
    message: str
    pdf_name: str
    chunks_created: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model_loaded: bool


class PDFListResponse(BaseModel):
    pdfs: List[str]
    total_chunks: int
