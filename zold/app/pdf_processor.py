"""
PDF Processing Pipeline — Phase 2
==================================
Handles:
  1. Text extraction from PDFs (page-by-page, via PyMuPDF)
  2. Text cleaning
  3. Sliding-window chunking with token-level control
  4. Deduplication of near-identical chunks
  5. Metadata generation & persistence to data/processed/
"""

import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from app.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    DEDUP_SIMILARITY_THRESHOLD,
    PROCESSED_DIR,
    UPLOADS_DIR,
)
from app.models import ChunkMetadata, PageContent, PDFContent
from app.utils import clean_text, compute_text_similarity

logger = logging.getLogger(__name__)


# ── Tokenization Helpers ──────────────────────────────────────────────────
# We use a lightweight whitespace tokenizer so there is no dependency on the
# embedding model at this stage.  Each "token" is simply a whitespace-
# delimited word, which is a reasonable proxy (~1.3 words per BPE token on
# average for English text).

def _tokenize(text: str) -> List[str]:
    """Split text into whitespace tokens."""
    return text.split()


def _detokenize(tokens: List[str]) -> str:
    """Join tokens back into a string."""
    return " ".join(tokens)


# ── PDF Text Extraction ──────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> PDFContent:
    """Extract cleaned text from every page of a PDF.

    Uses PyMuPDF (fitz) for fast, high-quality extraction that handles
    multi-column layouts better than most alternatives.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A PDFContent object containing per-page text and metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a valid / readable PDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{pdf_path.name}': {exc}") from exc

    pages: List[PageContent] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # "text" sort mode reorders blocks by position which helps with
        # multi-column layouts — text flows more naturally top-to-bottom,
        # left-to-right.
        raw_text = page.get_text("text", sort=True)
        cleaned = clean_text(raw_text)
        if cleaned:  # skip completely blank pages
            pages.append(PageContent(page_number=page_num + 1, text=cleaned))

    total_pages = len(doc)
    doc.close()

    logger.info(
        "Extracted %d non-empty pages from '%s' (%d total pages)",
        len(pages),
        pdf_path.name,
        total_pages,
    )

    return PDFContent(
        pdf_name=pdf_path.name,
        total_pages=len(pages),
        pages=pages,
    )


# ── Text Chunking ────────────────────────────────────────────────────────

def chunk_text(
    pdf_content: PDFContent,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    dedup_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> List[ChunkMetadata]:
    """Create overlapping chunks from extracted PDF text.

    Strategy (from the plan):
      - Sliding window of *chunk_size* tokens with *overlap* token overlap.
      - Each chunk carries metadata linking it back to its source PDF & page.
      - Near-duplicate chunks (Jaccard similarity > *dedup_threshold*) are
        dropped to keep the index lean.

    Args:
        pdf_content: Extracted PDF content (from extract_text_from_pdf).
        chunk_size:  Number of tokens per chunk (default 300).
        overlap:     Number of overlapping tokens between consecutive chunks
                     (default 50).
        dedup_threshold: Similarity above which a chunk is considered a
                         duplicate and skipped (default 0.90).

    Returns:
        List of ChunkMetadata objects ready for embedding.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    # ── 1. Build a flat list of (token, page_number) tuples ───────────
    token_page_map: List[tuple] = []  # (token_str, page_number)
    for page in pdf_content.pages:
        tokens = _tokenize(page.text)
        for tok in tokens:
            token_page_map.append((tok, page.page_number))

    if not token_page_map:
        logger.warning("No tokens extracted from '%s'", pdf_content.pdf_name)
        return []

    total_tokens = len(token_page_map)
    step = chunk_size - overlap  # how far the window slides each iteration

    # ── 2. Slide the window and build raw chunks ──────────────────────
    raw_chunks: List[dict] = []
    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        window = token_page_map[start:end]

        chunk_tokens = [t[0] for t in window]
        chunk_pages = [t[1] for t in window]

        # The "page_number" for the chunk is the page where most of its
        # tokens originate (majority vote).
        page_number = max(set(chunk_pages), key=chunk_pages.count)

        # Character offsets are relative to the full concatenated text
        char_start = sum(len(t[0]) + 1 for t in token_page_map[:start])
        chunk_text_str = _detokenize(chunk_tokens)
        char_end = char_start + len(chunk_text_str)

        raw_chunks.append({
            "text": chunk_text_str,
            "page_number": page_number,
            "char_start": char_start,
            "char_end": char_end,
        })

        # If we've reached the end of the document, stop
        if end >= total_tokens:
            break
        start += step

    # ── 3. Deduplicate near-identical chunks ──────────────────────────
    deduped_chunks: List[dict] = []
    for chunk in raw_chunks:
        is_dup = False
        for existing in deduped_chunks:
            sim = compute_text_similarity(chunk["text"], existing["text"])
            if sim > dedup_threshold:
                is_dup = True
                logger.debug(
                    "Skipping duplicate chunk (similarity=%.2f)", sim
                )
                break
        if not is_dup:
            deduped_chunks.append(chunk)

    # ── 4. Build final ChunkMetadata list ─────────────────────────────
    total_chunks = len(deduped_chunks)
    chunks: List[ChunkMetadata] = []
    for idx, chunk in enumerate(deduped_chunks):
        chunks.append(
            ChunkMetadata(
                chunk_id=str(uuid.uuid4()),
                pdf_name=pdf_content.pdf_name,
                page_number=chunk["page_number"],
                chunk_text=chunk["text"],
                chunk_index=idx,
                total_chunks=total_chunks,
                char_start=chunk["char_start"],
                char_end=chunk["char_end"],
            )
        )

    logger.info(
        "Chunked '%s': %d raw → %d after dedup (chunk_size=%d, overlap=%d)",
        pdf_content.pdf_name,
        len(raw_chunks),
        total_chunks,
        chunk_size,
        overlap,
    )
    return chunks


# ── Persistence ──────────────────────────────────────────────────────────

def save_chunks(chunks: List[ChunkMetadata], output_dir: Optional[Path] = None) -> Path:
    """Persist chunk metadata to a JSON file in data/processed/.

    Args:
        chunks: List of ChunkMetadata objects to save.
        output_dir: Directory to write to (defaults to PROCESSED_DIR).

    Returns:
        Path to the written JSON file.
    """
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunks:
        logger.warning("No chunks to save.")
        return output_dir

    pdf_name = chunks[0].pdf_name
    # Use the PDF stem (without extension) as the filename
    out_path = output_dir / f"{Path(pdf_name).stem}_chunks.json"

    data = [chunk.model_dump() for chunk in chunks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d chunks to %s", len(chunks), out_path)
    return out_path


def load_chunks(json_path: Path) -> List[ChunkMetadata]:
    """Load chunk metadata from a previously saved JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ChunkMetadata(**item) for item in data]


# ── High-Level Pipeline ──────────────────────────────────────────────────

def process_pdf(pdf_path: Path) -> List[ChunkMetadata]:
    """End-to-end pipeline: extract → chunk → save.

    This is the main entry point that later phases (embedding, indexing)
    will call.

    Args:
        pdf_path: Path to the uploaded PDF.

    Returns:
        List of ChunkMetadata objects (also persisted to disk).
    """
    logger.info("Processing PDF: %s", pdf_path)

    # Step 1 — Extract text
    pdf_content = extract_text_from_pdf(pdf_path)

    # Step 2 — Chunk text
    chunks = chunk_text(pdf_content)

    # Step 3 — Save to disk
    save_chunks(chunks)

    logger.info(
        "Pipeline complete for '%s': %d chunks created",
        pdf_path.name,
        len(chunks),
    )
    return chunks
