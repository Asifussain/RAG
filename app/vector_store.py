"""
FAISS Vector Store — Phase 3.2
===============================
Handles:
  1. Creating and managing a FAISS index (IndexFlatIP for normalized vectors)
  2. Adding embeddings with associated metadata
  3. Searching by query vector
  4. Persisting / loading the index and metadata to/from disk
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.config import (
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from app.models import ChunkMetadata

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory FAISS index with a parallel metadata list.

    Index type decision (from the plan):
      - < 1 000 chunks  → IndexFlatIP  (exact search, <50 ms)
      - 1 000 – 50 000  → IndexIVFFlat (approximate, upgrade path documented)
      - > 50 000        → IndexIVFPQ   (compressed, future)

    We start with IndexFlatIP (Inner Product).  Because all embeddings are
    L2-normalized at encoding time, inner-product search is equivalent to
    cosine similarity — and FAISS IP search returns *higher = better* scores
    directly, which is more intuitive than L2 distances.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []  # parallel list — metadata[i] ↔ vector i

    # ── Adding Vectors ────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, chunks: List[ChunkMetadata]) -> None:
        """Add embeddings and their metadata to the index.

        Args:
            embeddings: Array of shape (n, dimension), L2-normalized.
            chunks:     Corresponding ChunkMetadata list (same length as embeddings).
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and chunks ({len(chunks)}) "
                "must have the same length"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)

        for chunk in chunks:
            self.metadata.append({
                "chunk_id": chunk.chunk_id,
                "pdf_name": chunk.pdf_name,
                "page_number": chunk.page_number,
                "chunk_text": chunk.chunk_text,
                "chunk_index": chunk.chunk_index,
            })

        logger.info(
            "Added %d vectors (total index size: %d)", len(chunks), self.index.ntotal
        )

    # ── Searching ─────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[Dict]:
        """Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: Array of shape (1, dimension), L2-normalized.
            top_k:           Number of results to return.

        Returns:
            List of dicts, each containing metadata fields plus
            'similarity_score' (cosine similarity, higher is better).
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        # Clamp top_k to the number of vectors available
        top_k = min(top_k, self.index.ntotal)

        start = time.perf_counter()
        scores, indices = self.index.search(query_embedding, top_k)
        elapsed = (time.perf_counter() - start) * 1000

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for missing results
            meta = self.metadata[idx].copy()
            meta["similarity_score"] = round(float(score), 4)
            results.append(meta)

        logger.info(
            "Search returned %d results in %.1f ms", len(results), elapsed
        )
        return results

    # ── Persistence ───────────────────────────────────────────────────

    def save(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Write the FAISS index and metadata to disk.

        Args:
            index_path:    Where to save the FAISS index (default: config path).
            metadata_path: Where to save the metadata JSON (default: config path).
        """
        index_path = Path(index_path or FAISS_INDEX_PATH)
        metadata_path = Path(metadata_path or METADATA_PATH)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        logger.info(
            "Saved index (%d vectors) to %s", self.index.ntotal, index_path
        )

    def load(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Load a previously saved FAISS index and metadata from disk.

        Args:
            index_path:    Path to the FAISS index file.
            metadata_path: Path to the metadata JSON file.
        """
        index_path = Path(index_path or FAISS_INDEX_PATH)
        metadata_path = Path(metadata_path or METADATA_PATH)

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(
                "No saved index found at %s — starting with empty index",
                index_path,
            )
            return

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(
            "Loaded index with %d vectors from %s", self.index.ntotal, index_path
        )

    # ── Utilities ─────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    def get_all_pdf_names(self) -> List[str]:
        """Return a deduplicated list of PDF names in the store."""
        return list({m["pdf_name"] for m in self.metadata})

    def clear(self) -> None:
        """Reset the index and metadata (useful for testing)."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Vector store cleared")
