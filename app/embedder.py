"""
Embedding Pipeline — Phase 3.1
===============================
Handles:
  1. Loading the sentence-transformer model (once, at startup)
  2. Encoding text chunks into dense vectors (batch processing)
  3. Encoding single queries for search
"""

import logging
import time
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from app.models import ChunkMetadata

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformer model for embedding generation.

    The model is loaded once on instantiation and kept in memory so that
    subsequent calls (upload-time batch encoding, query-time single encoding)
    don't pay the model-load cost again.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info("Loading embedding model '%s' ...", model_name)
        start = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        load_time = (time.perf_counter() - start) * 1000
        self.dimension = EMBEDDING_DIMENSION
        logger.info("Model loaded in %.0f ms (dim=%d)", load_time, self.dimension)

    # ── Batch Embedding (for document upload) ─────────────────────────

    def embed_chunks(
        self,
        chunks: List[ChunkMetadata],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of text chunks.

        Args:
            chunks:        List of ChunkMetadata whose chunk_text will be encoded.
            batch_size:    Number of texts encoded per forward pass (default 32).
            show_progress: Show a tqdm progress bar.

        Returns:
            NumPy array of shape (n_chunks, embedding_dim), L2-normalized.
        """
        texts = [c.chunk_text for c in chunks]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.info(
            "Embedding %d chunks (batch_size=%d) ...", len(texts), batch_size
        )
        start = time.perf_counter()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
            convert_to_numpy=True,
        )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Embedded %d chunks in %.0f ms (%.1f ms/chunk)",
            len(texts),
            elapsed,
            elapsed / len(texts),
        )

        return embeddings.astype(np.float32)

    # ── Single Query Embedding (for search) ───────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a normalized embedding vector.

        Args:
            query: The natural-language search query.

        Returns:
            NumPy array of shape (1, embedding_dim), L2-normalized.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)
