"""
config.py — Central configuration for the RAG pipeline.
Tune these values to balance speed vs. accuracy.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
UPLOAD_DIR   = BASE_DIR / "uploads"
INDEX_DIR    = BASE_DIR / "indexes"

UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# ── Embedding Model ────────────────────────────────────────────────────────
# all-MiniLM-L6-v2  → fast, small, good general-purpose baseline (~80MB)
# all-mpnet-base-v2 → slower but more accurate (~420MB) — swap if accuracy matters more
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ───────────────────────────────────────────────────────────────
# Smaller chunks = more focused retrieval, but more index entries
# Larger chunks = more context per result, but noisier matches
CHUNK_SIZE    = 200
CHUNK_OVERLAP = 30

# ── Retrieval ──────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 3      # Results returned per query
MAX_TOP_K     = 10     # Hard cap to prevent abuse / latency blowout

# ── Upload constraints ─────────────────────────────────────────────────────
MAX_FILE_SIZE_MB  = 50       # Reject files above this size
ALLOWED_MIME_TYPE = "application/pdf"