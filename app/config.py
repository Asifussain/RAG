import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# Ensure directories exist
for _dir in [UPLOADS_DIR, PROCESSED_DIR, INDEX_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Chunking Parameters ───────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 300        # ~400 words per chunk
CHUNK_OVERLAP_TOKENS = 50     # ~67 words overlap between chunks
DEDUP_SIMILARITY_THRESHOLD = 0.90  # Skip chunks with >90% similarity

# ── Embedding Model ───────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# ── FAISS ─────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# ── API ───────────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf"}
DEFAULT_TOP_K = 3
MAX_TOP_K = 10
