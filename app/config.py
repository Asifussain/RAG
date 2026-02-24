from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR  = BASE_DIR / "indexes"

UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 60

DEFAULT_TOP_K = 5
MAX_TOP_K     = 10
CANDIDATE_K   = 20
SCORE_THRESHOLD = 0.3

MAX_FILE_SIZE_MB  = 50
ALLOWED_MIME_TYPE = "application/pdf"

"""
config.py — Central configuration for the RAG pipeline.
All sensitive values loaded from .env via python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR  = BASE_DIR / "indexes"

UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Stage 1: Bi-encoder
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Stage 2: Cross-encoder reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking 
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 60

# Retrieval
DEFAULT_TOP_K   = 5
MAX_TOP_K       = 10
CANDIDATE_K     = 20
SCORE_THRESHOLD = 0.3

# Upload constraints
MAX_FILE_SIZE_MB  = 50
ALLOWED_MIME_TYPE = "application/pdf"

# Supabase PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Upstash Redis Cache
UPSTASH_REDIS_REST_URL   = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

# TTL for cached query results (seconds)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))

# Whether caching is enabled (auto-disabled if credentials missing)
CACHE_ENABLED = bool(UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN)
DB_ENABLED    = bool(DATABASE_URL)