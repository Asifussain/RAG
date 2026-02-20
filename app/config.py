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