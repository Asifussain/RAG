"""
pipeline.py — PDF RAG Pipeline (server-ready refactor).

Key differences from the original script:
- Embedding model loaded ONCE at startup (not per request) → no cold-start penalty
- Each uploaded PDF gets its own FAISS index keyed by index_id (UUID)
- Indexes can be persisted to disk and reloaded across restarts
- Returns structured dataclasses rather than printing to stdout
"""

import re
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    DEFAULT_TOP_K, MAX_TOP_K, INDEX_DIR
)


# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class SearchResult:
    content: str
    filename: str
    page_number: int
    total_pages: int
    chunk_index: int
    score: float


# ─────────────────────────────────────────────
# TEXT CLEANING
# Real estate PDFs are notorious for:
#   - OCR garbage (????, ####, ©, weird unicode)
#   - Table artifacts (||||, -----)
#   - Header/footer noise (page numbers, watermarks)
#   - Ligature issues (ﬁ, ﬂ instead of fi, fl)
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    ligatures = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff",
        "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    }
    for bad, good in ligatures.items():
        text = text.replace(bad, good)

    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"[?#|_\-]{3,}", " ", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [
        line for line in lines
        if len(line) > 15 or re.search(r"\d", line)
    ]
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────
# PDF LOADER
# ─────────────────────────────────────────────

def load_pdf(pdf_path: str) -> List[Document]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = []
    pdf = fitz.open(pdf_path)
    total_pages = len(pdf)

    for page_num in range(total_pages):
        page = pdf[page_num]
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)

        if len(cleaned) < 50:
            continue

        docs.append(Document(
            page_content=cleaned,
            metadata={
                "filename": path.name,
                "page_number": page_num + 1,
                "total_pages": total_pages,
                "source": str(path.resolve()),
            }
        ))

    pdf.close()
    return docs, total_pages


# ─────────────────────────────────────────────
# PIPELINE — singleton shared across requests
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Singleton-style pipeline.
    - Embedding model is loaded once at app startup.
    - Each uploaded PDF creates a separate FAISS index stored in self._indexes.
    - Indexes are optionally persisted to disk for restart survival.

    Scalability notes:
    - In-memory dict works fine for a prototype / single-server deployment.
    - For multi-worker production: replace with a shared index store
      (e.g., Redis + serialized FAISS, or a dedicated vector DB like Qdrant/Weaviate).
    - PyMuPDF (fitz) is C-based — much faster extraction than pdfminer/pypdf.
    - Cleaning happens before chunking so dirty tokens don't pollute embeddings.
    """

    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # index_id → FAISS vectorstore
        self._indexes: Dict[str, FAISS] = {}

    # ── Index management ──────────────────────────────────────────────────

    def index_pdf(self, pdf_path: str) -> dict:
        """
        Load, clean, chunk, and index a single PDF.
        Returns index metadata: index_id, filename, total_pages, total_chunks.
        """
        pages, total_pages = load_pdf(pdf_path)

        if not pages:
            raise ValueError("No extractable text found in this PDF.")

        chunks = self._splitter.split_documents(pages)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        index_id = str(uuid.uuid4())
        vectorstore = FAISS.from_documents(chunks, self._embeddings)
        self._indexes[index_id] = vectorstore

        # Persist to disk so we can reload on restart
        save_path = INDEX_DIR / index_id
        vectorstore.save_local(str(save_path))

        return {
            "index_id": index_id,
            "filename": Path(pdf_path).name,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
        }

    def index_collection(self, pdf_paths: List[str]) -> dict:
        """
        Merge multiple PDFs into a single FAISS index (collection).
        All PDFs are searchable together — results include which file/page matched.

        Why merge instead of querying each index separately:
        - Single embedding lookup vs. N lookups
        - Results are globally ranked by similarity across all docs
        - Simpler API — one collection_id covers everything

        Tradeoff: you can't remove a single PDF later without rebuilding the index.
        For a prototype this is fine.
        """
        all_chunks = []
        file_summaries = []
        global_chunk_idx = 0

        for pdf_path in pdf_paths:
            try:
                pages, total_pages = load_pdf(pdf_path)
                if not pages:
                    continue

                chunks = self._splitter.split_documents(pages)
                for chunk in chunks:
                    chunk.metadata["chunk_index"] = global_chunk_idx
                    global_chunk_idx += 1

                all_chunks.extend(chunks)
                file_summaries.append({
                    "filename": Path(pdf_path).name,
                    "total_pages": total_pages,
                    "chunks": len(chunks),
                })
            except Exception as e:
                # Skip bad PDFs, don't fail the whole collection
                file_summaries.append({
                    "filename": Path(pdf_path).name,
                    "error": str(e),
                })

        if not all_chunks:
            raise ValueError("No extractable text found across all uploaded PDFs.")

        collection_id = str(uuid.uuid4())
        vectorstore = FAISS.from_documents(all_chunks, self._embeddings)
        self._indexes[collection_id] = vectorstore

        save_path = INDEX_DIR / collection_id
        vectorstore.save_local(str(save_path))

        return {
            "collection_id": collection_id,
            "files": file_summaries,
            "total_chunks": len(all_chunks),
        }

    def load_index(self, index_id: str) -> bool:
        """Load a previously saved index from disk into memory."""
        save_path = INDEX_DIR / index_id
        if not save_path.exists():
            return False
        if index_id not in self._indexes:
            self._indexes[index_id] = FAISS.load_local(
                str(save_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        return True

    def has_index(self, index_id: str) -> bool:
        if index_id in self._indexes:
            return True
        # Try loading from disk (handles server restarts)
        return self.load_index(index_id)

    def active_index_count(self) -> int:
        return len(self._indexes)

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, index_id: str, question: str, top_k: int = DEFAULT_TOP_K) -> tuple:
        """
        Semantic search against a specific index.
        Returns (List[SearchResult], latency_ms).
        """
        if not self.has_index(index_id):
            raise KeyError(f"Index '{index_id}' not found.")

        top_k = min(top_k, MAX_TOP_K)
        vectorstore = self._indexes[index_id]

        t0 = time.perf_counter()
        raw_results = vectorstore.similarity_search_with_score(question, k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Deduplicate — same text can appear on consecutive pages (repeated layouts)
        seen = set()
        deduped = []
        for doc, score in raw_results:
            h = hash(doc.page_content.strip())
            if h not in seen:
                seen.add(h)
                deduped.append((doc, score))

        results = [
            SearchResult(
                content=doc.page_content,
                filename=doc.metadata.get("filename", "unknown"),
                page_number=doc.metadata.get("page_number", -1),
                total_pages=doc.metadata.get("total_pages", -1),
                chunk_index=doc.metadata.get("chunk_index", -1),
                score=round(float(score), 4),
            )
            for doc, score in deduped
        ]
        return results, latency_ms


# ── Module-level singleton ─────────────────────────────────────────────────
# FastAPI imports this; the model loads once when the process starts.
rag_pipeline = RAGPipeline()