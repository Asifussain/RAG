"""
pipeline.py — PDF RAG Pipeline with Two-Stage Retrieval.

Two-stage retrieval strategy:
  Stage 1 — Bi-encoder (FAISS):
    Fast approximate search. Embeds query + chunks independently.
    Retrieves CANDIDATE_K candidates (e.g. 20) quickly via cosine similarity.
    Good at recall, not always perfect at ranking.

  Stage 2 — Cross-encoder (reranker):
    Reads (query, chunk) TOGETHER — understands full context and intent.
    Re-scores all candidates and returns the true top_k.
    Slower per-pair but only runs on ~20 candidates, not the full index.
    Dramatically improves precision — especially for specific factual queries
    like "carpet area of Sky Villa 1" where the bi-encoder may rank
    descriptive text above the actual number.

"""

import re
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from app.config import (
    EMBEDDING_MODEL, RERANKER_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    DEFAULT_TOP_K, MAX_TOP_K, CANDIDATE_K,
    INDEX_DIR, SCORE_THRESHOLD, DB_ENABLED,
)
from app.db import save_document, get_all_documents, init_db
from app.cache import get_cached_results, set_cached_results

MASTER_INDEX_ID = "master"


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
    score: float                        # bi-encoder cosine similarity score
    rerank_score: float = field(default=None)  # cross-encoder relevance score


# ─────────────────────────────────────────────
# TEXT CLEANING
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

def load_pdf(pdf_path: str) -> Tuple[List[Document], int]:
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
# FAISS INDEX FACTORY — Inner Product (cosine)
# ─────────────────────────────────────────────

def make_faiss_index(chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Build FAISS index using IndexFlatIP (inner product).
    Since embeddings are L2-normalized, IP = cosine similarity.
    Scores are in [0, 1] where higher = more similar.
    """
    texts     = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Batch embed all texts in one call — always fresh for new documents
    t0 = time.perf_counter()
    raw_vectors = embeddings.embed_documents(texts)
    vectors = np.array(raw_vectors, dtype=np.float32)
    print(f"  Embedding {len(texts)} chunks: {(time.perf_counter()-t0)*1000:.0f}ms")

    t1 = time.perf_counter()
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"  FAISS index build: {(time.perf_counter()-t1)*1000:.0f}ms")

    docstore_dict = {}
    index_to_id   = {}
    for i, (text, meta) in enumerate(zip(texts, metadatas)):
        doc_id = str(uuid.uuid4())
        index_to_id[i]      = doc_id
        docstore_dict[doc_id] = Document(page_content=text, metadata=meta)

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(docstore_dict),
        index_to_docstore_id=index_to_id,
    )


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Two-stage retrieval pipeline.

    Stage 1 — FAISS bi-encoder: fast, fetches CANDIDATE_K candidates.
    Stage 2 — CrossEncoder reranker: precise, re-scores candidates and
              returns the true top_k in correct relevance order.

    Both models load once at startup — zero per-request model loading cost.
    """

    def __init__(self):
        # Stage 1: bi-encoder for embedding + FAISS search
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Stage 2: cross-encoder for reranking candidates
        # ms-marco-MiniLM-L-6-v2 is trained on MS MARCO passage ranking
        print(f"Loading reranker: {RERANKER_MODEL} ...")
        self._reranker = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,     # truncate long chunks safely
            device="cpu",
        )
        print("Reranker loaded.\n")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._indexes: Dict[str, FAISS] = {}

        # Init Supabase table on startup
        init_db()

        self._load_master()

    # ── Master index ──────────────────────────────────────────────────────

    def _load_master(self):
        master_path = INDEX_DIR / MASTER_INDEX_ID
        if master_path.exists():
            self._indexes[MASTER_INDEX_ID] = FAISS.load_local(
                str(master_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )

    # ── Registry — Supabase PostgreSQL ────────────────────────────────────
    # Stores filename + metadata per index_id.
    # Falls back to registry.json if DATABASE_URL not configured.

    def _save_registry_entry(self, index_id: str, filename: str,
                              total_pages: int, total_chunks: int,
                              doc_type: str = "document", extra: dict = None):
        if DB_ENABLED:
            save_document(index_id, filename, total_pages,
                         total_chunks, doc_type, extra)
        else:
            # Fallback: local JSON registry
            import json
            p = INDEX_DIR / "registry.json"
            reg = json.loads(p.read_text()) if p.exists() else {}
            reg[index_id] = {
                "filename": filename, "total_pages": total_pages,
                "total_chunks": total_chunks, "doc_type": doc_type,
            }
            p.write_text(json.dumps(reg, indent=2))

    def get_registry(self) -> list:
        if DB_ENABLED:
            return get_all_documents()
        # Fallback: local JSON registry
        import json
        p = INDEX_DIR / "registry.json"
        if not p.exists():
            return []
        reg = json.loads(p.read_text())
        return [{"index_id": k, **v} for k, v in reg.items()]

    def _save_master(self):
        if MASTER_INDEX_ID in self._indexes:
            self._indexes[MASTER_INDEX_ID].save_local(
                str(INDEX_DIR / MASTER_INDEX_ID)
            )

    def _update_master(self, new_vectorstore: FAISS):
        t0 = time.perf_counter()
        if MASTER_INDEX_ID not in self._indexes:
            master_path = str(INDEX_DIR / MASTER_INDEX_ID)
            new_vectorstore.save_local(master_path)
            self._indexes[MASTER_INDEX_ID] = FAISS.load_local(
                master_path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self._indexes[MASTER_INDEX_ID].merge_from(new_vectorstore)
            self._save_master()
        print(f"  Master index update: {(time.perf_counter()-t0)*1000:.0f}ms")

    def has_master(self) -> bool:
        return MASTER_INDEX_ID in self._indexes

    # ── Per-document indexing ─────────────────────────────────────────────

    def index_pdf(self, pdf_path: str) -> dict:
        pages, total_pages = load_pdf(pdf_path)
        if not pages:
            raise ValueError("No extractable text found in this PDF.")

        chunks = self._splitter.split_documents(pages)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        print(f"Indexing {Path(pdf_path).name} — {len(chunks)} chunks")
        vectorstore = make_faiss_index(chunks, self._embeddings)

        self._update_master(vectorstore)

        filename = Path(pdf_path).name
        doc_id   = str(uuid.uuid4())  
        self._save_registry_entry(
            doc_id, filename,
            total_pages, len(chunks), "document"
        )
        return {
            "filename":     filename,
            "total_pages":  total_pages,
            "total_chunks": len(chunks),
        }

    def index_collection(self, pdf_paths: List[str]) -> dict:
        all_chunks     = []
        file_summaries = []
        global_idx     = 0

        for pdf_path in pdf_paths:
            try:
                pages, total_pages = load_pdf(pdf_path)
                if not pages:
                    continue
                chunks = self._splitter.split_documents(pages)
                for chunk in chunks:
                    chunk.metadata["chunk_index"] = global_idx
                    global_idx += 1
                all_chunks.extend(chunks)
                file_summaries.append({
                    "filename":    Path(pdf_path).name,
                    "total_pages": total_pages,
                    "chunks":      len(chunks),
                })
            except Exception as e:
                file_summaries.append({"filename": Path(pdf_path).name, "error": str(e)})

        if not all_chunks:
            raise ValueError("No extractable text found across all uploaded PDFs.")

        vectorstore   = make_faiss_index(all_chunks, self._embeddings)
        collection_id = str(uuid.uuid4())

        self._update_master(vectorstore)

        col_filename = ", ".join(
            f["filename"] for f in file_summaries if "error" not in f
        )
        col_pages = sum(
            f.get("total_pages", 0) for f in file_summaries if "error" not in f
        )
        self._save_registry_entry(
            collection_id, col_filename,
            col_pages, len(all_chunks), "collection",
            extra={"files": file_summaries}
        )

        return {
            "collection_id": collection_id,
            "files":         file_summaries,
            "total_chunks":  len(all_chunks),
        }

    # ── Index loading ─────────────────────────────────────────────────────

    def has_index(self, index_id: str) -> bool:
        # Only master index exists now
        return index_id == MASTER_INDEX_ID and self.has_master()

    def active_index_count(self) -> int:
        return len(self._indexes)

    # ── Two-stage search ──────────────────────────────────────────────────

    def _run_search(self, vectorstore: FAISS, question: str, top_k: int) -> Tuple[List[SearchResult], float, float]:
        """
        Stage 1: Fetch CANDIDATE_K candidates via FAISS cosine search.
        Stage 2: Rerank with CrossEncoder, return true top_k.

        Returns (results, stage1_latency_ms, stage2_latency_ms).
        """
        top_k      = min(top_k, MAX_TOP_K)
        candidate_k = max(CANDIDATE_K, top_k)  # always fetch at least top_k

        # ── Stage 1: FAISS bi-encoder search ──────────────────────────────
        t0 = time.perf_counter()
        raw_results = vectorstore.similarity_search_with_score(question, k=candidate_k)
        stage1_ms = (time.perf_counter() - t0) * 1000

        # Dedup by content hash
        seen, candidates = set(), []
        for doc, score in raw_results:
            h = hash(doc.page_content.strip())
            if h not in seen and score >= SCORE_THRESHOLD:
                seen.add(h)
                candidates.append((doc, score))

        if not candidates:
            return [], stage1_ms, 0.0

        # ── Stage 2: Cross-encoder reranking ──────────────────────────────
        # Feed (query, chunk_text) pairs — cross-encoder reads both together
        # and produces a relevance score that understands intent vs. content
        t1 = time.perf_counter()
        pairs         = [(question, doc.page_content) for doc, _ in candidates]
        rerank_scores = self._reranker.predict(pairs)   # returns raw logits (higher = more relevant)
        stage2_ms     = (time.perf_counter() - t1) * 1000

        # Sort by rerank score descending, take top_k
        reranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = [
            SearchResult(
                content=doc.page_content,
                filename=doc.metadata.get("filename", "unknown"),
                page_number=doc.metadata.get("page_number", -1),
                total_pages=doc.metadata.get("total_pages", -1),
                chunk_index=doc.metadata.get("chunk_index", -1),
                score=round(float(bi_score), 4),
                rerank_score=round(float(rerank_score), 4),
            )
            for (doc, bi_score), rerank_score in reranked
        ]

        return results, stage1_ms, stage2_ms

    def query_all(self, question: str, top_k: int = DEFAULT_TOP_K):
        if not self.has_master():
            raise RuntimeError("No documents uploaded yet.")
        return self._run_search(self._indexes[MASTER_INDEX_ID], question, top_k)


# ── Module-level singleton ─────────────────────────────────────────────────
rag_pipeline = RAGPipeline()