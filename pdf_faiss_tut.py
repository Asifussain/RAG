"""
PDF RAG Pipeline — Real Estate Edition
---------------------------------------
PDF → Clean Text → Chunks → FAISS → Semantic Search

Requirements:
    pip install pymupdf faiss-cpu sentence-transformers langchain-community langchain-text-splitters langchain-huggingface

Usage:
    python pdf_rag_faiss.py
"""

import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 200    # smaller = more focused chunks = better retrieval
CHUNK_OVERLAP   = 30
TOP_K           = 3


# ─────────────────────────────────────────────
# TEXT CLEANING
# Real estate PDFs are notorious for:
#   - OCR garbage (????, ####, ©, weird unicode)
#   - Table artifacts (||||, -----)
#   - Header/footer noise (page numbers, watermarks)
#   - Ligature issues (ﬁ, ﬂ instead of fi, fl)
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Aggressively clean extracted PDF text.
    Removes noise while preserving real estate content like
    prices, addresses, sq ft, bedroom counts, etc.
    """

    # Fix common ligature issues (PDF font artifacts)
    ligatures = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi",
        "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    }
    for bad, good in ligatures.items():
        text = text.replace(bad, good)

    # Remove non-printable / control characters (but keep newlines)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Remove repeated garbage symbols like ????, ####, ||||, ----
    text = re.sub(r"[?#|_\-]{3,}", " ", text)

    # Remove standalone numbers that are just page numbers (single/double digit alone on line)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Collapse more than 2 consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]

    # Drop lines that are too short to be meaningful (likely artifacts)
    # but keep lines with numbers (prices, sq ft, etc.)
    lines = [
        line for line in lines
        if len(line) > 15 or re.search(r"\d", line)
    ]

    return "\n".join(lines).strip()


# ─────────────────────────────────────────────
# PDF LOADER — uses PyMuPDF directly
# Much better than LangChain's default PDF loaders
# because we control extraction + cleaning per page
# ─────────────────────────────────────────────

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF file page by page using PyMuPDF.
    Returns a list of LangChain Documents, one per page,
    with metadata: filename, page_number, total_pages.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = []
    pdf = fitz.open(pdf_path)
    total_pages = len(pdf)

    for page_num in range(total_pages):
        page = pdf[page_num]

        # Extract text with layout preservation
        # "text" mode preserves reading order better than raw extraction
        raw_text = page.get_text("text")

        cleaned = clean_text(raw_text)

        # Skip pages that are essentially empty after cleaning
        # (cover pages, blank pages, full-image pages)
        if len(cleaned) < 50:
            print(f"  [SKIP] Page {page_num + 1} — too little text after cleaning")
            continue

        docs.append(Document(
            page_content=cleaned,
            metadata={
                "filename": path.name,
                "page_number": page_num + 1,   # 1-indexed for humans
                "total_pages": total_pages,
                "source": str(path.resolve()),
            }
        ))

    pdf.close()
    print(f"  → Loaded {len(docs)} usable pages from {path.name} (total: {total_pages})")
    return docs


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
# CORE PIPELINE
# ─────────────────────────────────────────────

class PDFRAGPipeline:
    """
    Scalability notes:
    - PyMuPDF is C-based, much faster than pdfminer/pypdf
    - Cleaning happens before chunking to avoid dirty chunks
    - Smaller chunks (200 chars) tuned for real estate detail queries
    - save_index/load_index avoids re-embedding on every restart
    - Embeddings normalized for cosine similarity (more accurate scores)
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vectorstore = None
        print("Model loaded.\n")

    def load_pdfs(self, pdf_paths: List[str]):
        """Load one or more PDFs, clean, chunk, and build FAISS index."""
        all_chunks = []

        for path in pdf_paths:
            print(f"Processing: {path}")
            try:
                pages = load_pdf(path)
                chunks = self.splitter.split_documents(pages)

                # Tag each chunk with its index for traceability
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = i

                print(f"  → {len(chunks)} chunks from {Path(path).name}\n")
                all_chunks.extend(chunks)

            except Exception as e:
                print(f"  [ERROR] Failed to process {path}: {e}\n")

        if not all_chunks:
            raise ValueError("No valid content extracted from PDFs.")

        print(f"Building FAISS index from {len(all_chunks)} total chunks...")
        t0 = time.time()
        self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
        print(f"Index built in {time.time() - t0:.2f}s\n")

    def save_index(self, path: str = "pdf_faiss_index"):
        """Save index to disk — skip re-indexing on next run."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Index saved to '{path}/'")

    def load_index(self, path: str = "pdf_faiss_index"):
        """Load previously saved index."""
        if not Path(path).exists():
            raise FileNotFoundError(f"No saved index at '{path}'")
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        print(f"Index loaded from '{path}/'\n")

    def query(self, question: str, top_k: int = TOP_K) -> List[SearchResult]:
        """Semantic search — returns top_k relevant chunks with metadata."""
        if not self.vectorstore:
            raise RuntimeError("No index. Call load_pdfs() or load_index() first.")

        t0 = time.time()
        results = self.vectorstore.similarity_search_with_score(question, k=top_k)
        latency = time.time() - t0

        search_results = []
        for doc, score in results:
            search_results.append(SearchResult(
                content=doc.page_content,
                filename=doc.metadata.get("filename", "unknown"),
                page_number=doc.metadata.get("page_number", -1),
                total_pages=doc.metadata.get("total_pages", -1),
                chunk_index=doc.metadata.get("chunk_index", -1),
                score=round(float(score), 4),
            ))

        print(f"Query completed in {latency*1000:.1f}ms")
        return search_results


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def display_results(results: List[SearchResult]):
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n{'─'*55}")
        print(f"  Result #{i}")
        print(f"  File      : {r.filename}")
        print(f"  Page      : {r.page_number} / {r.total_pages}")
        print(f"  Chunk     : #{r.chunk_index}")
        print(f"  Score     : {r.score}  (lower = more similar)")
        print(f"  Content   :\n{r.content.strip()}")
    print(f"\n{'─'*55}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Add your PDF files here ──
    PDF_FILES = [
        "E128-Skyvilla-Document.pdf",           # ← replace with your real estate PDF paths
        # "property_report.pdf",
        # "listing_brochure.pdf",
    ]

    pipeline = PDFRAGPipeline()

    # ── Step 2: Load and index PDFs ──
    pipeline.load_pdfs(PDF_FILES)

    # Save so you don't re-index every run (uncomment after first run)
    # pipeline.save_index("pdf_faiss_index")

    # Load saved index instead (uncomment after saving)
    # pipeline.load_index("pdf_faiss_index")

    # ── Step 3: Query ──
    queries = [
        "What are the nearby landmarks?",
        "What is the price of the property?",
        "How many bedrooms does the property have?",
        "What is the square footage?",
        "Is there parking available?",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = pipeline.query(q)
        display_results(results)