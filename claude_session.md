Context: Real Estate Doc Intelligence Project

1. Project Architecture & Stack:

    Backend: FastAPI (Python 3.11/3.12).

    PDF Engine: PyMuPDF (fitz) with a custom cleaning pipeline in utils.py that filters out layout artifacts (standalone page numbers and short orphan lines).

    Embeddings: all-MiniLM-L6-v2 via sentence-transformers>=2.7.0 (pinned to fix huggingface_hub compatibility).

    Vector Store: FAISS (IndexFlatIP) using L2-normalized vectors to provide Cosine Similarity scores (0-100%).

    Frontend: Single-page HTML/Vanilla JS with Tailwind CSS (CDN), served via FastAPI StaticFiles. Features a live health-check badge, drag-and-drop upload, and ranked search results with metadata citations.

2. Key Implementation Details:

    Chunking Logic: 300-token sliding window with 50-token overlap. Majority-vote page assignment for chunks spanning multiple pages.

    Persistence: FAISS index and metadata are saved to data/index/ after every upload and reloaded on startup.

    Fixes Applied: * Resolved ImportError for cached_download by upgrading sentence-transformers.

        Fixed ValueError on document closure by caching page_count before doc.close().

        Enhanced clean_text in utils.py to strip standalone digits (page numbers) often found in design-heavy brochures.

3. Current Progress vs. Agmentis Deliverables:

    [x] Phase 1-2: Extraction & Chunking Pipeline.

    [x] Phase 3: Embedding & FAISS Vector Store.

    [x] Phase 4: FastAPI Backend & CRUD Endpoints.

    [x] Extra: Polished Web UI with Metadata Citations (PDF Name + Page Number).

    [ ] Phase 5 (PENDING): Creation of 18-20 test questions + Ground Truth.

    [ ] Phase 6 (PENDING): Latency Analysis (Average & P95) and Retrieval Accuracy (Top-1/Top-3).

    [ ] Phase 7 (PENDING): Final README.md and metrics report.

4. Folder Structure:
Plaintext

real-estate-doc-intelligence/
├── app/
│   ├── main.py (FastAPI + Static Mount)
│   ├── pdf_processor.py (PyMuPDF + Chunking)
│   ├── embedder.py (Sentence-Transformers)
│   ├── vector_store.py (FAISS Index Management)
│   ├── models.py (Pydantic Schemas)
│   └── utils.py (Cleaning + Jaccard Similarity)
├── data/ (uploads, processed, index)
└── static/ (index.html)