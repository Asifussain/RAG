import os
import argparse
import json
import time
import sys
import numpy as np
import requests
from pathlib import Path

API   = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TOP_K       = 5

PDF_FILES = [
    "E128-Skyvilla-Document.pdf",
    "222-rajpur-brochure.pdf",
]

# ── Evaluation set ─────────────────────────────────────────────────────────
# NOTE: keywords are used ONLY by this evaluation script to verify
# whether retrieved chunks contain the correct answer.
# The RAG pipeline receives ONLY the 'question' field, nothing else.

EVAL_SET = [
    {
        "id": "Q01",
        "question": "What is the carpet area of Sky Villa 1 compared to Sky Villa 2 in Estate 128?",
        "keywords": ["5789", "6827", "537", "634"],
        "source_hint": "E128-Skyvilla-Document.pdf",
    },
    {
        "id": "Q02",
        "question": "Does the Estate 128 triplex include a private swimming pool and a dedicated gym?",
        "keywords": ["swimming pool", "gym", "private pool", "exclusive gym"],
        "source_hint": "E128-Skyvilla-Document.pdf",
    },
    {
        "id": "Q03",
        "question": "Are there separate entrances for guest rooms and bedrooms in the Estate 128 Sky Villas?",
        "keywords": ["separate entrance", "guest room", "privacy"],
        "source_hint": "E128-Skyvilla-Document.pdf",
    },
    {
        "id": "Q04",
        "question": "Describe the architectural features of the double-height living room in the Sky Villas.",
        "keywords": ["double height", "double heighted", "living room", "wraparound"],
        "source_hint": "E128-Skyvilla-Document.pdf",
    },
    {
        "id": "Q05",
        "question": "How wide are the wraparound balconies at Estate 128?",
        "keywords": ["12", "13", "feet", "wide", "wraparound"],
        "source_hint": "E128-Skyvilla-Document.pdf",
    },
    {
        "id": "Q06",
        "question": "Is Estate 128 a RERA registered project and what is its registration number?",
        "keywords": ["rera", "registration", "upreraprj"],
        "source_hint": "E128-Skyvilla-Document.pdf",
    },
    {
        "id": "Q07",
        "question": "Is there a provision for private elevators in the Sky Villas of Estate 128?",
        "keywords": ["elevator", "lift", "private elevator", "private lift"],
        "source_hint": "E128-Skyvilla-Document.pdf",
    },
    {
        "id": "Q08",
        "question": "What kind of air conditioning system is used in Estate 128?",
        "keywords": ["air conditioning", "ac", "vrf", "vrv", "central air", "ducted"],
        "source_hint": "E128-Skyvilla-Document.pdf",
    },
    {
        "id": "Q09",
        "question": "What is the total built-up area and carpet area of a Townhouse in 222 Rajpur?",
        "keywords": ["built-up", "carpet area", "townhouse", "sq. ft", "sq ft"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q10",
        "question": "How many units are available for Courtyard Villas versus Forest Villas in the Dehradun project?",
        "keywords": ["courtyard", "forest villa", "units", "number"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q11",
        "question": "What is the range of plot sizes for the Townhouses in 222 Rajpur?",
        "keywords": ["plot", "sq. ft", "sq ft", "townhouse", "range"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q12",
        "question": "What are the key features of the Forest Villas in 222 Rajpur regarding their connection to the landscape?",
        "keywords": ["forest", "landscape", "nature", "trees", "botanical", "green"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q13",
        "question": "Do the townhouses in 222 Rajpur feature a sky court or atrium for natural light?",
        "keywords": ["sky court", "atrium", "natural light", "skylight"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q14",
        "question": "How are public service areas like kitchens and staff rooms separated from private spaces in 222 Rajpur?",
        "keywords": ["kitchen", "staff", "service", "private", "separated", "separation"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q15",
        "question": "Is there a provision for private elevators in the Forest Villas of Dehradun?",
        "keywords": ["elevator", "lift", "private"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q16",
        "question": "What unique botanical feature is reserved exclusively for residents at 222 Rajpur?",
        "keywords": ["botanical", "orchard", "garden", "exclusive", "resident"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q17",
        "question": "What are the primary natural views offered to residents of 222 Rajpur?",
        "keywords": ["view", "mountain", "himalaya", "valley", "forest", "nature"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q18",
        "question": "What is the UKRERA registration number for the 222 Rajpur project in Dehradun?",
        "keywords": ["ukrera", "registration", "rera"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q19",
        "question": "To which Seismic Zone does the structural design of 222 Rajpur adhere?",
        "keywords": ["seismic", "zone", "zone iv", "zone 4"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
    {
        "id": "Q20",
        "question": "What specific security measures are implemented for resident car recognition at the Dehradun site and how long does it take to drive to Jolly Grant Airport?",
        "keywords": ["car recognition", "rfid", "number plate", "jolly grant", "airport", "minutes"],
        "source_hint": "222-Rajpur-Document.pdf",
    },
]


def check_server():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        d = r.json()
        print(f"✓ Server online — model: {d['model']}")
        print(f"  reranker: {d.get('reranker', 'n/a')}")
        print(f"  master index ready: {d.get('master_index_ready', False)}\n")
        return True
    except Exception as e:
        print(f"✗ Server not reachable at {API}: {e}")
        return False


def upload_pdfs():
    """Upload all PDFs and build master index."""
    print("── Uploading PDFs ────────────────────────────────")
    for pdf in PDF_FILES:
        path = Path(pdf)
        if not path.exists():
            print(f"  [SKIP] {pdf} not found. Download from https://maxestates.in/downloads")
            continue
        with open(path, "rb") as f:
            r = requests.post(
                f"{API}/upload",
                files={"file": (path.name, f, "application/pdf")},
                timeout=120,
            )
        if r.ok:
            d = r.json()
            print(f"  ✓ {d['filename']} — {d['total_pages']} pages, {d['total_chunks']} chunks")
        else:
            print(f"  ✗ {pdf} — {r.json().get('detail', 'unknown error')}")
    print()


def contains_keyword(text: str, keywords: list) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run_query(question: str) -> dict:
    r = requests.post(
        f"{API}/query/all",
        json={"question": question, "top_k": TOP_K},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def run_eval(verbose: bool = False):
    print("── Running Evaluation ────────────────────────────")
    print(f"  Questions : {len(EVAL_SET)}")
    print(f"  Top-K     : {TOP_K}")
    print(f"  Endpoint  : {API}/query/all\n")

    latencies    = []
    s1_latencies = []
    s2_latencies = []
    top1_hits    = 0
    top3_hits    = 0
    results_log  = []

    for i, item in enumerate(EVAL_SET, 1):
        qid      = item["id"]
        question = item["question"]
        keywords = item["keywords"]

        try:
            data = run_query(question)
        except Exception as e:
            print(f"  [{qid}] ERROR: {e}")
            results_log.append({**item, "top1": False, "top3": False, "error": str(e)})
            continue

        total_ms = data.get("total_latency_ms", 0)
        s1_ms    = data.get("stage1_latency_ms", 0)
        s2_ms    = data.get("stage2_latency_ms", 0)
        results  = data.get("results", [])

        latencies.append(total_ms)
        s1_latencies.append(s1_ms)
        s2_latencies.append(s2_ms)

        # Check top-1
        top1 = False
        if results:
            top1 = contains_keyword(results[0]["content"], keywords)

        # Check top-3
        top3 = any(
            contains_keyword(results[j]["content"], keywords)
            for j in range(min(3, len(results)))
        )

        if top1: top1_hits += 1
        if top3: top3_hits += 1

        status = "✓" if top1 else ("~" if top3 else "✗")
        label  = "TOP-1" if top1 else ("TOP-3" if top3 else "MISS ")
        print(f"  [{qid}] {status} {label}  {total_ms:6.1f}ms  {question[:60]}")

        if verbose:
            for j, r in enumerate(results[:3], 1):
                mark = "✓" if contains_keyword(r["content"], keywords) else " "
                print(f"         {mark} #{j} rerank={r.get('rerank_score','?'):.2f}  "
                      f"pg{r['page_number']}  {r['content'][:80].replace(chr(10),' ')}")
            print()

        results_log.append({
            **item,
            "top1": top1,
            "top3": top3,
            "latency_ms": total_ms,
            "top_result": results[0]["content"][:120] if results else "",
        })

    n = len(latencies)
    if n == 0:
        print("\nNo results collected — check server and PDFs.")
        return

    avg_ms  = np.mean(latencies)
    p95_ms  = np.percentile(latencies, 95)
    p99_ms  = np.percentile(latencies, 99)
    min_ms  = np.min(latencies)
    max_ms  = np.max(latencies)

    avg_s1  = np.mean(s1_latencies)
    avg_s2  = np.mean(s2_latencies)

    top1_pct = top1_hits / len(EVAL_SET) * 100
    top3_pct = top3_hits / len(EVAL_SET) * 100

    print(f"""
══════════════════════════════════════════════════
  EVALUATION RESULTS
══════════════════════════════════════════════════

  RETRIEVAL QUALITY
  ─────────────────
  Top-1 Accuracy : {top1_hits}/{len(EVAL_SET)}  ({top1_pct:.1f}%)
  Top-3 Accuracy : {top3_hits}/{len(EVAL_SET)}  ({top3_pct:.1f}%)

  LATENCY  (n={n} queries)
  ────────────────────────
  Average total  : {avg_ms:.1f} ms
  P95 total      : {p95_ms:.1f} ms   ← key submission metric
  P99 total      : {p99_ms:.1f} ms
  Min / Max      : {min_ms:.1f} / {max_ms:.1f} ms

  Stage breakdown (avg):
    Stage 1 FAISS  : {avg_s1:.1f} ms
    Stage 2 Rerank : {avg_s2:.1f} ms

══════════════════════════════════════════════════
""")

    print("  Per-question breakdown:")
    print(f"  {'ID':<5} {'Top1':<6} {'Top3':<6} {'Latency':>10}  Question")
    print(f"  {'─'*5} {'─'*6} {'─'*6} {'─'*10}  {'─'*40}")
    for r in results_log:
        t1 = "✓" if r.get("top1") else "✗"
        t3 = "✓" if r.get("top3") else "✗"
        ms = f"{r.get('latency_ms', 0):.1f}ms"
        q  = r["question"][:50]
        print(f"  {r['id']:<5} {t1:<6} {t3:<6} {ms:>10}  {q}")

    output = {
        "summary": {
            "top1_accuracy": round(top1_pct, 1),
            "top3_accuracy": round(top3_pct, 1),
            "top1_hits": top1_hits,
            "top3_hits": top3_hits,
            "total_questions": len(EVAL_SET),
            "latency": {
                "avg_ms":   round(avg_ms, 1),
                "p95_ms":   round(p95_ms, 1),
                "p99_ms":   round(p99_ms, 1),
                "min_ms":   round(min_ms, 1),
                "max_ms":   round(max_ms, 1),
                "avg_stage1_ms": round(avg_s1, 1),
                "avg_stage2_ms": round(avg_s2, 1),
            },
        },
        "questions": results_log,
    }

    out_path = Path("eval_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Full results saved to: {out_path}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluator")
    parser.add_argument("--upload",  action="store_true", help="Upload PDFs before evaluating")
    parser.add_argument("--verbose", action="store_true", help="Show top-3 chunks per question")
    args = parser.parse_args()

    if not check_server():
        sys.exit(1)

    if args.upload:
        upload_pdfs()

    run_eval(verbose=args.verbose)