"""
eval.py — Comprehensive RAG Pipeline Evaluation
-------------------------------------------------
Covers all metrics required by Agmentis review round:

  Retrieval Quality:
    Recall@K, Top-1/3 Accuracy, MRR, nDCG

  Caching:
    Cold vs warm latency, % improvement

  Latency:
    Stage-wise breakdown, Avg, P95, P99

  Robustness:
    Paraphrase Robustness Score
    Entity Coverage Score

  Hallucination / False Positives:
    Negative query test — questions with no answer in documents

Usage:
    python eval.py --upload     # upload PDFs then evaluate
    python eval.py              # evaluate only
    python eval.py --verbose    # show retrieved chunks per question
    API_BASE_URL=https://your-api.com python eval.py
"""

import os
import math
import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

API   = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TOP_K = 5

PDF_FILES = [
    "E128-Skyvilla-Document.pdf",
    "222-rajpur-brochure.pdf",
]

# ── Evaluation set ─────────────────────────────────────────────────────────
EVAL_SET = [
    {"id":"Q01","question":"What is the carpet area of Sky Villa 1 compared to Sky Villa 2 in Estate 128?","keywords":["5789","6827","537","634"],"entities":["5789","6827"],"source":"E128"},
    {"id":"Q02","question":"Does the Estate 128 triplex include a private swimming pool and a dedicated gym?","keywords":["swimming pool","gym","private pool","exclusive gym"],"entities":["swimming pool","gym"],"source":"E128"},
    {"id":"Q03","question":"Are there separate entrances for guest rooms and bedrooms in the Estate 128 Sky Villas?","keywords":["separate entrance","guest room","privacy"],"entities":["separate entrance","guest"],"source":"E128"},
    {"id":"Q04","question":"Describe the architectural features of the double-height living room in the Sky Villas.","keywords":["double height","double heighted","living room","wraparound"],"entities":["double height","living room"],"source":"E128"},
    {"id":"Q05","question":"How wide are the wraparound balconies at Estate 128?","keywords":["12","13","feet","wide","wraparound"],"entities":["12","feet"],"source":"E128"},
    {"id":"Q06","question":"Is Estate 128 a RERA registered project and what is its registration number?","keywords":["rera","registration","upreraprj"],"entities":["rera"],"source":"E128"},
    {"id":"Q07","question":"Is there a provision for private elevators in the Sky Villas of Estate 128?","keywords":["elevator","lift","private elevator","private lift"],"entities":["elevator"],"source":"E128"},
    {"id":"Q08","question":"What kind of air conditioning system is used in Estate 128?","keywords":["air conditioning","ac","vrf","vrv","central air","ducted"],"entities":["air conditioning"],"source":"E128"},
    {"id":"Q09","question":"What is the total built-up area and carpet area of a Townhouse in 222 Rajpur?","keywords":["built-up","carpet area","townhouse","sq. ft","sq ft"],"entities":["carpet area","townhouse"],"source":"222Rajpur"},
    {"id":"Q10","question":"How many units are available for Courtyard Villas versus Forest Villas in the Dehradun project?","keywords":["courtyard","forest villa","units"],"entities":["courtyard","forest villa"],"source":"222Rajpur"},
    {"id":"Q11","question":"What is the range of plot sizes for the Townhouses in 222 Rajpur?","keywords":["plot","sq. ft","sq ft","townhouse"],"entities":["plot","townhouse"],"source":"222Rajpur"},
    {"id":"Q12","question":"What are the key features of the Forest Villas in 222 Rajpur regarding their connection to the landscape?","keywords":["forest","landscape","nature","trees","botanical","green"],"entities":["forest","landscape"],"source":"222Rajpur"},
    {"id":"Q13","question":"Do the townhouses in 222 Rajpur feature a sky court or atrium for natural light?","keywords":["sky court","atrium","natural light","skylight"],"entities":["sky court"],"source":"222Rajpur"},
    {"id":"Q14","question":"How are public service areas like kitchens and staff rooms separated from private spaces in 222 Rajpur?","keywords":["kitchen","staff","service","private","separated"],"entities":["kitchen","staff"],"source":"222Rajpur"},
    {"id":"Q15","question":"Is there a provision for private elevators in the Forest Villas of Dehradun?","keywords":["elevator","lift","private"],"entities":["elevator"],"source":"222Rajpur"},
    {"id":"Q16","question":"What unique botanical feature is reserved exclusively for residents at 222 Rajpur?","keywords":["botanical","orchard","garden","exclusive","resident"],"entities":["botanical"],"source":"222Rajpur"},
    {"id":"Q17","question":"What are the primary natural views offered to residents of 222 Rajpur?","keywords":["view","mountain","himalaya","valley","forest","nature"],"entities":["view"],"source":"222Rajpur"},
    {"id":"Q18","question":"What is the UKRERA registration number for the 222 Rajpur project in Dehradun?","keywords":["ukrera","registration","rera"],"entities":["ukrera"],"source":"222Rajpur"},
    {"id":"Q19","question":"To which Seismic Zone does the structural design of 222 Rajpur adhere?","keywords":["seismic","zone","zone iv","zone 4"],"entities":["seismic","zone"],"source":"222Rajpur"},
    {"id":"Q20","question":"What specific security measures are implemented for resident car recognition at the Dehradun site?","keywords":["car recognition","rfid","number plate","security","vehicle"],"entities":["car recognition"],"source":"222Rajpur"},
]

# ── Paraphrase set ─────────────────────────────────────────────────────────
PARAPHRASE_SET = [
    {"original_id":"Q01","paraphrase":"Tell me the size in square feet of Sky Villa 1 and Sky Villa 2.","keywords":["5789","6827","537","634"]},
    {"original_id":"Q05","paraphrase":"What is the width of the balconies in Estate 128 Sky Villas?","keywords":["12","13","feet","wide"]},
    {"original_id":"Q13","paraphrase":"Is natural light brought in through a sky court or atrium in the 222 Rajpur townhouses?","keywords":["sky court","atrium","natural light"]},
    {"original_id":"Q18","paraphrase":"What is the RERA registration number for the Dehradun project?","keywords":["ukrera","registration","rera"]},
    {"original_id":"Q19","paraphrase":"Which earthquake zone rating applies to 222 Rajpur construction?","keywords":["seismic","zone","zone iv","zone 4"]},
]

# ── Negative set (answers NOT in documents) ────────────────────────────────
NEGATIVE_SET = [
    {"id":"N01","question":"What is the price per square foot of Sky Villa 1?","reason":"Pricing not in documents"},
    {"id":"N02","question":"Who is the architect of Estate 128?","reason":"Architect not mentioned"},
    {"id":"N03","question":"What is the monthly maintenance fee for 222 Rajpur?","reason":"Maintenance fees not in documents"},
    {"id":"N04","question":"How many parking spaces does each unit get in Estate 128?","reason":"Parking count not specified"},
    {"id":"N05","question":"What is the possession date for the Forest Villas in Dehradun?","reason":"Possession date not mentioned"},
]

FALSE_POSITIVE_RERANK_THRESHOLD = 0.8


# ── Helpers ────────────────────────────────────────────────────────────────

def check_server():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        d = r.json()
        print(f"✓ Server online")
        print(f"  Model    : {d.get('model','n/a')}")
        print(f"  Reranker : {d.get('reranker','n/a')}")
        print(f"  Cache    : {d.get('cache',{})}")
        print(f"  Master   : {d.get('master_index_ready',False)}\n")
        return True
    except Exception as e:
        print(f"✗ Server not reachable: {e}")
        return False


def upload_pdfs():
    print("── Uploading PDFs ─────────────────────────────────────────")
    for pdf in PDF_FILES:
        path = Path(pdf)
        if not path.exists():
            print(f"  [SKIP] {pdf} not found")
            continue
        with open(path,"rb") as f:
            r = requests.post(f"{API}/upload",files={"file":(path.name,f,"application/pdf")},timeout=120)
        if r.ok:
            d = r.json()
            print(f"  ✓ {d['filename']} — {d['total_pages']}p, {d['total_chunks']} chunks")
        else:
            print(f"  ✗ {pdf} — {r.json().get('detail','error')}")
    print()


def run_query(question):
    r = requests.post(f"{API}/query/all",json={"question":question,"top_k":TOP_K},timeout=30)
    r.raise_for_status()
    return r.json()


def has_keyword(text, keywords):
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def hit_rank(results, keywords):
    for i,r in enumerate(results,1):
        if has_keyword(r["content"],keywords):
            return i
    return 0


def calc_mrr(ranks):
    return round(sum(1.0/r for r in ranks if r>0)/len(ranks),4) if ranks else 0.0


def calc_ndcg(results, keywords, k=5):
    gains = [1.0 if has_keyword(results[i]["content"],keywords) else 0.0 for i in range(min(k,len(results)))]
    dcg   = sum(g/math.log2(i+2) for i,g in enumerate(gains))
    idcg  = sum(1.0/math.log2(i+2) for i in range(int(sum(gains))))
    return round(dcg/idcg,4) if idcg>0 else 0.0


def calc_entity_coverage(content, entities):
    if not entities or not content:
        return 0.0
    cl = content.lower()
    return round(sum(1 for e in entities if e.lower() in cl)/len(entities),4)


# ── Main evaluation ────────────────────────────────────────────────────────

def run_eval(verbose=False):

    # ── Section 1: Retrieval Quality ──────────────────────────────────────
    print("── Section 1: Retrieval Quality ───────────────────────────")

    latencies,s1_lats,s2_lats = [],[],[]
    ranks,ndcgs,entity_covs   = [],[],[]
    results_log               = []

    for item in EVAL_SET:
        try:
            data = run_query(item["question"])
        except Exception as e:
            print(f"  [{item['id']}] ERROR: {e}")
            results_log.append({**item,"error":str(e)})
            continue

        res    = data.get("results",[])
        tot_ms = data.get("total_latency_ms",0)
        s1_ms  = data.get("stage1_latency_ms",0)
        s2_ms  = data.get("stage2_latency_ms",0)
        cached = data.get("cached",False)

        if not cached:
            latencies.append(tot_ms)
            s1_lats.append(s1_ms)
            s2_lats.append(s2_ms)

        rank       = hit_rank(res, item["keywords"])
        ndcg       = calc_ndcg(res, item["keywords"], k=TOP_K)
        top1_txt   = res[0]["content"] if res else ""
        ent_cov    = calc_entity_coverage(top1_txt, item["entities"])

        ranks.append(rank)
        ndcgs.append(ndcg)
        entity_covs.append(ent_cov)

        top1   = rank==1
        top3   = 0<rank<=3
        status = "✓" if top1 else ("~" if top3 else "✗")
        label  = "TOP-1" if top1 else ("TOP-3" if top3 else "MISS ")
        tag    = " [CACHE]" if cached else ""
        print(f"  [{item['id']}] {status} {label}  {tot_ms:6.1f}ms{tag}  nDCG={ndcg:.2f}  ent={ent_cov:.2f}  {item['question'][:52]}")

        if verbose and res:
            for j,r in enumerate(res[:3],1):
                m = "✓" if has_keyword(r["content"],item["keywords"]) else " "
                print(f"         {m} #{j} rerank={r.get('rerank_score',0):.2f}  {r['content'][:80].replace(chr(10),' ')}")
            print()

        results_log.append({**item,"rank":rank,"ndcg":ndcg,"entity_coverage":ent_cov,"latency_ms":tot_ms,"cached":cached})

    n       = len(EVAL_SET)
    hits_1  = sum(1 for r in ranks if r==1)
    hits_3  = sum(1 for r in ranks if 0<r<=3)
    hits_5  = sum(1 for r in ranks if 0<r<=5)
    mrr     = calc_mrr(ranks)
    avg_ndcg = round(sum(ndcgs)/len(ndcgs),4) if ndcgs else 0
    avg_ent  = round(sum(entity_covs)/len(entity_covs),4) if entity_covs else 0

    # ── Section 2: Caching ─────────────────────────────────────────────────
    print("\n── Section 2: Caching Strategy ────────────────────────────")
    print("  Re-running first 5 queries to measure cache speedup...\n")

    cold_times, warm_times = [], []
    for item in EVAL_SET[:5]:
        match = next((r for r in results_log if r.get("id")==item["id"] and not r.get("cached")),None)
        if match:
            cold_times.append(match["latency_ms"])
        try:
            t0   = time.perf_counter()
            data = run_query(item["question"])
            wall = (time.perf_counter()-t0)*1000
            if data.get("cached"):
                warm_times.append(wall)
                print(f"  CACHE HIT  {wall:6.1f}ms  {item['question'][:52]}")
            else:
                print(f"  CACHE MISS {data.get('total_latency_ms',0):6.1f}ms  {item['question'][:52]}")
        except Exception as e:
            print(f"  ERROR: {e}")

    avg_cold = round(sum(cold_times)/len(cold_times),1) if cold_times else 0
    avg_warm = round(sum(warm_times)/len(warm_times),1) if warm_times else 0
    cache_pct = round((1-avg_warm/avg_cold)*100,1) if avg_cold>0 and avg_warm>0 else 0

    print(f"\n  Cold avg : {avg_cold}ms  |  Warm avg : {avg_warm}ms  |  Reduction : {cache_pct}%")

    # ── Section 3: Paraphrase Robustness ──────────────────────────────────
    print("\n── Section 3: Paraphrase Robustness ───────────────────────")

    para_results = []
    for item in PARAPHRASE_SET:
        orig     = next((r for r in results_log if r.get("id")==item["original_id"]),None)
        orig_hit = orig and orig.get("rank",0)>0
        try:
            data     = run_query(item["paraphrase"])
            para_hit = any(has_keyword(r["content"],item["keywords"]) for r in data.get("results",[])[:3])
        except Exception:
            para_hit = False
        consistent = orig_hit==para_hit
        status = "✓" if consistent else "✗"
        print(f"  {status} [{item['original_id']}] orig={'HIT' if orig_hit else 'MISS'}  para={'HIT' if para_hit else 'MISS'}  {item['paraphrase'][:52]}")
        para_results.append(consistent)

    para_score = round(sum(para_results)/len(para_results)*100,1)
    print(f"\n  Paraphrase Robustness Score: {para_score}%")

    # ── Section 4: Hallucination / False Positive Rate ────────────────────
    print("\n── Section 4: Hallucination / False Positive Rate ─────────")

    fp_count = 0
    for item in NEGATIVE_SET:
        try:
            data = run_query(item["question"])
            res  = data.get("results",[])
            if res and res[0].get("rerank_score",-999)>FALSE_POSITIVE_RERANK_THRESHOLD:
                fp_count += 1
                tag   = "✗ FP"
                score = res[0].get("rerank_score","n/a")
            else:
                tag   = "✓ OK"
                score = res[0].get("rerank_score","n/a") if res else "no results"
            print(f"  {tag}  rerank={score}  [{item['id']}] {item['question'][:52]}")
        except Exception as e:
            print(f"  ERROR [{item['id']}]: {e}")

    fpr              = round(fp_count/len(NEGATIVE_SET)*100,1)
    hallucination    = fpr
    print(f"\n  False Positive Rate: {fpr}%  ({fp_count}/{len(NEGATIVE_SET)})")
    print(f"  Hallucination Rate : {hallucination}%  (retrieval-level)")

    # ── Section 5: Latency ─────────────────────────────────────────────────
    if latencies:
        avg_tot = round(np.mean(latencies),1)
        p95     = round(np.percentile(latencies,95),1)
        p99     = round(np.percentile(latencies,99),1)
        avg_s1  = round(np.mean(s1_lats),1)
        avg_s2  = round(np.mean(s2_lats),1)
    else:
        avg_tot=p95=p99=avg_s1=avg_s2=0

    rerank_ok = (hits_3/n*100)>=90
    rerank_verdict = "YES — Top-3 ≥ 90% justifies reranking latency" if rerank_ok else \
                     "MARGINAL — consider ms-marco-MiniLM-L-2-v2 for speed"

    # ── Final Summary ──────────────────────────────────────────────────────
    print(f"""
══════════════════════════════════════════════════════════
  FINAL EVALUATION SUMMARY
══════════════════════════════════════════════════════════

  RETRIEVAL QUALITY  ({n} questions)
  ──────────────────────────────────────────
  Recall@1  (Top-1 Accuracy) : {hits_1}/{n}  ({hits_1/n*100:.1f}%)   target ≥ 75%  {'✓' if hits_1/n>=0.75 else '✗'}
  Recall@3  (Top-3 Accuracy) : {hits_3}/{n}  ({hits_3/n*100:.1f}%)   target ≥ 90%  {'✓' if hits_3/n>=0.90 else '✗'}
  Recall@5                   : {hits_5}/{n}  ({hits_5/n*100:.1f}%)
  MRR                        : {mrr}
  nDCG@{TOP_K}                    : {avg_ndcg}
  Entity Coverage Score      : {avg_ent}
  Paraphrase Robustness      : {para_score}%
  Hallucination Rate         : {hallucination}%
  False Positive Rate        : {fpr}%

  CACHING
  ──────────────────────────────────────────
  Cold avg  : {avg_cold}ms
  Warm avg  : {avg_warm}ms
  Reduction : {cache_pct}%

  LATENCY  (cold queries, n={len(latencies)})
  ──────────────────────────────────────────
  Stage 1 FAISS avg  : {avg_s1}ms
  Stage 2 Rerank avg : {avg_s2}ms
  Total avg          : {avg_tot}ms
  P95                : {p95}ms
  P99                : {p99}ms

  Reranking worth it? {rerank_verdict}

══════════════════════════════════════════════════════════
""")

    output = {
        "summary": {
            "recall_at_1": round(hits_1/n*100,1),
            "recall_at_3": round(hits_3/n*100,1),
            "recall_at_5": round(hits_5/n*100,1),
            "top1_accuracy": round(hits_1/n*100,1),
            "top3_accuracy": round(hits_3/n*100,1),
            "mrr": mrr,
            "ndcg_at_5": avg_ndcg,
            "entity_coverage_score": avg_ent,
            "paraphrase_robustness_pct": para_score,
            "hallucination_rate_pct": hallucination,
            "false_positive_rate_pct": fpr,
            "caching": {"cold_avg_ms":avg_cold,"warm_avg_ms":avg_warm,"reduction_pct":cache_pct},
            "latency": {"avg_ms":avg_tot,"p95_ms":p95,"p99_ms":p99,"avg_stage1_ms":avg_s1,"avg_stage2_ms":avg_s2},
        },
        "questions": results_log,
    }

    Path("eval_results.json").write_text(json.dumps(output,indent=2))
    print(f"  Results saved → eval_results.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload",  action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not check_server():
        sys.exit(1)
    if args.upload:
        upload_pdfs()
    run_eval(verbose=args.verbose)