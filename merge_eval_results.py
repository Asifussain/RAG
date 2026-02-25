"""
merge_eval_results.py — Combine eval1 + eval2 + eval3 into final report
------------------------------------------------------------------------
Run after all three eval files complete:
    python merge_eval_results.py

Outputs: eval_final_results.json
"""

import json
from pathlib import Path

def load(f):
    p = Path(f)
    if not p.exists():
        print(f"  ⚠ {f} not found — skipping")
        return None
    return json.loads(p.read_text())

e1 = load("eval1_results.json")
e2 = load("eval2_results.json")
e3 = load("eval3_results.json")

if not any([e1, e2, e3]):
    print("No eval result files found. Run eval1.py, eval2.py, eval3.py first.")
    raise SystemExit

all_questions = []
if e1: all_questions.extend(e1.get("questions",[]))
if e2: all_questions.extend(e2.get("questions",[]))

n = len(all_questions)
hits_1 = sum(1 for q in all_questions if q.get("rank")==1)
hits_3 = sum(1 for q in all_questions if 0<q.get("rank",0)<=3)
hits_5 = sum(1 for q in all_questions if 0<q.get("rank",0)<=5)

ranks = [q.get("rank",0) for q in all_questions]
reciprocals = [1.0/r for r in ranks if r>0]
mrr = round(sum(reciprocals)/len(ranks),4) if ranks else 0

ndcgs = [q.get("ndcg",0) for q in all_questions]
avg_ndcg = round(sum(ndcgs)/len(ndcgs),4) if ndcgs else 0

ents = [q.get("entity_coverage",0) for q in all_questions]
avg_ent = round(sum(ents)/len(ents),4) if ents else 0

def avg_latency(key, *sources):
    vals = [s["summary"]["latency"][key] for s in sources if s and s["summary"]["latency"].get(key,0)>0]
    return round(sum(vals)/len(vals),1) if vals else 0

sources_12 = [s for s in [e1,e2] if s]

para_score    = e3["summary"]["paraphrase_robustness_pct"] if e3 else "n/a"
fpr           = e3["summary"]["false_positive_rate_pct"] if e3 else "n/a"
hal_rate      = e3["summary"]["hallucination_rate_pct"] if e3 else "n/a"
clar_score    = e3["summary"]["clarification_score_pct"] if e3 else "n/a"
cold_avg      = e3["summary"]["caching"]["cold_avg_ms"] if e3 else 0
warm_avg      = e3["summary"]["caching"]["warm_avg_ms"] if e3 else 0
cache_pct     = e3["summary"]["caching"]["reduction_pct"] if e3 else 0

print(f"""
{'═'*60}
  FINAL COMBINED EVALUATION REPORT
{'═'*60}

  RETRIEVAL QUALITY  ({n} questions, Sections A–E)
  ──────────────────────────────────────────────
  Recall@1  : {hits_1}/{n}  ({hits_1/n*100:.1f}%)   target ≥ 75%  {'✓' if n>0 and hits_1/n>=0.75 else '✗'}
  Recall@3  : {hits_3}/{n}  ({hits_3/n*100:.1f}%)   target ≥ 90%  {'✓' if n>0 and hits_3/n>=0.90 else '✗'}
  Recall@5  : {hits_5}/{n}  ({hits_5/n*100:.1f}%)
  MRR       : {mrr}
  nDCG@5    : {avg_ndcg}
  Entity Coverage Score : {avg_ent}

  ROBUSTNESS & QUALITY
  ──────────────────────────────────────────────
  Paraphrase Robustness : {para_score}%
  Hallucination Rate    : {hal_rate}%
  False Positive Rate   : {fpr}%
  Clarification Score   : {clar_score}%

  CACHING
  ──────────────────────────────────────────────
  Cold avg  : {cold_avg}ms  |  Warm avg : {warm_avg}ms  |  Reduction : {cache_pct}%

  LATENCY
  ──────────────────────────────────────────────
  Stage 1 FAISS avg  : {avg_latency('avg_stage1_ms', *sources_12)}ms
  Stage 2 Rerank avg : {avg_latency('avg_stage2_ms', *sources_12)}ms
  Retrieval avg      : {avg_latency('avg_retrieval_ms', *sources_12)}ms  (P95: {avg_latency('p95_retrieval_ms', *sources_12)}ms)
  Generation avg     : {avg_latency('avg_generation_ms', *sources_12)}ms
  Total avg          : {avg_latency('avg_total_ms', *sources_12)}ms  (P95: {avg_latency('p95_total_ms', *sources_12)}ms)

{'═'*60}
""")

final = {
    "summary": {
        "recall_at_1": round(hits_1/n*100,1) if n else 0,
        "recall_at_3": round(hits_3/n*100,1) if n else 0,
        "recall_at_5": round(hits_5/n*100,1) if n else 0,
        "mrr": mrr, "ndcg_at_5": avg_ndcg,
        "entity_coverage_score": avg_ent,
        "paraphrase_robustness_pct": para_score,
        "hallucination_rate_pct": hal_rate,
        "false_positive_rate_pct": fpr,
        "clarification_score_pct": clar_score,
        "caching": {"cold_avg_ms": cold_avg, "warm_avg_ms": warm_avg, "reduction_pct": cache_pct},
        "latency": {
            "avg_stage1_ms":    avg_latency("avg_stage1_ms",    *sources_12),
            "avg_stage2_ms":    avg_latency("avg_stage2_ms",    *sources_12),
            "avg_retrieval_ms": avg_latency("avg_retrieval_ms", *sources_12),
            "p95_retrieval_ms": avg_latency("p95_retrieval_ms", *sources_12),
            "avg_generation_ms":avg_latency("avg_generation_ms",*sources_12),
            "avg_total_ms":     avg_latency("avg_total_ms",     *sources_12),
            "p95_total_ms":     avg_latency("p95_total_ms",     *sources_12),
        },
    },
    "questions": all_questions,
}

Path("eval_final_results.json").write_text(json.dumps(final,indent=2))
print("  Saved → eval_final_results.json")