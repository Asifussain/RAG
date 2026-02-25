"""
eval3.py — Iteration 3 of 3
─────────────────────────────
Covers: Section F (Paraphrase, 15 questions) + Section G (Negative, 10 questions)
        + Section H (Ambiguous/Clarification, 5 questions)
        + Caching benchmark
LLM calls: ~30  |  Est. tokens: ~45,000  |  Est. time: ~2 min (with 2.5s sleep)

HOW TO RUN:
  python eval3.py
  python eval3.py --retrieval  # skip LLM calls
  python eval3.py --verbose    # print answers

After all 3 eval files complete, run:
  python merge_eval_results.py
to get the combined final report.
"""

import os, math, json, argparse, sys, time
from pathlib import Path
import numpy as np
import requests

API   = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TOP_K = 5

# ══════════════════════════════════════════════════════════════════════════
# SECTION F — Paraphrase robustness (15 questions)
# ══════════════════════════════════════════════════════════════════════════
SECTION_F = [
    {"id":"F01","paraphrase":"Which of the three developments is a housing project rather than an office building?","original_keywords":["222 rajpur","residential","housing"]},
    {"id":"F02","paraphrase":"Identify the project that is not meant for commercial office use.","original_keywords":["222 rajpur","residential","housing"]},
    {"id":"F03","paraphrase":"Among the three properties, which one is exclusively residential in nature?","original_keywords":["222 rajpur","residential","housing"]},
    {"id":"F04","paraphrase":"Between Max Towers and Max House, which holds the higher level of LEED certification?","original_keywords":["leed","platinum","gold","max towers","max house"]},
    {"id":"F05","paraphrase":"If sustainability certification level is the deciding factor, which property ranks highest?","original_keywords":["leed","platinum","max towers"]},
    {"id":"F06","paraphrase":"Which development has achieved Platinum-level green certification?","original_keywords":["platinum","leed","max towers"]},
    {"id":"F07","paraphrase":"Which office property can employees walk to from a metro station?","original_keywords":["metro","walking","max house","max towers"]},
    {"id":"F08","paraphrase":"Identify the development located within walking distance of a metro stop.","original_keywords":["metro","walking","max house","max towers"]},
    {"id":"F09","paraphrase":"Between the Noida and Okhla projects, which one offers closer metro access?","original_keywords":["metro","max house","max towers","closer"]},
    {"id":"F10","paraphrase":"Which project is larger in overall constructed area: Max Towers or Max House?","original_keywords":["built-up","max towers","max house","larger"]},
    {"id":"F11","paraphrase":"Between the Delhi and Noida office developments, which spans more total square footage?","original_keywords":["built-up","max towers","max house"]},
    {"id":"F12","paraphrase":"Which property has the greater overall scale in terms of built-up space?","original_keywords":["built-up","max towers","max house"]},
    {"id":"F13","paraphrase":"Which property includes an indoor swimming facility?","original_keywords":["swimming pool","pool","max towers","max house"]},
    {"id":"F14","paraphrase":"Identify the development that provides decompression or relaxation spaces.","original_keywords":["decompression","relaxation","max towers","max house"]},
    {"id":"F15","paraphrase":"If employee wellness is a priority, which property explicitly supports it through facilities?","original_keywords":["wellness","fitness","max towers","max house"]},
]

# ══════════════════════════════════════════════════════════════════════════
# SECTION G — Negative / adversarial (10 questions — answers NOT in docs)
# ══════════════════════════════════════════════════════════════════════════
SECTION_G = [
    {"id":"G01","question":"Which property among the three includes a helipad?","reason":"No helipad mentioned"},
    {"id":"G02","question":"Which development offers a golf course within the premises?","reason":"No golf course mentioned"},
    {"id":"G03","question":"Is any of the properties located in Mumbai?","reason":"None in Mumbai"},
    {"id":"G04","question":"Which project provides co-living or serviced apartments?","reason":"Not mentioned"},
    {"id":"G05","question":"What is the rental yield percentage of Max Towers?","reason":"Rental yield not in documents"},
    {"id":"G06","question":"Which property has a shopping mall attached to it?","reason":"No shopping mall mentioned"},
    {"id":"G07","question":"Do any of the properties include a five-star hotel?","reason":"No hotel mentioned"},
    {"id":"G08","question":"Which development offers beachfront views?","reason":"None near beach"},
    {"id":"G09","question":"Is there a data center facility mentioned in any of the properties?","reason":"No data center mentioned"},
    {"id":"G10","question":"Which property includes an amusement park or entertainment zone?","reason":"Not mentioned"},
]

# ══════════════════════════════════════════════════════════════════════════
# SECTION H — Ambiguous / clarification test (5 questions)
# ══════════════════════════════════════════════════════════════════════════
SECTION_H = [
    {"id":"H01","question":"What is the total area?","clarification_keywords":["which property","clarify","specify","222 rajpur","max towers","max house","please clarify","are you referring"]},
    {"id":"H02","question":"How many floors does it have?","clarification_keywords":["which property","clarify","specify","222 rajpur","max towers","max house","please clarify","are you referring"]},
    {"id":"H03","question":"What certification does it hold?","clarification_keywords":["which property","clarify","specify","leed","rera","ukrera","please clarify","are you referring"]},
    {"id":"H04","question":"How far is it from the airport?","clarification_keywords":["which property","clarify","specify","jolly grant","igi","please clarify","are you referring"]},
    {"id":"H05","question":"Does it offer parking?","clarification_keywords":["which property","clarify","specify","parking","please clarify","are you referring"]},
]

FALSE_POSITIVE_THRESHOLD = -3.0

ABSTENTION_PHRASES = [
    "not available","not found","not mentioned","not in the",
    "no information","cannot find","do not","does not mention",
    "not provided","not specified","no reference","not stated",
    "unable to find","not present","not included","not described",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def check_server():
    try:
        r = requests.get(f"{API}/health",timeout=5); d=r.json()
        print(f"✓ Server online | Master: {d.get('master_index_ready')} | Cache: {d.get('cache',{}).get('enabled','?')}")
        return True
    except Exception as e:
        print(f"✗ {e}"); return False

def run_retrieval(q, top_k=TOP_K):
    r = requests.post(f"{API}/query/all",json={"question":q,"top_k":top_k},timeout=30)
    r.raise_for_status(); return r.json()

def run_answer(q, top_k=TOP_K):
    r = requests.post(f"{API}/answer",json={"question":q,"top_k":top_k},timeout=60)
    r.raise_for_status(); return r.json()

def has_kw(text, kws): return any(k.lower() in text.lower() for k in kws)
def print_section(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}")


def run_eval(retrieval_only=False, verbose=False):

    # ── Section F: Paraphrase Robustness ──────────────────────────────────
    print_section("Section F — Paraphrase Robustness")

    para_results = []
    para_latencies = []
    for item in SECTION_F:
        try:
            ret_data = run_retrieval(item["paraphrase"])
            chunks   = ret_data.get("results",[])
            ret_ms   = ret_data.get("stage1_latency_ms",0) + ret_data.get("stage2_latency_ms",0)
            if not ret_data.get("cached"): para_latencies.append(ret_ms)

            hit = any(has_kw(r["content"], item["original_keywords"]) for r in chunks[:3])
            para_results.append(hit)
            status = "✓" if hit else "✗"
            print(f"  {status} [{item['id']}] {'HIT' if hit else 'MISS'}  ret={ret_ms:.0f}ms  {item['paraphrase'][:55]}")
            if verbose and chunks:
                print(f"         top: {chunks[0]['content'][:80].replace(chr(10),' ')}")
        except Exception as e:
            print(f"  [{item['id']}] ERROR: {e}")
            para_results.append(False)

    para_score = round(sum(para_results)/len(para_results)*100,1) if para_results else 0
    print(f"\n  Paraphrase Robustness Score: {para_score}%  ({sum(para_results)}/{len(para_results)} consistent)")

    # ── Section G: Negative / Adversarial ─────────────────────────────────
    print_section("Section G — Negative / Adversarial")

    fp_count = 0
    hallucination_count = 0
    hal_total = 0

    for item in SECTION_G:
        try:
            ret_data = run_retrieval(item["question"])
            chunks   = ret_data.get("results",[])
            is_fp    = bool(chunks and chunks[0].get("rerank_score",-999) > FALSE_POSITIVE_THRESHOLD)
            if is_fp: fp_count += 1

            hal_text = ""
            if not retrieval_only:
                try:
                    ans_data = run_answer(item["question"])
                    raw = ans_data.get("answer","")
                    if raw.startswith("Generation error:"):
                        print(f"  [{item['id']}] ⚠ RATE LIMIT — skipping hallucination check")
                    else:
                        hal_text = raw.lower()
                        hal_total += 1
                        is_hal = len(hal_text)>50 and not any(p in hal_text for p in ABSTENTION_PHRASES)
                        if is_hal: hallucination_count += 1
                        if not ans_data.get("cached"): time.sleep(2.5)
                except Exception as e:
                    print(f"  [{item['id']}] LLM ERROR: {e}")

            fp_tag  = "FP" if is_fp else "OK"
            score   = f"{chunks[0].get('rerank_score',0):.2f}" if chunks else "n/a"
            hal_tag = ""
            if hal_text:
                is_hal = len(hal_text)>50 and not any(p in hal_text for p in ABSTENTION_PHRASES)
                hal_tag = "  llm=HAL" if is_hal else "  llm=OK"
            print(f"  [{item['id']}] retrieval={fp_tag}  rerank={score}{hal_tag}  {item['question'][:50]}")
            if verbose and hal_text: print(f"         → {hal_text[:100]}")

        except Exception as e:
            print(f"  [{item['id']}] ERROR: {e}")

    fpr      = round(fp_count/len(SECTION_G)*100,1)
    hal_rate = round(hallucination_count/hal_total*100,1) if hal_total>0 else 0
    print(f"\n  False Positive Rate : {fpr}%  ({fp_count}/{len(SECTION_G)})")
    print(f"  Hallucination Rate  : {hal_rate}%  ({hallucination_count}/{hal_total} answered questions)")

    # ── Section H: Ambiguous / Clarification ──────────────────────────────
    print_section("Section H — Ambiguous / Clarification")

    clarification_hits = 0
    clarification_total = 0

    for item in SECTION_H:
        try:
            if retrieval_only:
                data = run_retrieval(item["question"])
                answer_text = " ".join(r["content"] for r in data.get("results",[])[:2]).lower()
            else:
                ans_data = run_answer(item["question"])
                raw = ans_data.get("answer","")
                if raw.startswith("Generation error:"):
                    print(f"  [SKIP] [{item['id']}] rate limit error — not counted")
                    continue
                answer_text = raw.lower()
                if not ans_data.get("cached"): time.sleep(2.5)

            clarification_total += 1
            triggered = has_kw(answer_text, item["clarification_keywords"])
            if triggered: clarification_hits += 1
            status = "✓" if triggered else "✗"
            print(f"  {status} [{item['id']}] clarification={'triggered' if triggered else 'NOT triggered'}  {item['question']}")
            if verbose: print(f"         → {answer_text[:120]}")
        except Exception as e:
            print(f"  [{item['id']}] ERROR: {e}")

    clarification_score = round(clarification_hits/clarification_total*100,1) if clarification_total>0 else 0
    print(f"\n  Clarification Score : {clarification_score}%  ({clarification_hits}/{clarification_total})")

    # ── Caching benchmark ──────────────────────────────────────────────────
    print_section("Caching Benchmark")
    print("  Re-running 5 questions to measure warm cache speedup...\n")

    cold_times, warm_times = [], []
    sample_questions = [item["paraphrase"] for item in SECTION_F[:5]]

    for q in sample_questions:
        try:
            t0   = time.perf_counter()
            data = run_retrieval(q)
            wall = (time.perf_counter()-t0)*1000
            if data.get("cached"):
                warm_times.append(wall)
                print(f"  CACHE HIT  {wall:6.1f}ms  {q[:55]}")
            else:
                cold_times.append(wall)
                print(f"  COLD       {wall:6.1f}ms  {q[:55]}")
        except Exception as e:
            print(f"  ERROR: {e}")

    avg_cold = round(sum(cold_times)/len(cold_times),1) if cold_times else 0
    avg_warm = round(sum(warm_times)/len(warm_times),1) if warm_times else 0
    cache_pct = round((1-avg_warm/avg_cold)*100,1) if avg_cold>0 and avg_warm>0 else 0
    print(f"\n  Cold avg : {avg_cold}ms  |  Warm avg : {avg_warm}ms  |  Reduction : {cache_pct}%")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"""
{'═'*60}
  EVAL 3/3 SUMMARY  (Sections F + G + H)
{'═'*60}
  Paraphrase Robustness : {para_score}%
  False Positive Rate   : {fpr}%
  Hallucination Rate    : {hal_rate}%  (retrieval-level, answered questions only)
  Clarification Score   : {clarification_score}%
  Cache Reduction       : {cache_pct}%  (cold: {avg_cold}ms → warm: {avg_warm}ms)
{'═'*60}
""")

    output = {
        "eval_file": "eval3", "sections": "F + G + H",
        "summary": {
            "paraphrase_robustness_pct": para_score,
            "false_positive_rate_pct": fpr,
            "hallucination_rate_pct": hal_rate,
            "clarification_score_pct": clarification_score,
            "caching": {"cold_avg_ms": avg_cold, "warm_avg_ms": avg_warm, "reduction_pct": cache_pct},
        },
        "section_f": [{"id":item["id"],"paraphrase":item["paraphrase"],"hit":hit} for item,hit in zip(SECTION_F,para_results)],
    }
    Path("eval3_results.json").write_text(json.dumps(output,indent=2))
    print(f"  Saved → eval3_results.json\n  Now run: python merge_eval_results.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval 3/3 — Sections F + G + H (~30 LLM calls, ~45k tokens)")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()
    if not check_server(): sys.exit(1)
    run_eval(retrieval_only=args.retrieval, verbose=args.verbose)