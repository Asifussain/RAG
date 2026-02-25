"""
eval2.py — Iteration 2 of 3
─────────────────────────────
Covers: Section C (Max House, 13 questions) + Section D (Cross-property, 15 questions)
        + Section E (Client Simulation, 9 questions)

HOW TO RUN:
  If you have no token limit   → run eval1.py, eval2.py, eval3.py back to back
  If you have a 100k TPD limit → run one file per API key

  python eval2.py
  python eval2.py --retrieval  # skip LLM calls
  python eval2.py --verbose    # print answers

Results saved to eval2_results.json
"""

import os, math, json, argparse, sys, time
from pathlib import Path
import numpy as np
import requests

API   = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TOP_K = 5

# ══════════════════════════════════════════════════════════════════════════
# SECTION C — Max House, Okhla (13 questions)
# ══════════════════════════════════════════════════════════════════════════
SECTION_C = [
    {"id":"C01","question":"What is the total super built-up area of Max House, Okhla?","keywords":["super built-up","sq. ft","sq ft","max house"],"entities":["super built-up","max house"]},
    {"id":"C02","question":"How many tenant floors are there in Max House?","keywords":["tenant","floors","max house"],"entities":["floors","max house"]},
    {"id":"C03","question":"What is the typical floor plate size at Max House?","keywords":["floor plate","sq. ft","sq ft","max house"],"entities":["floor plate"]},
    {"id":"C04","question":"What is the green rating of Max House?","keywords":["leed","green","gold","platinum","rating","certification"],"entities":["leed"]},
    {"id":"C05","question":"How far is Max House, Okhla from the Okhla NSIC Metro Station?","keywords":["nsic","metro","okhla","distance","km","minutes","walking"],"entities":["nsic","metro"]},
    {"id":"C06","question":"How far is Max House from IGI Airport?","keywords":["igi","airport","distance","km","minutes"],"entities":["igi","airport"]},
    {"id":"C07","question":"Is Max House within walking distance of a metro station?","keywords":["metro","walking","distance","station"],"entities":["metro"]},
    {"id":"C08","question":"What facade material is used in Max House?","keywords":["facade","glass","material","glazing","aluminium"],"entities":["facade"]},
    {"id":"C09","question":"What is the floor-to-ceiling height at Max House?","keywords":["floor-to-ceiling","height","metres","feet"],"entities":["floor-to-ceiling","height"]},
    {"id":"C10","question":"Does Max House use double-glazed windows?","keywords":["double-glazed","double glazed","glazing","windows"],"entities":["double-glazed"]},
    {"id":"C11","question":"What air treatment technology is used in Max House?","keywords":["air treatment","filtration","purification","hepa","air quality","ionization"],"entities":["air treatment"]},
    {"id":"C12","question":"Is Max House LEED certified?","keywords":["leed","certified","certification","green"],"entities":["leed"]},
    {"id":"C13","question":"Does Max House incorporate biophilic design principles?","keywords":["biophilic","nature","green wall","plants","landscape"],"entities":["biophilic"]},
]

# ══════════════════════════════════════════════════════════════════════════
# SECTION D — Cross-property comparison (15 questions)
# ══════════════════════════════════════════════════════════════════════════
SECTION_D = [
    {"id":"D01","question":"Which property among 222 Rajpur, Max Towers, and Max House is purely residential?","keywords":["222 rajpur","residential","housing"],"entities":["222 rajpur","residential"]},
    {"id":"D02","question":"Which property is located in Dehradun: 222 Rajpur or Max Towers?","keywords":["222 rajpur","dehradun"],"entities":["222 rajpur","dehradun"]},
    {"id":"D03","question":"Between Max Towers and Max House, which one has a higher LEED certification?","keywords":["leed","platinum","gold","max towers","max house"],"entities":["leed","platinum"]},
    {"id":"D04","question":"Compare the typical floor plate size of Max Towers and Max House.","keywords":["floor plate","max towers","max house","sq. ft","sq ft"],"entities":["floor plate"]},
    {"id":"D05","question":"Which property has a larger total built-up area: Max Towers or Max House?","keywords":["built-up","max towers","max house","larger"],"entities":["built-up"]},
    {"id":"D06","question":"Which property has more tenant floors: Max Towers or Max House?","keywords":["floors","tenant","max towers","max house"],"entities":["floors"]},
    {"id":"D07","question":"Which property is closer to a metro station: Max House or Max Towers?","keywords":["metro","walking","max house","max towers","closer"],"entities":["metro"]},
    {"id":"D08","question":"Between 222 Rajpur and Max House, which property is closer to an airport?","keywords":["airport","222 rajpur","max house","closer","distance"],"entities":["airport"]},
    {"id":"D09","question":"Which property offers direct access to the DND Flyway: Max Towers or Max House?","keywords":["dnd","flyway","max towers","max house"],"entities":["dnd"]},
    {"id":"D10","question":"Which property has LEED Platinum certification: Max Towers or Max House?","keywords":["leed","platinum","max towers","max house"],"entities":["leed","platinum"]},
    {"id":"D11","question":"Which properties use advanced air treatment systems: Max Towers, Max House, or both?","keywords":["air treatment","max towers","max house","both"],"entities":["air treatment"]},
    {"id":"D12","question":"Which property offers on-site wastewater treatment: Max Towers or Max House?","keywords":["wastewater","treatment","max towers","max house"],"entities":["wastewater"]},
    {"id":"D13","question":"Which property offers a swimming pool: Max Towers or Max House?","keywords":["swimming pool","pool","max towers","max house"],"entities":["swimming pool"]},
    {"id":"D14","question":"Does 222 Rajpur offer wellness amenities comparable to Max Towers?","keywords":["wellness","amenity","222 rajpur","max towers"],"entities":["wellness"]},
    {"id":"D15","question":"Which property explicitly mentions decompression spaces: Max Towers or Max House?","keywords":["decompression","relaxation","spaces","max towers","max house"],"entities":["decompression"]},
]

# ══════════════════════════════════════════════════════════════════════════
# SECTION E — Real-world client simulation (9 questions)
# ══════════════════════════════════════════════════════════════════════════
SECTION_E = [
    {"id":"E01","question":"I'm looking for a 4-bedroom villa with staff accommodation in 222 Rajpur — which unit type should I consider?","keywords":["courtyard villa","forest villa","staff","accommodation","bedroom"],"entities":["staff","villa"]},
    {"id":"E02","question":"My company needs a 25,000 sq. ft. office in Noida — can Max Towers accommodate this on a single floor?","keywords":["floor plate","max towers","sq. ft","single floor","25000"],"entities":["floor plate","max towers"]},
    {"id":"E03","question":"We want an office in Delhi with LEED Gold certification — is Max House, Okhla suitable?","keywords":["leed","gold","max house","delhi","okhla"],"entities":["leed","gold","max house"]},
    {"id":"E04","question":"I want a residential property near the forest with private garden space — does 222 Rajpur offer this?","keywords":["forest","garden","private","222 rajpur","residential"],"entities":["forest","garden"]},
    {"id":"E05","question":"We are a wellness-focused company — between Max Towers and Max House, which better supports employee wellbeing?","keywords":["wellness","fitness","pool","max towers","max house","wellbeing"],"entities":["wellness"]},
    {"id":"E06","question":"I need an office within walking distance of the metro in Delhi — is Max House a good option?","keywords":["metro","walking","max house","delhi"],"entities":["metro","max house"]},
    {"id":"E07","question":"Which property among 222 Rajpur, Max Towers, and Max House offers private elevators?","keywords":["elevator","private","222 rajpur","forest villa"],"entities":["elevator","private"]},
    {"id":"E08","question":"If sustainability is a top priority, should I choose Max Towers or Max House?","keywords":["leed","platinum","gold","sustainability","max towers","max house"],"entities":["leed","sustainability"]},
    {"id":"E09","question":"I need a property with daycare facilities — which of these three properties provides that?","keywords":["daycare","creche","childcare","max towers"],"entities":["daycare"]},
]


# Helpers

def check_server():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        d = r.json()
        print(f"✓ Server online | Master: {d.get('master_index_ready')} | Cache: {d.get('cache',{}).get('enabled','?')}")
        return True
    except Exception as e:
        print(f"✗ Server not reachable: {e}"); return False

def run_retrieval(q, top_k=TOP_K):
    r = requests.post(f"{API}/query/all",json={"question":q,"top_k":top_k},timeout=30)
    r.raise_for_status(); return r.json()

def run_answer(q, top_k=TOP_K):
    r = requests.post(f"{API}/answer",json={"question":q,"top_k":top_k},timeout=60)
    r.raise_for_status(); return r.json()

def has_kw(text, kws): return any(k.lower() in text.lower() for k in kws)
def hit_rank(results, kws):
    for i,r in enumerate(results,1):
        if has_kw(r["content"],kws): return i
    return 0
def calc_mrr(ranks): return round(sum(1.0/r for r in ranks if r>0)/len(ranks),4) if ranks else 0.0
def calc_ndcg(results, kws, k=5):
    gains=[1.0 if has_kw(results[i]["content"],kws) else 0.0 for i in range(min(k,len(results)))]
    dcg=sum(g/math.log2(i+2) for i,g in enumerate(gains))
    idcg=sum(1.0/math.log2(i+2) for i in range(int(sum(gains))))
    return round(dcg/idcg,4) if idcg>0 else 0.0
def calc_entity_coverage(ans, ents):
    if not ents or not ans: return 0.0
    return round(sum(1 for e in ents if e.lower() in ans.lower())/len(ents),4)
def print_section(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}")


def run_eval(retrieval_only=False, verbose=False):
    all_sections = [
        ("Section C — Max House",          SECTION_C),
        ("Section D — Cross-property",     SECTION_D),
        ("Section E — Client Simulation",  SECTION_E),
    ]

    retrieval_latencies,s1_lats,s2_lats = [],[],[]
    generation_latencies,total_latencies = [],[]
    ranks,ndcgs,entity_covs = [],[],[]
    results_log = []

    for section_name, section in all_sections:
        print_section(section_name)
        for item in section:
            try:
                ret_data = run_retrieval(item["question"])
            except Exception as e:
                print(f"  [{item['id']}] RETRIEVAL ERROR: {e}"); continue

            chunks  = ret_data.get("results",[])
            s1_ms   = ret_data.get("stage1_latency_ms",0)
            s2_ms   = ret_data.get("stage2_latency_ms",0)
            ret_ms  = s1_ms+s2_ms
            if not ret_data.get("cached"):
                retrieval_latencies.append(ret_ms); s1_lats.append(s1_ms); s2_lats.append(s2_ms)

            rank = hit_rank(chunks, item["keywords"])
            ndcg = calc_ndcg(chunks, item["keywords"])
            ranks.append(rank); ndcgs.append(ndcg)

            answer_text,gen_ms,entity_cov = "",0.0,0.0
            if not retrieval_only:
                try:
                    ans_data = run_answer(item["question"])
                    raw = ans_data.get("answer","")
                    if raw.startswith("Generation error:"):
                        print(f"  [{item['id']}] ⚠ RATE LIMIT — skipping LLM metric")
                    else:
                        answer_text = raw
                        gen_ms  = ans_data.get("generation_ms",0)
                        tot_ms  = ans_data.get("total_ms",0)
                        if not ans_data.get("cached"):
                            generation_latencies.append(gen_ms)
                            total_latencies.append(tot_ms)
                            time.sleep(2.6)
                    entity_cov = calc_entity_coverage(answer_text, item.get("entities",[]))
                except Exception as e:
                    print(f"  [{item['id']}] LLM ERROR: {e}")

            entity_covs.append(entity_cov)
            top1=rank==1; top3=0<rank<=3
            status="✓" if top1 else ("~" if top3 else "✗")
            label="TOP-1" if top1 else ("TOP-3" if top3 else "MISS ")
            print(f"  [{item['id']}] {status} {label}  ret={ret_ms:5.0f}ms  gen={gen_ms:5.0f}ms  nDCG={ndcg:.2f}  ent={entity_cov:.2f}  {item['question'][:50]}")
            if verbose and answer_text: print(f"         → {answer_text[:120]}")

            results_log.append({
                "id":item["id"],"section":section_name,"question":item["question"],
                "rank":rank,"ndcg":ndcg,"entity_coverage":entity_cov,
                "retrieval_ms":round(ret_ms,2),"generation_ms":round(gen_ms,2),
                "answer":answer_text[:300],
            })

    n=len(ranks)
    hits_1=sum(1 for r in ranks if r==1)
    hits_3=sum(1 for r in ranks if 0<r<=3)
    hits_5=sum(1 for r in ranks if 0<r<=5)
    mrr=calc_mrr(ranks)
    avg_ndcg=round(sum(ndcgs)/len(ndcgs),4) if ndcgs else 0
    avg_ent=round(sum(entity_covs)/len(entity_covs),4) if entity_covs else 0
    avg_s1=round(sum(s1_lats)/len(s1_lats),1) if s1_lats else 0
    avg_s2=round(sum(s2_lats)/len(s2_lats),1) if s2_lats else 0
    avg_ret=round(sum(retrieval_latencies)/len(retrieval_latencies),1) if retrieval_latencies else 0
    p95_ret=round(np.percentile(retrieval_latencies,95),1) if retrieval_latencies else 0
    avg_gen=round(sum(generation_latencies)/len(generation_latencies),1) if generation_latencies else 0
    avg_tot=round(sum(total_latencies)/len(total_latencies),1) if total_latencies else 0
    p95_tot=round(np.percentile(total_latencies,95),1) if total_latencies else 0
    p99_tot=round(np.percentile(total_latencies,99),1) if total_latencies else 0

    print(f"""
{'═'*60}
  EVAL 2/3 SUMMARY  (Sections C + D + E, {n} questions)
{'═'*60}
  Recall@1 : {hits_1}/{n}  ({hits_1/n*100:.1f}%)   target ≥ 75%  {'✓' if hits_1/n>=0.75 else '✗'}
  Recall@3 : {hits_3}/{n}  ({hits_3/n*100:.1f}%)   target ≥ 90%  {'✓' if hits_3/n>=0.90 else '✗'}
  Recall@5 : {hits_5}/{n}  ({hits_5/n*100:.1f}%)
  MRR      : {mrr}  |  nDCG@5 : {avg_ndcg}  |  Entity Coverage : {avg_ent}
  Retrieval avg : {avg_ret}ms (P95: {p95_ret}ms)  |  Generation avg : {avg_gen}ms  |  Total avg : {avg_tot}ms
{'═'*60}
""")

    output = {
        "eval_file":"eval2","sections":"C + D + E",
        "summary":{
            "recall_at_1":round(hits_1/n*100,1),"recall_at_3":round(hits_3/n*100,1),
            "recall_at_5":round(hits_5/n*100,1),"mrr":mrr,"ndcg_at_5":avg_ndcg,
            "entity_coverage_score":avg_ent,
            "latency":{"avg_stage1_ms":avg_s1,"avg_stage2_ms":avg_s2,
                "avg_retrieval_ms":avg_ret,"p95_retrieval_ms":p95_ret,
                "avg_generation_ms":avg_gen,"avg_total_ms":avg_tot,
                "p95_total_ms":p95_tot,"p99_total_ms":p99_tot},
        },
        "questions":results_log,
    }
    Path("eval2_results.json").write_text(json.dumps(output,indent=2))
    print(f"  Saved → eval2_results.json\n  Next: run eval3.py  (Sections F + G + H)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval 2/3 — Sections C + D + E (~37 LLM calls, ~55k tokens)")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()
    if not check_server(): sys.exit(1)
    run_eval(retrieval_only=args.retrieval, verbose=args.verbose)