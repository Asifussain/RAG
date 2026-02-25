"""
eval1.py — Iteration 1 of 3
─────────────────────────────
Covers: Section A (222 Rajpur, 25 questions) + Section B (Max Towers, 18 questions)

HOW TO RUN:
  If you have no token limit   → run eval1.py, eval2.py, eval3.py back to back
  If you have a 100k TPD limit → run one file per API key

  python eval1.py              # runs eval, PDFs must already be indexed
  python eval1.py --upload     # uploads all 4 PDFs first
  python eval1.py --retrieval  # retrieval metrics only, no LLM calls
  python eval1.py --verbose    # print answers per question

Results saved to eval1_results.json
Merge all 3 with: python merge_eval_results.py
"""

import os, math, json, argparse, sys, time
from pathlib import Path
import numpy as np
import requests

API   = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TOP_K = 5

PDF_FILES = [
    "222-rajpur-brochure.pdf",
    "max-towers-brochure.pdf",
    "max-house-brochure.pdf",
]

# ══════════════════════════════════════════════════════════════════════════
# SECTION A — 222 Rajpur, Dehradun (25 questions)
# ══════════════════════════════════════════════════════════════════════════
SECTION_A = [
    {"id":"A01","question":"For 222 Rajpur, Dehradun, how many total residences are planned and over how many acres is the project spread?","keywords":["residences","acres","total"],"entities":["222 Rajpur","acres"]},
    {"id":"A02","question":"At 222 Rajpur, what types of residences are available?","keywords":["townhouse","courtyard villa","forest villa","villa"],"entities":["townhouse","villa"]},
    {"id":"A03","question":"Is 222 Rajpur adjacent to any forest area? If yes, which one?","keywords":["forest","adjacent","rajaji","reserve"],"entities":["forest"]},
    {"id":"A04","question":"What are the views offered from residences at 222 Rajpur, Dehradun?","keywords":["view","mountain","himalaya","valley","forest","nature"],"entities":["view"]},
    {"id":"A05","question":"How far is Jolly Grant Airport from 222 Rajpur, Dehradun?","keywords":["jolly grant","airport","km","minutes","distance"],"entities":["jolly grant","airport"]},
    {"id":"A06","question":"What is the distance between 222 Rajpur and The Doon School?","keywords":["doon school","km","minutes","distance"],"entities":["doon school"]},
    {"id":"A07","question":"How long does it take to reach Pacific Mall from 222 Rajpur?","keywords":["pacific mall","minutes","km","distance"],"entities":["pacific mall"]},
    {"id":"A08","question":"Is 222 Rajpur close to Max Super Specialty Hospital?","keywords":["max","hospital","specialty","km","minutes"],"entities":["hospital"]},
    {"id":"A09","question":"How many Townhouse units are available at 222 Rajpur?","keywords":["townhouse","units","number"],"entities":["townhouse"]},
    {"id":"A10","question":"What is the built-up area and carpet area of a Townhouse at 222 Rajpur?","keywords":["built-up","carpet area","townhouse","sq. ft","sq ft"],"entities":["carpet area","townhouse"]},
    {"id":"A11","question":"Does the Townhouse at 222 Rajpur include a sky court?","keywords":["sky court","atrium","skylight","natural light"],"entities":["sky court"]},
    {"id":"A12","question":"What is the ceiling height in the Townhouse units at 222 Rajpur?","keywords":["ceiling","height","feet","metres","ft"],"entities":["ceiling"]},
    {"id":"A13","question":"How many parking spaces are provided with each Townhouse at 222 Rajpur?","keywords":["parking","spaces","car","garage"],"entities":["parking"]},
    {"id":"A14","question":"How many Courtyard Villas are available at 222 Rajpur?","keywords":["courtyard","villa","units","number"],"entities":["courtyard"]},
    {"id":"A15","question":"What is the plot size range for Courtyard Villas at 222 Rajpur?","keywords":["plot","sq. ft","sq ft","courtyard"],"entities":["plot","courtyard"]},
    {"id":"A16","question":"Do the Courtyard Villas at 222 Rajpur include staff accommodation?","keywords":["staff","accommodation","quarters","servant"],"entities":["staff"]},
    {"id":"A17","question":"What is the terrace size of a Courtyard Villa at 222 Rajpur?","keywords":["terrace","sq. ft","sq ft","courtyard"],"entities":["terrace"]},
    {"id":"A18","question":"How many Forest Villas are available at 222 Rajpur?","keywords":["forest villa","units","number"],"entities":["forest villa"]},
    {"id":"A19","question":"What is the built-up area of a Forest Villa at 222 Rajpur?","keywords":["built-up","forest villa","sq. ft","sq ft"],"entities":["forest villa","built-up"]},
    {"id":"A20","question":"Do Forest Villas at 222 Rajpur have private elevators?","keywords":["elevator","lift","private"],"entities":["elevator"]},
    {"id":"A21","question":"What special landscape feature is included in the lower ground floor of Forest Villas at 222 Rajpur?","keywords":["landscape","garden","lower ground","ground floor","botanical"],"entities":["landscape","ground floor"]},
    {"id":"A22","question":"Does 222 Rajpur provide round-the-clock security?","keywords":["security","24","cctv","guard","surveillance"],"entities":["security"]},
    {"id":"A23","question":"What wellness or nature-focused amenities are offered at 222 Rajpur?","keywords":["wellness","nature","amenity","pool","garden","orchard"],"entities":["wellness"]},
    {"id":"A24","question":"Does 222 Rajpur offer power backup and uninterrupted water supply?","keywords":["power backup","water supply","uninterrupted","generator"],"entities":["power backup","water"]},
    {"id":"A25","question":"Is there a private orchard at 222 Rajpur?","keywords":["orchard","private","botanical","fruit"],"entities":["orchard"]},
]

# ══════════════════════════════════════════════════════════════════════════
# SECTION B — Max Towers, Noida (18 questions)
# ══════════════════════════════════════════════════════════════════════════
SECTION_B = [
    {"id":"B01","question":"What is the total super built-up area of Max Towers, Noida?","keywords":["super built-up","sq. ft","sq ft","max towers"],"entities":["super built-up","max towers"]},
    {"id":"B02","question":"How many office floors and amenity floors are there in Max Towers?","keywords":["office floors","amenity","floors","max towers"],"entities":["floors","max towers"]},
    {"id":"B03","question":"What is the typical floor plate size at Max Towers?","keywords":["floor plate","sq. ft","sq ft","max towers"],"entities":["floor plate"]},
    {"id":"B04","question":"What is the floor-to-floor height at Max Towers?","keywords":["floor-to-floor","height","metres","feet","max towers"],"entities":["floor-to-floor","height"]},
    {"id":"B05","question":"What green rating has Max Towers achieved?","keywords":["leed","green","platinum","gold","rating","certification"],"entities":["leed"]},
    {"id":"B06","question":"Does Max Towers offer on-site wastewater treatment?","keywords":["wastewater","treatment","sewage","recycl"],"entities":["wastewater"]},
    {"id":"B07","question":"What is the coefficient of performance (COP) of the chiller system at Max Towers?","keywords":["cop","coefficient","chiller","performance"],"entities":["cop","chiller"]},
    {"id":"B08","question":"Does Max Towers support electric vehicle parking?","keywords":["electric vehicle","ev","charging","parking"],"entities":["electric vehicle"]},
    {"id":"B09","question":"Does Max Towers have a swimming pool?","keywords":["swimming pool","pool","aqua"],"entities":["swimming pool"]},
    {"id":"B10","question":"What kind of fitness facilities are available at Max Towers?","keywords":["fitness","gym","wellness","sport","exercise"],"entities":["fitness"]},
    {"id":"B11","question":"Does Max Towers provide daycare facilities?","keywords":["daycare","creche","childcare","child"],"entities":["daycare"]},
    {"id":"B12","question":"What air treatment system is used in Max Towers?","keywords":["air treatment","filtration","purification","hepa","ionization","air quality"],"entities":["air treatment"]},
    {"id":"B13","question":"Where is Max Towers located?","keywords":["sector","noida","location","address"],"entities":["noida"]},
    {"id":"B14","question":"Is Max Towers within walking distance of a metro station?","keywords":["metro","walking","distance","station"],"entities":["metro"]},
    {"id":"B15","question":"Does Max Towers have direct access to the DND Flyway?","keywords":["dnd","flyway","access","expressway"],"entities":["dnd"]},
    {"id":"B16","question":"What type of façade glass is used in Max Towers?","keywords":["facade","glass","glazing","double","triple"],"entities":["facade","glass"]},
    {"id":"B17","question":"What is the solar heat gain coefficient of the façade at Max Towers?","keywords":["solar heat gain","shgc","coefficient","facade"],"entities":["solar heat gain"]},
    {"id":"B18","question":"What percentage of regular occupied space at Max Towers gets line-of-sight to the outside?","keywords":["line-of-sight","outside","occupied","percentage","views"],"entities":["line-of-sight"]},
]

FALSE_POSITIVE_THRESHOLD = -3.0


#  Helpers 

def check_server():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        d = r.json()
        print(f"✓ Server online | Master: {d.get('master_index_ready')} | Cache: {d.get('cache',{}).get('enabled','?')}")
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
            print(f"  ✗ {pdf} — {r.text[:100]}")
    print()


def run_retrieval(question, top_k=TOP_K):
    r = requests.post(f"{API}/query/all",json={"question":question,"top_k":top_k},timeout=30)
    r.raise_for_status()
    return r.json()


def run_answer(question, top_k=TOP_K):
    r = requests.post(f"{API}/answer",json={"question":question,"top_k":top_k},timeout=60)
    r.raise_for_status()
    return r.json()


def has_kw(text, keywords):
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def hit_rank(results, keywords):
    for i,r in enumerate(results,1):
        if has_kw(r["content"], keywords):
            return i
    return 0


def calc_mrr(ranks):
    return round(sum(1.0/r for r in ranks if r>0)/len(ranks),4) if ranks else 0.0


def calc_ndcg(results, keywords, k=5):
    gains = [1.0 if has_kw(results[i]["content"],keywords) else 0.0 for i in range(min(k,len(results)))]
    dcg   = sum(g/math.log2(i+2) for i,g in enumerate(gains))
    idcg  = sum(1.0/math.log2(i+2) for i in range(int(sum(gains))))
    return round(dcg/idcg,4) if idcg>0 else 0.0


def calc_entity_coverage(answer_text, entities):
    if not entities or not answer_text:
        return 0.0
    al = answer_text.lower()
    return round(sum(1 for e in entities if e.lower() in al)/len(entities),4)


def print_section(title):
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")


#  Main 

def run_eval(retrieval_only=False, verbose=False):

    all_sections = [
        ("Section A — 222 Rajpur",  SECTION_A),
        ("Section B — Max Towers",  SECTION_B),
    ]

    retrieval_latencies, s1_lats, s2_lats = [], [], []
    generation_latencies, total_latencies = [], []
    ranks, ndcgs, entity_covs = [], [], []
    results_log = []

    for section_name, section in all_sections:
        print_section(section_name)
        for item in section:
            try:
                ret_data = run_retrieval(item["question"])
            except Exception as e:
                print(f"  [{item['id']}] RETRIEVAL ERROR: {e}")
                continue

            chunks   = ret_data.get("results",[])
            s1_ms    = ret_data.get("stage1_latency_ms",0)
            s2_ms    = ret_data.get("stage2_latency_ms",0)
            ret_ms   = s1_ms + s2_ms
            cached_r = ret_data.get("cached",False)

            if not cached_r:
                retrieval_latencies.append(ret_ms)
                s1_lats.append(s1_ms)
                s2_lats.append(s2_ms)

            rank = hit_rank(chunks, item["keywords"])
            ndcg = calc_ndcg(chunks, item["keywords"])
            ranks.append(rank)
            ndcgs.append(ndcg)

            answer_text = ""
            gen_ms = 0.0
            entity_cov = 0.0

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
                        cached_a = ans_data.get("cached",False)
                        if not cached_a:
                            generation_latencies.append(gen_ms)
                            total_latencies.append(tot_ms)
                            time.sleep(2.6)
                    entity_cov = calc_entity_coverage(answer_text, item.get("entities",[]))
                except Exception as e:
                    print(f"  [{item['id']}] LLM ERROR: {e}")

            entity_covs.append(entity_cov)

            top1 = rank==1; top3 = 0<rank<=3
            status = "✓" if top1 else ("~" if top3 else "✗")
            label  = "TOP-1" if top1 else ("TOP-3" if top3 else "MISS ")
            print(f"  [{item['id']}] {status} {label}  ret={ret_ms:5.0f}ms  gen={gen_ms:5.0f}ms  nDCG={ndcg:.2f}  ent={entity_cov:.2f}  {item['question'][:50]}")
            if verbose and answer_text:
                print(f"         → {answer_text[:120]}")

            results_log.append({
                "id": item["id"], "section": section_name,
                "question": item["question"],
                "rank": rank, "ndcg": ndcg, "entity_coverage": entity_cov,
                "retrieval_ms": round(ret_ms,2), "generation_ms": round(gen_ms,2),
                "answer": answer_text[:300],
            })

    n = len(ranks)
    hits_1 = sum(1 for r in ranks if r==1)
    hits_3 = sum(1 for r in ranks if 0<r<=3)
    hits_5 = sum(1 for r in ranks if 0<r<=5)
    mrr    = calc_mrr(ranks)
    avg_ndcg = round(sum(ndcgs)/len(ndcgs),4) if ndcgs else 0
    avg_ent  = round(sum(entity_covs)/len(entity_covs),4) if entity_covs else 0

    avg_s1  = round(sum(s1_lats)/len(s1_lats),1) if s1_lats else 0
    avg_s2  = round(sum(s2_lats)/len(s2_lats),1) if s2_lats else 0
    avg_ret = round(sum(retrieval_latencies)/len(retrieval_latencies),1) if retrieval_latencies else 0
    p95_ret = round(np.percentile(retrieval_latencies,95),1) if retrieval_latencies else 0
    avg_gen = round(sum(generation_latencies)/len(generation_latencies),1) if generation_latencies else 0
    avg_tot = round(sum(total_latencies)/len(total_latencies),1) if total_latencies else 0
    p95_tot = round(np.percentile(total_latencies,95),1) if total_latencies else 0
    p99_tot = round(np.percentile(total_latencies,99),1) if total_latencies else 0

    print(f"""
{'═'*60}
  EVAL 1/3 SUMMARY  (Sections A + B, {n} questions)
{'═'*60}

  Recall@1 : {hits_1}/{n}  ({hits_1/n*100:.1f}%)   target ≥ 75%  {'✓' if hits_1/n>=0.75 else '✗'}
  Recall@3 : {hits_3}/{n}  ({hits_3/n*100:.1f}%)   target ≥ 90%  {'✓' if hits_3/n>=0.90 else '✗'}
  Recall@5 : {hits_5}/{n}  ({hits_5/n*100:.1f}%)
  MRR      : {mrr}
  nDCG@5   : {avg_ndcg}
  Entity Coverage : {avg_ent}

  Stage 1 FAISS avg  : {avg_s1}ms
  Stage 2 Rerank avg : {avg_s2}ms
  Retrieval avg      : {avg_ret}ms  (P95: {p95_ret}ms)
  Generation avg     : {avg_gen}ms
  Total avg          : {avg_tot}ms  (P95: {p95_tot}ms  P99: {p99_tot}ms)
{'═'*60}
""")

    output = {
        "eval_file": "eval1",
        "sections": "A + B",
        "summary": {
            "recall_at_1": round(hits_1/n*100,1),
            "recall_at_3": round(hits_3/n*100,1),
            "recall_at_5": round(hits_5/n*100,1),
            "mrr": mrr, "ndcg_at_5": avg_ndcg,
            "entity_coverage_score": avg_ent,
            "latency": {
                "avg_stage1_ms": avg_s1, "avg_stage2_ms": avg_s2,
                "avg_retrieval_ms": avg_ret, "p95_retrieval_ms": p95_ret,
                "avg_generation_ms": avg_gen,
                "avg_total_ms": avg_tot, "p95_total_ms": p95_tot, "p99_total_ms": p99_tot,
            },
        },
        "questions": results_log,
    }
    Path("eval1_results.json").write_text(json.dumps(output,indent=2))
    print(f"  Saved → eval1_results.json\n")
    print(f"  Next: run eval2.py  (Sections C + D + E)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval 1/3 — Sections A + B (~43 LLM calls, ~65k tokens)")
    parser.add_argument("--upload",    action="store_true")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()
    if not check_server(): sys.exit(1)
    if args.upload: upload_pdfs()
    run_eval(retrieval_only=args.retrieval, verbose=args.verbose)