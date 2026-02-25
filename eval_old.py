"""
eval.py — Comprehensive RAG + LLM Evaluation
---------------------------------------------
Evaluates the full pipeline against the official Agmentis test question set.

Sections covered:
  A  — 222 Rajpur, Dehradun (25 questions)
  B  — Max Towers, Noida (18 questions)
  C  — Max House, Okhla (13 questions)
  D  — Cross-property comparison (15 questions)
  E  — Real-world client simulation (9 questions)
  F  — Paraphrase robustness (15 questions)
  G  — Negative / adversarial (10 questions)
  H  — Ambiguous / clarification (5 questions)

Metrics reported:
  Retrieval : Recall@1, Recall@3, Recall@5, MRR, nDCG@5
  LLM       : Entity Coverage Score, Hallucination Rate
  Robustness: Paraphrase Robustness Score
  Negative  : False Positive Rate
  Latency   : Stage-wise breakdown, Avg, P95, P99
  Caching   : Cold vs warm latency, % improvement

Usage:
    python eval.py --upload     # upload all 4 PDFs then evaluate
    python eval.py              # evaluate only
    python eval.py --retrieval  # retrieval metrics only (no LLM calls)
    python eval.py --verbose    # show answers per question
    API_BASE_URL=https://your-api.com python eval.py
"""

import os, math, json, argparse, sys, time
from pathlib import Path
import numpy as np
import requests

API   = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TOP_K = 5

PDF_FILES = [
    "222-Rajpur-Document.pdf",
    "E128-Skyvilla-Document.pdf",
    "MaxTowers-Document.pdf",
    "MaxHouse-Document.pdf",
]

# ══════════════════════════════════════════════════════════════════════════
# SECTION A — 222 Rajpur, Dehradun
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
# SECTION B — Max Towers, Noida
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

# ══════════════════════════════════════════════════════════════════════════
# SECTION C — Max House, Okhla
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
# SECTION D — Cross-property comparison
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
# SECTION E — Real-world client simulation
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

# ══════════════════════════════════════════════════════════════════════════
# SECTION F — Paraphrase robustness
# Maps to canonical questions in sections A-E
# ══════════════════════════════════════════════════════════════════════════
SECTION_F = [
    {"id":"F01","paraphrase":"Which of the three developments is a housing project rather than an office building?","original_id":"D01","keywords":["222 rajpur","residential","housing"]},
    {"id":"F02","paraphrase":"Identify the project that is not meant for commercial office use.","original_id":"D01","keywords":["222 rajpur","residential","housing"]},
    {"id":"F03","paraphrase":"Among the three properties, which one is exclusively residential in nature?","original_id":"D01","keywords":["222 rajpur","residential","housing"]},
    {"id":"F04","paraphrase":"Between Max Towers and Max House, which holds the higher level of LEED certification?","original_id":"D03","keywords":["leed","platinum","gold","max towers","max house"]},
    {"id":"F05","paraphrase":"If sustainability certification level is the deciding factor, which property ranks highest?","original_id":"D03","keywords":["leed","platinum","max towers"]},
    {"id":"F06","paraphrase":"Which development has achieved Platinum-level green certification?","original_id":"D10","keywords":["platinum","leed","max towers"]},
    {"id":"F07","paraphrase":"Which office property can employees walk to from a metro station?","original_id":"D07","keywords":["metro","walking","max house","max towers"]},
    {"id":"F08","paraphrase":"Identify the development located within walking distance of a metro stop.","original_id":"D07","keywords":["metro","walking","max house","max towers"]},
    {"id":"F09","paraphrase":"Between the Noida and Okhla projects, which one offers closer metro access?","original_id":"D07","keywords":["metro","max house","max towers","closer"]},
    {"id":"F10","paraphrase":"Which project is larger in overall constructed area: Max Towers or Max House?","original_id":"D05","keywords":["built-up","max towers","max house","larger"]},
    {"id":"F11","paraphrase":"Between the Delhi and Noida office developments, which spans more total square footage?","original_id":"D05","keywords":["built-up","max towers","max house"]},
    {"id":"F12","paraphrase":"Which property has the greater overall scale in terms of built-up space?","original_id":"D05","keywords":["built-up","max towers","max house"]},
    {"id":"F13","paraphrase":"Which property includes an indoor swimming facility?","original_id":"D13","keywords":["swimming pool","pool","max towers","max house"]},
    {"id":"F14","paraphrase":"Identify the development that provides decompression or relaxation spaces.","original_id":"D15","keywords":["decompression","relaxation","max towers","max house"]},
    {"id":"F15","paraphrase":"If employee wellness is a priority, which property explicitly supports it through facilities?","original_id":"E05","keywords":["wellness","fitness","max towers","max house"]},
]

# ══════════════════════════════════════════════════════════════════════════
# SECTION G — Negative / adversarial (answers NOT in documents)
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
# SECTION H — Ambiguous / clarification test
# System should ask for clarification or give structured multi-entity response
# ══════════════════════════════════════════════════════════════════════════
SECTION_H = [
    {"id":"H01","question":"What is the total area?","clarification_keywords":["which property","clarify","specify","222 rajpur","max towers","max house"]},
    {"id":"H02","question":"How many floors does it have?","clarification_keywords":["which property","clarify","specify","222 rajpur","max towers","max house"]},
    {"id":"H03","question":"What certification does it hold?","clarification_keywords":["which property","clarify","specify","leed","rera","ukrera"]},
    {"id":"H04","question":"How far is it from the airport?","clarification_keywords":["which property","clarify","specify","jolly grant","igi"]},
    {"id":"H05","question":"Does it offer parking?","clarification_keywords":["which property","clarify","specify","parking"]},
]

# Cross-encoder (ms-marco-MiniLM-L-6-v2) outputs raw logits, NOT [0,1] scores.
# Scores for genuinely relevant chunks typically range from -2 to +5.
# Scores for irrelevant chunks (no answer in docs) typically fall below -3.
# Threshold of -3.0 means: if the BEST chunk the system can find scores
# below -3.0, the query has no real answer in the documents.
# This is calibrated from observed score distributions — adjust if needed.
FALSE_POSITIVE_THRESHOLD = -3.0


# ── Helpers ────────────────────────────────────────────────────────────────

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
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ── Main evaluation ────────────────────────────────────────────────────────

def run_eval(retrieval_only=False, verbose=False):

    all_sections = [
        ("Section A — 222 Rajpur",        SECTION_A),
        ("Section B — Max Towers",         SECTION_B),
        ("Section C — Max House",          SECTION_C),
        ("Section D — Cross-property",     SECTION_D),
        ("Section E — Client Simulation",  SECTION_E),
    ]

    # ── Retrieval + LLM metrics for A-E ───────────────────────────────────
    retrieval_latencies, s1_lats, s2_lats = [], [], []
    generation_latencies                  = []
    total_latencies                       = []
    ranks, ndcgs, entity_covs             = [], [], []
    all_results_log                       = []

    for section_name, section in all_sections:
        print_section(section_name)

        for item in section:
            # Always run retrieval
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

            rank  = hit_rank(chunks, item["keywords"])
            ndcg  = calc_ndcg(chunks, item["keywords"])
            ranks.append(rank)
            ndcgs.append(ndcg)

            # LLM answer
            answer_text  = ""
            gen_ms       = 0.0
            entity_cov   = 0.0

            if not retrieval_only:
                try:
                    ans_data    = run_answer(item["question"])
                    answer_text = ans_data.get("answer","")
                    gen_ms      = ans_data.get("generation_ms",0)
                    tot_ms      = ans_data.get("total_ms",0)
                    cached_a    = ans_data.get("cached",False)

                    if not cached_a:
                        generation_latencies.append(gen_ms)
                        total_latencies.append(tot_ms)
                        time.sleep(2.8)  # 30 RPM limit = 1 req/2s. Extra 0.8s buffer for safety

                    entity_cov = calc_entity_coverage(answer_text, item.get("entities",[]))
                except Exception as e:
                    answer_text = f"ERROR: {e}"

            entity_covs.append(entity_cov)

            top1   = rank==1
            top3   = 0<rank<=3
            status = "✓" if top1 else ("~" if top3 else "✗")
            label  = "TOP-1" if top1 else ("TOP-3" if top3 else "MISS ")
            print(f"  [{item['id']}] {status} {label}  ret={ret_ms:5.0f}ms  gen={gen_ms:5.0f}ms  nDCG={ndcg:.2f}  ent={entity_cov:.2f}  {item['question'][:50]}")

            if verbose and answer_text:
                print(f"         → {answer_text[:120]}")
            if verbose and chunks:
                print(f"         top chunk: {chunks[0]['content'][:80].replace(chr(10),' ')}")

            all_results_log.append({
                "id": item["id"], "section": section_name,
                "question": item["question"],
                "rank": rank, "ndcg": ndcg, "entity_coverage": entity_cov,
                "retrieval_ms": round(ret_ms,2), "generation_ms": round(gen_ms,2),
                "answer": answer_text[:300] if answer_text else "",
            })

    n       = len(ranks)
    hits_1  = sum(1 for r in ranks if r==1)
    hits_3  = sum(1 for r in ranks if 0<r<=3)
    hits_5  = sum(1 for r in ranks if 0<r<=5)
    mrr     = calc_mrr(ranks)
    avg_ndcg = round(sum(ndcgs)/len(ndcgs),4) if ndcgs else 0
    avg_ent  = round(sum(entity_covs)/len(entity_covs),4) if entity_covs else 0

    # ── Section F: Paraphrase Robustness ──────────────────────────────────
    print_section("Section F — Paraphrase Robustness")

    para_results = []
    for item in SECTION_F:
        orig = next((r for r in all_results_log if r["id"]==item["original_id"]),None)
        orig_hit = orig and orig.get("rank",0)>0

        try:
            data     = run_retrieval(item["paraphrase"])
            para_hit = any(has_kw(r["content"],item["keywords"]) for r in data.get("results",[])[:3])
        except Exception:
            para_hit = False

        consistent = orig_hit==para_hit
        status = "✓" if consistent else "✗"
        print(f"  {status} [{item['id']}] orig={'HIT' if orig_hit else 'MISS'}  para={'HIT' if para_hit else 'MISS'}  {item['paraphrase'][:55]}")
        para_results.append(consistent)

    para_score = round(sum(para_results)/len(para_results)*100,1) if para_results else 0

    # ── Section G: Negative / Adversarial ─────────────────────────────────
    print_section("Section G — Negative / Adversarial")

    fp_count = 0
    hallucination_count = 0

    for item in SECTION_G:
        try:
            ret_data = run_retrieval(item["question"])
            chunks   = ret_data.get("results",[])
            is_fp    = bool(chunks and chunks[0].get("rerank_score",-999) > FALSE_POSITIVE_THRESHOLD)

            hal_text = ""
            if not retrieval_only and chunks:
                try:
                    ans_data = run_answer(item["question"])
                    hal_text = ans_data.get("answer","").lower()
                except Exception:
                    pass

            # Hallucination: LLM answers confidently without any abstention phrase.
            # A well-behaved system should say one of these when the answer
            # doesn't exist in the documents. If none appear, it hallucinated.
            ABSTENTION_PHRASES = [
                "not available", "not found", "not mentioned", "not in the",
                "no information", "cannot find", "do not", "does not mention",
                "not provided", "not specified", "no reference", "not stated",
                "unable to find", "not present", "not included", "not described",
            ]
            is_hallucination = bool(
                hal_text and
                len(hal_text) > 50 and
                not any(p in hal_text for p in ABSTENTION_PHRASES)
            )

            if is_fp:     fp_count += 1
            if is_hallucination: hallucination_count += 1

            fp_tag  = "FP" if is_fp else "OK"
            hal_tag = "HAL" if is_hallucination else " OK"
            score   = chunks[0].get("rerank_score","n/a") if chunks else "no results"
            print(f"  [{item['id']}] retrieval={fp_tag}  llm={hal_tag}  rerank={score}  {item['question'][:50]}")
            if verbose and hal_text:
                print(f"         → {hal_text[:100]}")

        except Exception as e:
            print(f"  [{item['id']}] ERROR: {e}")

    fpr           = round(fp_count/len(SECTION_G)*100,1)
    hal_rate      = round(hallucination_count/len(SECTION_G)*100,1)

    # ── Section H: Ambiguous / Clarification ──────────────────────────────
    print_section("Section H — Ambiguous / Clarification")

    clarification_hits = 0
    for item in SECTION_H:
        try:
            if retrieval_only:
                data = run_retrieval(item["question"])
                answer_text = " ".join(r["content"] for r in data.get("results",[])[:2])
            else:
                data = run_answer(item["question"])
                answer_text = data.get("answer","").lower()

            triggered = has_kw(answer_text, item["clarification_keywords"])
            if triggered: clarification_hits += 1
            status = "✓" if triggered else "✗"
            print(f"  {status} [{item['id']}] clarification={'triggered' if triggered else 'NOT triggered'}  {item['question']}")
            if verbose:
                print(f"         → {answer_text[:120]}")
        except Exception as e:
            print(f"  [{item['id']}] ERROR: {e}")

    clarification_score = round(clarification_hits/len(SECTION_H)*100,1)

    # ── Caching ────────────────────────────────────────────────────────────
    print_section("Caching Strategy")
    print("  Re-running first 5 questions to measure cache speedup...\n")

    cold_times, warm_times = [], []
    sample = (SECTION_A + SECTION_B)[:5]

    for item in sample:
        match = next((r for r in all_results_log if r["id"]==item["id"]),None)
        if match: cold_times.append(match["retrieval_ms"])
        try:
            t0   = time.perf_counter()
            data = run_retrieval(item["question"])
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

    # ── Latency ────────────────────────────────────────────────────────────
    avg_ret  = round(sum(retrieval_latencies)/len(retrieval_latencies),1) if retrieval_latencies else 0
    p95_ret  = round(np.percentile(retrieval_latencies,95),1) if retrieval_latencies else 0
    avg_s1   = round(sum(s1_lats)/len(s1_lats),1) if s1_lats else 0
    avg_s2   = round(sum(s2_lats)/len(s2_lats),1) if s2_lats else 0
    avg_gen  = round(sum(generation_latencies)/len(generation_latencies),1) if generation_latencies else 0
    avg_tot  = round(sum(total_latencies)/len(total_latencies),1) if total_latencies else 0
    p95_tot  = round(np.percentile(total_latencies,95),1) if total_latencies else 0
    p99_tot  = round(np.percentile(total_latencies,99),1) if total_latencies else 0

    # ── Final Summary ──────────────────────────────────────────────────────
    print(f"""
{'═'*60}
  FINAL EVALUATION SUMMARY
{'═'*60}

  RETRIEVAL QUALITY  ({n} questions, Sections A–E)
  ──────────────────────────────────────────────
  Recall@1  (Top-1 Accuracy) : {hits_1}/{n}  ({hits_1/n*100:.1f}%)   target ≥ 75%  {'✓' if hits_1/n>=0.75 else '✗'}
  Recall@3  (Top-3 Accuracy) : {hits_3}/{n}  ({hits_3/n*100:.1f}%)   target ≥ 90%  {'✓' if hits_3/n>=0.90 else '✗'}
  Recall@5                   : {hits_5}/{n}  ({hits_5/n*100:.1f}%)
  MRR                        : {mrr}
  nDCG@5                     : {avg_ndcg}
  Entity Coverage Score      : {avg_ent}   (measured on LLM answer)

  ROBUSTNESS  (Section F — 15 paraphrase questions)
  ──────────────────────────────────────────────
  Paraphrase Robustness Score: {para_score}%

  NEGATIVE QUERIES  (Section G — 10 adversarial questions)
  ──────────────────────────────────────────────
  False Positive Rate        : {fpr}%   ({fp_count}/{len(SECTION_G)} retrieval FPs)
  Hallucination Rate         : {hal_rate}%   ({hallucination_count}/{len(SECTION_G)} LLM hallucinations)

  AMBIGUOUS QUERIES  (Section H — 5 clarification questions)
  ──────────────────────────────────────────────
  Clarification Score        : {clarification_score}%   ({clarification_hits}/{len(SECTION_H)} correctly clarified)

  CACHING
  ──────────────────────────────────────────────
  Cold avg   : {avg_cold}ms
  Warm avg   : {avg_warm}ms
  Reduction  : {cache_pct}%

  LATENCY  (cold queries only)
  ──────────────────────────────────────────────
  Stage 1  FAISS avg         : {avg_s1}ms
  Stage 2  Rerank avg        : {avg_s2}ms
  Retrieval total avg        : {avg_ret}ms   (P95: {p95_ret}ms)
  Generation avg (Groq)      : {avg_gen}ms
  End-to-end avg             : {avg_tot}ms   (P95: {p95_tot}ms  P99: {p99_tot}ms)

  Reranking worth it?  {'YES — Top-3 ≥ 90%' if hits_3/n>=0.90 else 'MARGINAL — consider smaller model'}

{'═'*60}
""")

    output = {
        "summary": {
            "recall_at_1": round(hits_1/n*100,1),
            "recall_at_3": round(hits_3/n*100,1),
            "recall_at_5": round(hits_5/n*100,1),
            "mrr": mrr,
            "ndcg_at_5": avg_ndcg,
            "entity_coverage_score": avg_ent,
            "paraphrase_robustness_pct": para_score,
            "false_positive_rate_pct": fpr,
            "hallucination_rate_pct": hal_rate,
            "clarification_score_pct": clarification_score,
            "caching": {"cold_avg_ms":avg_cold,"warm_avg_ms":avg_warm,"reduction_pct":cache_pct},
            "latency": {
                "avg_stage1_ms": avg_s1,
                "avg_stage2_ms": avg_s2,
                "avg_retrieval_ms": avg_ret,
                "p95_retrieval_ms": p95_ret,
                "avg_generation_ms": avg_gen,
                "avg_total_ms": avg_tot,
                "p95_total_ms": p95_tot,
                "p99_total_ms": p99_tot,
            },
        },
        "questions": all_results_log,
    }

    Path("eval_results.json").write_text(json.dumps(output,indent=2))
    print(f"  Results saved → eval_results.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload",       action="store_true", help="Upload all PDFs first")
    parser.add_argument("--retrieval",    action="store_true", help="Retrieval metrics only — skip LLM calls")
    parser.add_argument("--verbose",      action="store_true", help="Show answers + top chunks")
    args = parser.parse_args()

    if not check_server():
        sys.exit(1)

    if args.upload:
        upload_pdfs()

    run_eval(retrieval_only=args.retrieval, verbose=args.verbose)