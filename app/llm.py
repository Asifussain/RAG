"""
llm.py — Groq LLM generation layer.

Uses Groq's inference API (llama-3.3-70b-versatile) for fast, grounded
natural language answer synthesis over retrieved chunks.

Design principles:
  - Grounded only in retrieved chunks never hallucinate beyond context
  - If no relevant chunk exists, explicitly say so
  - Cite source (filename + page) for every factual claim
  - Handle ambiguous queries by asking for clarification
  - Multi-property comparison questions synthesised across all sources
"""

import time
from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL, LLM_ENABLED

# Client
_client = None

if LLM_ENABLED:
    _client = Groq(api_key=GROQ_API_KEY)
    print(f"✓ Groq ({GROQ_MODEL}) ready — LLM generation enabled")
else:
    print(" GROQ_API_KEY not set — /answer endpoint disabled")


# System prompt
SYSTEM_PROMPT = """You are a precise real estate assistant helping users query property documents. The available properties are: 222 Rajpur (Dehradun), Max Towers (Noida), and Max House (Okhla).

You will be given:
1. A user question
2. A set of retrieved document chunks with source information

Your job:
- Answer the question using ONLY the information in the provided chunks
- Be specific and factual — include exact numbers, measurements, and figures when available
- Cite your sources naturally by mentioning the property name
- If the question asks about multiple properties, compare them clearly and concisely
- If the answer is not in the chunks, respond exactly with: "This information is not available in the uploaded documents."
- If the question is ambiguous and does not specify which property (e.g. "What is the total area?", "How many floors does it have?", "How far is it from the airport?"), you MUST ask for clarification. Say: "Could you please clarify which property you are referring to — 222 Rajpur, Max Towers, or Max House?"
- IMPORTANT: Short vague questions with no property name like "What is the total area?", "Does it offer parking?", "What certification does it hold?" are always ambiguous. Always ask for clarification for these.
- Never make up or infer information not explicitly present in the chunks
- Keep answers concise but complete — typically 2-5 sentences unless a detailed comparison is needed
- Write in professional natural language"""


# Context builder

def build_context(results: list) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    if not results:
        return "No relevant chunks retrieved."
    lines = []
    for i, r in enumerate(results, 1):
        filename = r.get("filename", "unknown")
        page     = r.get("page_number", "?")
        content  = r.get("content", "").strip()
        lines.append(f"[Source {i}: {filename}, page {page}]\n{content}")
    return "\n\n".join(lines)


# Main generate function

def generate_answer(question: str, results: list) -> dict:
    """
    Generate a grounded answer from retrieved chunks using Groq.

    Returns:
        answer        — natural language answer string
        generation_ms — time taken for LLM call
        sources_used  — list of sources (filename + page)
    """
    if not LLM_ENABLED or _client is None:
        return {
            "answer":        "LLM generation is not enabled. Set GROQ_API_KEY in .env",
            "generation_ms": 0.0,
            "sources_used":  [],
        }

    use_model = GROQ_MODEL
    context   = build_context(results)

    t0 = time.perf_counter()
    try:
        response = _client.chat.completions.create(
            model    = use_model,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"RETRIEVED CONTEXT:\n{context}\n\nQUESTION: {question}"},
            ],
            temperature = 0.1,
            max_tokens  = 512,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Generation error: {str(e)}"

    generation_ms = (time.perf_counter() - t0) * 1000

    # Unique sources from retrieved results
    sources_used, seen = [], set()
    for r in results:
        key = (r.get("filename", ""), r.get("page_number", 0))
        if key not in seen:
            seen.add(key)
            sources_used.append({
                "filename":     r.get("filename", "unknown"),
                "page_number":  r.get("page_number", -1),
                "rerank_score": r.get("rerank_score", 0),
            })

    return {
        "answer":        answer,
        "generation_ms": round(generation_ms, 2),
        "sources_used":  sources_used,
    }