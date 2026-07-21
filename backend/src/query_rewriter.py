"""
src/query_rewriter.py — LLM-Powered Query Rewriting (Phase 5)
──────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Uses the active LLM provider (Ollama or Groq) to rewrite a user's
  question into 2-3 alternative phrasings optimized for document search.

WHY REWRITE QUERIES?
  Users ask questions conversationally: "what about the pump?"
  But document retrieval works best with specific, descriptive queries:
    - "pump maintenance procedures"
    - "pump specifications and models"
    - "pump operating instructions"

  By expanding one vague question into multiple focused queries, we cast
  a wider net during retrieval. The Phase 4 CrossEncoder reranker then
  picks the best results from the combined pool — best of both worlds.

PIPELINE POSITION:
  User question
    → [Phase 5] rewrite_query() generates 2-3 alternatives
    → [Phase 3] retrieve via BM25+FAISS for EACH query
    → merge + deduplicate candidates
    → [Phase 4] CrossEncoder rerank → top FINAL_CONTEXT_K docs
    → build context → LLM answer

COST/LATENCY:
  The rewrite prompt is tiny (~100 tokens in, ~50 tokens out).
  - Groq: ~100-300ms
  - Ollama: ~1-3s (depends on model/hardware)
  Disabled by default (QUERY_REWRITING_ENABLED=false) since it adds
  an extra LLM call per user request. Opt-in via env var or per-request.

CONNECTIONS:
  → Called from api.py's /stream and /chat endpoints (when enabled)
  → Uses src/llm_provider.LLMProvider.chat(stream=False)
  → Results fed into src/retriever.retrieve_and_rerank()
"""

import time
from src.llm_provider import LLMProvider


# ─── Prompt ───────────────────────────────────────────────────────────────────
# Kept intentionally short to minimize latency and token cost.
# The model returns one query per line — easy to parse, no JSON needed.

_REWRITE_SYSTEM_PROMPT = """\
You are a search query optimizer. Given a user's question about their documents, \
generate 2-3 alternative phrasings that would help find relevant information \
in a document search engine.

Rules:
- Each query should be a different angle or decomposition of the question
- Optimize for keyword/semantic search, not conversation
- One query per line, no numbering, no explanation
- Keep queries concise (under 15 words each)
- Do NOT repeat the original question"""


def rewrite_query(question: str, provider: LLMProvider) -> list[str]:
    """
    Rewrite a user's question into multiple search-optimized queries.

    Uses the active LLM provider to generate 2-3 alternative phrasings
    of the original question, optimized for document retrieval rather
    than conversation.

    The original question is ALWAYS included as the first query in the
    returned list — rewriting adds alternatives, never replaces.

    Args:
        question: The user's original question
        provider: The active LLM provider (Ollama or Groq)

    Returns:
        List of query strings, starting with the original question,
        followed by 2-3 rewritten alternatives.
        On any error, returns [question] (graceful fallback).
    """
    t0 = time.time()

    try:
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Non-streaming call — we need the full response to parse
        response = provider.chat(messages, stream=False)
        rewrite_ms = (time.time() - t0) * 1000

        # Parse: split by newlines, strip whitespace, filter empties
        raw_lines = response.strip().split("\n")
        alternatives = []
        seen_lower = {question.lower()}  # deduplicate against original

        for line in raw_lines:
            cleaned = line.strip()
            # Strip common formatting the LLM might add despite instructions
            # e.g., "1. query", "- query", "• query"
            for prefix in ("1.", "2.", "3.", "4.", "5.", "-", "•", "*"):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break

            if not cleaned:
                continue
            if cleaned.lower() in seen_lower:
                continue  # skip duplicates

            seen_lower.add(cleaned.lower())
            alternatives.append(cleaned)

        # Always start with the original, then append alternatives
        queries = [question] + alternatives

        # Log the rewriting results
        print(f"\n[QUERY_REWRITER] ── Query rewriting ({rewrite_ms:.0f}ms, {provider.name}) ──")
        print(f"[QUERY_REWRITER] Original: \"{question}\"")
        print(f"[QUERY_REWRITER] Rewritten queries:")
        for i, q in enumerate(queries):
            marker = "  (original)" if i == 0 else ""
            print(f"[QUERY_REWRITER]   {i+1}. \"{q}\"{marker}")
        print()

        return queries

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        print(f"\n[QUERY_REWRITER] ⚠ Rewriting failed ({elapsed_ms:.0f}ms): {e}")
        print(f"[QUERY_REWRITER] Falling back to original question only")
        return [question]
