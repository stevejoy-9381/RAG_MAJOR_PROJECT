"""
test_query_rewriting.py — Compare retrieval with vs without query rewriting
──────────────────────────────────────────────────────────────────────────────
Run from the backend directory:
  python test_query_rewriting.py

Prerequisites:
  - At least one PDF uploaded for a user
  - An LLM provider available (Ollama running or GROQ_API_KEY set)
  - CrossEncoder model will auto-download on first run (~80MB)

What it does:
  1. Finds a user with an existing FAISS index
  2. Gets an LLM provider (auto mode)
  3. Runs retrieval WITHOUT query rewriting (Phase 4 behavior)
  4. Runs retrieval WITH query rewriting (Phase 5 behavior)
  5. Prints the rewritten queries and a side-by-side source comparison
  6. Reports latency breakdown for each step
"""

import os
import sys
import time

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.retriever import (
    retrieve_and_rerank, format_sources,
    RETRIEVAL_CANDIDATES, FINAL_CONTEXT_K, RERANKING_ENABLED,
)
from src.query_rewriter import rewrite_query
from src.llm_provider import get_provider

# Deliberately vague question to show value of query rewriting
TEST_QUESTION = "what about the important stuff?"


def find_user_with_index() -> str:
    """Find a user_id that has a FAISS index on disk."""
    vs_dir = "vectorstore"
    if not os.path.exists(vs_dir):
        print("❌ No vectorstore directory found. Upload a document first.")
        sys.exit(1)

    for entry in os.listdir(vs_dir):
        index_path = os.path.join(vs_dir, entry, "faiss_index", "index.faiss")
        if os.path.exists(index_path):
            print(f"✅ Found user index: {entry[:8]}...")
            return entry

    print("❌ No user has a FAISS index. Upload a document first.")
    sys.exit(1)


def run_test():
    user_id = find_user_with_index()
    question = TEST_QUESTION

    print(f"\n{'='*70}")
    print(f"  QUERY REWRITING COMPARISON TEST")
    print(f"  Question: \"{question}\"")
    print(f"  Config: RETRIEVAL_CANDIDATES={RETRIEVAL_CANDIDATES}, FINAL_CONTEXT_K={FINAL_CONTEXT_K}")
    print(f"  RERANKING_ENABLED={RERANKING_ENABLED}")
    print(f"{'='*70}\n")

    # ── Get provider ──────────────────────────────────────────────────────
    try:
        provider = get_provider("auto")
        print(f"✅ LLM Provider: {provider.name}\n")
    except Exception as e:
        print(f"❌ No LLM provider available: {e}")
        print("   Start Ollama or set GROQ_API_KEY to run this test.")
        sys.exit(1)

    # ── Phase 1: Retrieve WITHOUT query rewriting ─────────────────────────
    print(f"─── STEP 1: Retrieval WITHOUT query rewriting (Phase 4 only) ───\n")

    t0 = time.time()
    docs_without = retrieve_and_rerank(question, user_id, queries=None)
    time_without = (time.time() - t0) * 1000

    print(f"\nResults ({time_without:.0f}ms):")
    sources_without = format_sources(docs_without)
    for i, s in enumerate(sources_without, 1):
        score_str = f" [{s['relevance_score']:.4f}]" if 'relevance_score' in s else ""
        print(f"  {i}. {s['file']} (p{s['page']}){score_str}")

    # ── Phase 2: Rewrite the query ────────────────────────────────────────
    print(f"\n─── STEP 2: Query rewriting ───\n")

    t1 = time.time()
    rewritten = rewrite_query(question, provider)
    rewrite_ms = (time.time() - t1) * 1000

    # ── Phase 3: Retrieve WITH rewritten queries ──────────────────────────
    print(f"─── STEP 3: Retrieval WITH query rewriting ───\n")

    t2 = time.time()
    docs_with = retrieve_and_rerank(question, user_id, queries=rewritten)
    time_with = (time.time() - t2) * 1000

    print(f"\nResults ({time_with:.0f}ms retrieval + {rewrite_ms:.0f}ms rewriting):")
    sources_with = format_sources(docs_with)
    for i, s in enumerate(sources_with, 1):
        score_str = f" [{s['relevance_score']:.4f}]" if 'relevance_score' in s else ""
        print(f"  {i}. {s['file']} (p{s['page']}){score_str}")

    # ── Side-by-side comparison ───────────────────────────────────────────
    print(f"\n─── STEP 4: Side-by-side comparison ───\n")
    print(f"{'WITHOUT rewriting':<40} {'WITH rewriting':<40}")
    print(f"{'─'*40} {'─'*40}")

    max_rows = max(len(sources_without), len(sources_with))
    for i in range(max_rows):
        # Left column: without rewriting
        if i < len(sources_without):
            s = sources_without[i]
            score_str = f" [{s['relevance_score']:.4f}]" if 'relevance_score' in s else ""
            left = f"{i+1}. {s['file']} (p{s['page']}){score_str}"
        else:
            left = ""

        # Right column: with rewriting
        if i < len(sources_with):
            s = sources_with[i]
            score_str = f" [{s['relevance_score']:.4f}]" if 'relevance_score' in s else ""
            right = f"{i+1}. {s['file']} (p{s['page']}){score_str}"
        else:
            right = ""

        print(f"{left:<40} {right:<40}")

    # ── Check if results changed ──────────────────────────────────────────
    without_keys = [(s['file'], s['page']) for s in sources_without]
    with_keys = [(s['file'], s['page']) for s in sources_with]

    if without_keys == with_keys:
        print("\n📌 Same sources returned (rewriting didn't change results)")
    else:
        new_sources = set(with_keys) - set(without_keys)
        if new_sources:
            print(f"\n✅ Query rewriting found {len(new_sources)} NEW source(s):")
            for f, p in new_sources:
                print(f"   • {f} (p{p})")
        else:
            print("\n✅ Query rewriting CHANGED the order of sources")

    # ── Latency summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  LATENCY SUMMARY")
    print(f"  Without rewriting:  {time_without:>7.0f}ms")
    print(f"  Query rewriting:    {rewrite_ms:>7.0f}ms  ({len(rewritten)} queries generated)")
    print(f"  With rewriting:     {time_with:>7.0f}ms  (retrieval only)")
    print(f"  Total with:         {rewrite_ms + time_with:>7.0f}ms  (rewrite + retrieval)")
    print(f"  Overhead:           {(rewrite_ms + time_with) - time_without:>+7.0f}ms")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_test()
