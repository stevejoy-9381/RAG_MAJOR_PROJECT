"""
test_reranking.py — Verify the re-ranking pipeline works correctly
──────────────────────────────────────────────────────────────────
Run from the backend directory:
  python test_reranking.py

Prerequisites:
  - At least one PDF uploaded for a user (the script uses the first user found)
  - The CrossEncoder model will auto-download on first run (~80MB)

What it does:
  1. Finds a user with an existing FAISS index
  2. Runs retrieve_and_rerank() with RERANKING_ENABLED=true
  3. Runs the same query with direct retriever (simulating RERANKING_ENABLED=false)
  4. Prints a side-by-side comparison of document ordering
  5. Reports total latency for each approach
"""

import os
import sys
import time

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.retriever import (
    get_hybrid_retriever, rerank, retrieve_and_rerank,
    format_sources, RETRIEVAL_CANDIDATES, FINAL_CONTEXT_K, RERANKING_ENABLED,
)

TEST_QUESTION = "What are the main findings or conclusions?"


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
    print(f"  RE-RANKING COMPARISON TEST")
    print(f"  Question: \"{question}\"")
    print(f"  Config: RETRIEVAL_CANDIDATES={RETRIEVAL_CANDIDATES}, FINAL_CONTEXT_K={FINAL_CONTEXT_K}")
    print(f"  RERANKING_ENABLED={RERANKING_ENABLED}")
    print(f"{'='*70}\n")

    # ── Phase 1: Retrieve candidates (without reranking) ──────────────────
    print("─── STEP 1: Raw retrieval (no reranking) ───")
    retriever = get_hybrid_retriever(user_id)

    t0 = time.time()
    raw_docs = retriever.invoke(question)
    retrieval_ms = (time.time() - t0) * 1000

    print(f"\nRetrieved {len(raw_docs)} candidates in {retrieval_ms:.0f}ms")
    print(f"{'Rank':<5} {'Source':<35} {'Page':>5}")
    print(f"{'─'*5} {'─'*35} {'─'*5}")
    for i, doc in enumerate(raw_docs, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0) + 1
        print(f"{i:<5} {src:<35} {page:>5}")

    # ── Phase 2: Rerank ──────────────────────────────────────────────────
    print(f"\n─── STEP 2: CrossEncoder re-ranking ───")

    t1 = time.time()
    ranked = rerank(question, raw_docs)
    rerank_ms = (time.time() - t1) * 1000

    print(f"\n─── STEP 3: Side-by-side comparison ───\n")
    print(f"{'Pre-rerank order':<40} {'Post-rerank order':<40}")
    print(f"{'─'*40} {'─'*40}")

    max_rows = max(len(raw_docs), len(ranked))
    for i in range(max_rows):
        # Pre-rerank column
        if i < len(raw_docs):
            src = raw_docs[i].metadata.get("source", "?")
            pg  = raw_docs[i].metadata.get("page", 0) + 1
            left = f"{i+1}. {src} (p{pg})"
        else:
            left = ""

        # Post-rerank column
        if i < len(ranked):
            doc, score = ranked[i]
            src = doc.metadata.get("source", "?")
            pg  = doc.metadata.get("page", 0) + 1
            right = f"{i+1}. {src} (p{pg}) [{float(score):.4f}]"
        else:
            right = ""

        print(f"{left:<40} {right:<40}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  LATENCY SUMMARY")
    print(f"  Retrieval:  {retrieval_ms:>7.0f}ms  ({len(raw_docs)} docs)")
    print(f"  Reranking:  {rerank_ms:>7.0f}ms  ({len(raw_docs)} → {len(ranked)} docs)")
    print(f"  Total:      {retrieval_ms + rerank_ms:>7.0f}ms")
    print(f"{'='*70}")

    # Check if order changed
    if len(ranked) > 0 and len(raw_docs) > 0:
        pre_order = [d.page_content[:50] for d in raw_docs[:FINAL_CONTEXT_K]]
        post_order = [d.page_content[:50] for d, _ in ranked]
        if pre_order == post_order:
            print("\n📌 Reranking did NOT change the order (top-K already optimal)")
        else:
            print("\n✅ Reranking CHANGED the document order (improved context selection)")

    # Formatted sources output
    print(f"\n─── format_sources() output ───")
    reranked_docs = [doc for doc, _ in ranked]
    sources = format_sources(reranked_docs)
    for s in sources:
        score_str = f" [score: {s['relevance_score']:.4f}]" if 'relevance_score' in s else ""
        print(f"  • {s['file']} p{s['page']}{score_str}")


if __name__ == "__main__":
    run_test()
