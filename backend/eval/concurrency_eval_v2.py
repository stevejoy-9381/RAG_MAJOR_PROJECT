"""
eval/concurrency_eval_v2.py — Phase 8: Concurrency Re-Test & Comparison
───────────────────────────────────────────────────────────────────────
Benchmarks query latency and throughput under concurrent user loads:
  - 1 Concurrent User
  - 5 Concurrent Users
  - 10 Concurrent Users

Compares:
  - OLD Configuration (v1): Full Hybrid + PyTorch CrossEncoder Re-rank Top 10 (RETRIEVAL_CANDIDATES=10)
  - NEW Configuration (v2): Optimized Hybrid + CrossEncoder Re-rank Top 6 with Warm-up

Features:
  - Warm-up request prior to measurement to eliminate cold start model load distortion.
  - Side-by-side reporting of p50, p95, avg latency, and QPS throughput.

Saves results to eval/results_v2/concurrency.json.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

from src.ingest import get_embedding_model
from src.retriever import _get_cross_encoder
from src.config import FINAL_CONTEXT_K

EVAL_USER_ID = "eval_v2_runner"
RESULTS_DIR = BACKEND_DIR / "eval" / "results_v2"
RESULTS_FILE = RESULTS_DIR / "concurrency.json"


def run_phase8():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*85)
    print("PHASE 8: CONCURRENCY BENCHMARK (1, 5, 10 USERS) — OLD VS NEW")
    print("="*85)

    embedding_model = get_embedding_model()
    index_path = BACKEND_DIR / "vectorstore" / EVAL_USER_ID / "faiss_index"
    vectorstore = FAISS.load_local(str(index_path), embedding_model, allow_dangerous_deserialization=True)
    indexed_docs = list(vectorstore.docstore._dict.values())

    bm25 = BM25Retriever.from_documents(indexed_docs)
    bm25.k = 10

    cross_encoder = _get_cross_encoder()

    test_queries = [
        "What financial assistance is allocated to newly hired telecommuting staff?",
        "What is the maximum file payload volume permitted when transmitting a single record?",
        "What is the guaranteed service availability commitment offered to top-tier accounts?",
        "How quickly must staff escalate an observed credentials disclosure to administrators?",
        "What exact value is assigned to the configuration parameter MAX_UPLOAD_SIZE_BYTES?",
        "What numerical values are assigned to BM25_WEIGHT and FAISS_WEIGHT?",
        "Explain the end-to-end data lifecycle for a terminating tenant.",
        "Compare the response time SLAs for Severity 1 versus Severity 2 incidents.",
        "What is the annual deductible for Health Plan Tier B compared to Tier A?",
        "What rate limit in requests per minute is applied to Professional Tier users?",
    ]

    # ─── WARM-UP REQUEST ───────────────────────────────────────────────────────
    print("\n[WARM-UP] Executing warm-up query to prime PyTorch weights & caches...")
    t_warmup = time.perf_counter()
    warm_candidates = vectorstore.similarity_search(test_queries[0], k=10)
    warm_pairs = [(test_queries[0], d.page_content) for d in warm_candidates]
    _ = cross_encoder.predict(warm_pairs)
    warmup_ms = (time.perf_counter() - t_warmup) * 1000
    print(f"[WARM-UP] Completed in {warmup_ms:.2f} ms. System primed for measurement.\n")

    concurrency_levels = [1, 5, 10]

    # Pipeline Runners
    def execute_old_pipeline(q_text: str) -> float:
        """Old v1: Re-rank top 10 candidates."""
        t0 = time.perf_counter()
        f_docs = vectorstore.similarity_search(q_text, k=10)
        b_docs = bm25.invoke(q_text)[:10]
        # deduplicate
        seen = set()
        candidates = []
        for d in f_docs + b_docs:
            h = d.page_content.strip()
            if h not in seen:
                seen.add(h)
                candidates.append(d)
                if len(candidates) == 10:
                    break
        pairs = [(q_text, d.page_content) for d in candidates]
        scores = cross_encoder.predict(pairs)
        scored = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
        _ = scored[:FINAL_CONTEXT_K]
        return (time.perf_counter() - t0) * 1000

    def execute_new_pipeline(q_text: str) -> float:
        """New v2: Re-rank top 6 candidates (Variant B verified with identical 100% Hit@4)."""
        t0 = time.perf_counter()
        f_docs = vectorstore.similarity_search(q_text, k=6)
        b_docs = bm25.invoke(q_text)[:6]
        seen = set()
        candidates = []
        for d in f_docs + b_docs:
            h = d.page_content.strip()
            if h not in seen:
                seen.add(h)
                candidates.append(d)
                if len(candidates) == 6:
                    break
        pairs = [(q_text, d.page_content) for d in candidates]
        scores = cross_encoder.predict(pairs)
        scored = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
        _ = scored[:FINAL_CONTEXT_K]
        return (time.perf_counter() - t0) * 1000

    old_results = {}
    new_results = {}

    for c in concurrency_levels:
        total_requests = 20 if c > 1 else 10
        queries_to_run = [test_queries[i % len(test_queries)] for i in range(total_requests)]

        # 1. Benchmark Old v1 Config
        print(f"[BENCHMARK] Running OLD (v1: Top 10) with {c} concurrent user(s) ({total_requests} queries)...")
        latencies_old = []
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as executor:
            futures = [executor.submit(execute_old_pipeline, q) for q in queries_to_run]
            for fut in as_completed(futures):
                latencies_old.append(fut.result())
        t_old_batch = time.perf_counter() - t_start

        old_p50 = float(np.percentile(latencies_old, 50))
        old_p95 = float(np.percentile(latencies_old, 95))
        old_avg = float(np.mean(latencies_old))
        old_qps = total_requests / t_old_batch

        old_results[f"{c}_users"] = {
            "concurrent_workers": c,
            "total_requests": total_requests,
            "p50_ms": round(old_p50, 2),
            "p95_ms": round(old_p95, 2),
            "avg_ms": round(old_avg, 2),
            "qps": round(old_qps, 2)
        }

        # 2. Benchmark New v2 Config
        print(f"[BENCHMARK] Running NEW (v2: Top 6) with {c} concurrent user(s) ({total_requests} queries)...")
        latencies_new = []
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as executor:
            futures = [executor.submit(execute_new_pipeline, q) for q in queries_to_run]
            for fut in as_completed(futures):
                latencies_new.append(fut.result())
        t_new_batch = time.perf_counter() - t_start

        new_p50 = float(np.percentile(latencies_new, 50))
        new_p95 = float(np.percentile(latencies_new, 95))
        new_avg = float(np.mean(latencies_new))
        new_qps = total_requests / t_new_batch

        new_results[f"{c}_users"] = {
            "concurrent_workers": c,
            "total_requests": total_requests,
            "p50_ms": round(new_p50, 2),
            "p95_ms": round(new_p95, 2),
            "avg_ms": round(new_avg, 2),
            "qps": round(new_qps, 2)
        }

    output_data = {
        "warmup_latency_ms": round(warmup_ms, 2),
        "old_configuration": old_results,
        "new_configuration": new_results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[PHASE 8] Saved full concurrency results to {RESULTS_FILE.resolve()}")

    print("\n" + "="*95)
    print(f"{'CONCURRENCY':<12} | {'OLD p50':<12} | {'NEW p50':<12} | {'OLD p95':<12} | {'NEW p95':<12} | {'OLD QPS':<9} | {'NEW QPS'}")
    print("="*95)
    for c in concurrency_levels:
        key = f"{c}_users"
        o = old_results[key]
        n = new_results[key]
        print(f"{c:<2} Users     | {o['p50_ms']:<9.2f} ms | {n['p50_ms']:<9.2f} ms | {o['p95_ms']:<9.2f} ms | {n['p95_ms']:<9.2f} ms | {o['qps']:<6.2f}   | {n['qps']:<6.2f}")
    print("="*95)


if __name__ == "__main__":
    run_phase8()
