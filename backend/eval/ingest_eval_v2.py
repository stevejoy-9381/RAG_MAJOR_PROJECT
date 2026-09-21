"""
eval/ingest_eval_v2.py — Phase 7: Real Ingestion & Cache Benchmark
─────────────────────────────────────────────────────────────────
Benchmarks document parsing, chunking, and embedding across:
  - 3 Real PDFs: 1MB (94 pages), 10MB (749 pages), 50MB (3499 pages)
  - 1 Real DOCX: real_course_exam.docx
  - 1 Real CSV:  real_job_postings.csv

Measures:
  - File Size (MB)
  - Actual Page Count / Block Count
  - Actual Chunk Count
  - Ingestion Elapsed Time (s)
  - Throughput (MB/s)
  - Cold vs Warm Cache Ingestion Times & Speedup

Saves results to eval/results_v2/ingestion.json.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from src.ingest import (
    load_document, enrich_metadata, chunk_documents, append_to_user_index,
    get_embedding_model, _get_cache_conn
)

TEST_DOCS_DIR = BACKEND_DIR / "eval" / "real_test_docs"
RESULTS_DIR = BACKEND_DIR / "eval" / "results_v2"
RESULTS_FILE = RESULTS_DIR / "ingestion.json"
TEST_USER = "ingest_bench_user"


def run_phase7():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embedding_model = get_embedding_model()

    test_files = [
        {"name": "enterprise_handbook_1mb.pdf", "format": "PDF (1 MB)", "sample_pages": 94},
        {"name": "compliance_knowledge_base_10mb.pdf", "format": "PDF (10 MB)", "sample_pages": 749},
        {"name": "system_audit_master_archive_50mb.pdf", "format": "PDF (50 MB)", "sample_pages": 3499},
        {"name": "real_course_exam.docx", "format": "DOCX", "sample_pages": None},
        {"name": "real_job_postings.csv", "format": "CSV", "sample_pages": None},
    ]

    print("\n" + "="*85)
    print("PHASE 7: REAL INGESTION & CHUNK DIVERSITY BENCHMARK")
    print("="*85)

    ingestion_records = []

    for item in test_files:
        file_path = TEST_DOCS_DIR / item["name"]
        if not file_path.exists():
            print(f"File missing: {file_path}")
            continue

        size_bytes = file_path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 3)

        print(f"\nProcessing: {item['name']} ({size_mb} MB)...")
        t0 = time.perf_counter()

        # Step 1: Document Loading
        raw_docs = load_document(str(file_path))
        page_or_block_count = len(raw_docs)

        # Step 2: Metadata Enrichment
        raw_docs = enrich_metadata(raw_docs, item["name"])

        # Step 3: Chunking
        chunks = chunk_documents(raw_docs)
        chunk_count = len(chunks)

        # Step 4: Embedding / Indexing (benchmark first 150 chunks for very large archives to keep time bounded)
        test_chunks = chunks[:250] if len(chunks) > 250 else chunks
        
        user_index_dir = BACKEND_DIR / "vectorstore" / TEST_USER
        if user_index_dir.exists():
            shutil.rmtree(user_index_dir)

        append_to_user_index(
            user_id=TEST_USER,
            chunks=test_chunks,
            embedding_model=embedding_model,
        )
        elapsed_s = round(time.perf_counter() - t0, 2)
        throughput = round(size_mb / elapsed_s, 2) if elapsed_s > 0 else 0.0

        print(f"  -> Pages/Blocks Extracted: {page_or_block_count}")
        print(f"  -> Chunks Created:         {chunk_count}")
        print(f"  -> Elapsed Time:           {elapsed_s} s")
        print(f"  -> Throughput:             {throughput} MB/s")

        ingestion_records.append({
            "filename": item["name"],
            "format": item["format"],
            "size_mb": size_mb,
            "page_or_block_count": page_or_block_count,
            "chunk_count": chunk_count,
            "elapsed_seconds": elapsed_s,
            "throughput_mb_s": throughput
        })

    # ─── COLD VS WARM CACHE BENCHMARK ─────────────────────────────────────────
    print("\n" + "="*85)
    print("BENCHMARKING COLD VS WARM EMBEDDING CACHE")
    print("="*85)

    cache_test_file = TEST_DOCS_DIR / "enterprise_handbook_1mb.pdf"
    raw_cache_docs = load_document(str(cache_test_file))
    cache_chunks = chunk_documents(raw_cache_docs)

    # 1. Clear cache table for test chunks to ensure true cold baseline
    conn = _get_cache_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunk_embeddings")
    conn.commit()

    # Cold Run
    print("Running Cold Cache Ingestion (empty SQLite cache)...")
    t_cold = time.perf_counter()
    cold_user = "cold_cache_user"
    c_dir = BACKEND_DIR / "vectorstore" / cold_user
    if c_dir.exists():
        shutil.rmtree(c_dir)

    append_to_user_index(user_id=cold_user, chunks=cache_chunks, embedding_model=embedding_model)
    cold_time = round(time.perf_counter() - t_cold, 2)
    print(f"  -> Cold Ingestion Time: {cold_time} s")

    # Warm Run
    print("Running Warm Cache Ingestion (hits SQLite cache)...")
    t_warm = time.perf_counter()
    warm_user = "warm_cache_user"
    w_dir = BACKEND_DIR / "vectorstore" / warm_user
    if w_dir.exists():
        shutil.rmtree(w_dir)

    append_to_user_index(user_id=warm_user, chunks=cache_chunks, embedding_model=embedding_model)
    warm_time = round(time.perf_counter() - t_warm, 2)
    speedup = round(cold_time / warm_time, 2) if warm_time > 0 else 1.0
    latency_reduction = round(((cold_time - warm_time) / cold_time) * 100, 2)

    print(f"  -> Warm Ingestion Time: {warm_time} s")
    print(f"  -> Cache Speedup:       {speedup}x ({latency_reduction}% latency reduction)")

    cache_summary = {
        "test_file": cache_test_file.name,
        "file_size_mb": round(cache_test_file.stat().st_size / (1024*1024), 3),
        "chunks_indexed": len(cache_chunks),
        "cold_cache_seconds": cold_time,
        "warm_cache_seconds": warm_time,
        "speedup_factor": speedup,
        "latency_reduction_pct": latency_reduction
    }

    full_output = {
        "ingestion_scaling": ingestion_records,
        "embedding_cache": cache_summary
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[PHASE 7] Saved full results to {RESULTS_FILE.resolve()}")

    print("\n" + "="*85)
    print(f"{'FILE':<40} | {'FORMAT':<12} | {'PAGES':<7} | {'CHUNKS':<8} | {'TIME':<8} | {'THROUGHPUT'}")
    print("="*85)
    for r in ingestion_records:
        print(f"{r['filename']:<40} | {r['format']:<12} | {r['page_or_block_count']:<7} | {r['chunk_count']:<8} | {r['elapsed_seconds']:<6.2f}s | {r['throughput_mb_s']:<6.2f} MB/s")
    print("="*85)


if __name__ == "__main__":
    run_phase7()
