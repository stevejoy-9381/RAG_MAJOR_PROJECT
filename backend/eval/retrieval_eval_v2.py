"""
eval/retrieval_eval_v2.py — Harder Retrieval Benchmark (Phase 3)
───────────────────────────────────────────────────────────────
Evaluates 4 retrieval setups across 15 real documents in eval/docs_v2
using the 60-question hard benchmark (eval/questions_v2.json):
  A) FAISS Dense Vector Search Only
  B) BM25 Sparse Keyword Search Only
  C) Hybrid (FAISS 0.6 + BM25 0.4)
  D) Hybrid + CrossEncoder Re-ranking

Outputs metrics overall and split by question type:
  - paraphrased (20)
  - keyword_heavy (15)
  - multi_chunk (10)
  - near_miss (10)
  - unanswerable (5)

Saves results to eval/results_v2/retrieval.csv and eval/results_v2/retrieval_summary.json.
"""

import os
import sys
import json
import time
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except Exception:
    pass

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

from src.ingest import load_document, enrich_metadata, chunk_documents, append_to_user_index, get_embedding_model
from src.retriever import rerank
from src.config import RETRIEVAL_CANDIDATES, FINAL_CONTEXT_K

EVAL_USER_ID = "eval_v2_runner"
DOCS_DIR = BACKEND_DIR / "eval" / "docs_v2"
QUESTIONS_FILE = BACKEND_DIR / "eval" / "questions_v2.json"
RESULTS_DIR = BACKEND_DIR / "eval" / "results_v2"
RESULTS_CSV = RESULTS_DIR / "retrieval.csv"
RESULTS_JSON = RESULTS_DIR / "retrieval_summary.json"


def prepare_eval_index_v2() -> Tuple[FAISS, List[Document]]:
    """Ingest all documents in eval/docs_v2 into a dedicated FAISS test index."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = BACKEND_DIR / "vectorstore" / EVAL_USER_ID
    if index_path.exists():
        shutil.rmtree(index_path)

    doc_files = sorted(list(DOCS_DIR.iterdir()))
    if not doc_files:
        raise FileNotFoundError(f"No documents found in {DOCS_DIR}")

    all_chunks: List[Document] = []
    print(f"\n[PHASE 3] Ingesting {len(doc_files)} evaluation documents from {DOCS_DIR.name}...")
    embedding_model = get_embedding_model()

    for doc_path in doc_files:
        raw_docs = load_document(str(doc_path))
        raw_docs = enrich_metadata(raw_docs, doc_path.name)
        chunks = chunk_documents(raw_docs)
        all_chunks.extend(chunks)
        print(f"  - {doc_path.name:<48}: {len(chunks)} chunks created.")

    append_to_user_index(
        user_id=EVAL_USER_ID,
        chunks=all_chunks,
        embedding_model=embedding_model,
    )

    vectorstore = FAISS.load_local(
        str(index_path / "faiss_index"),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    indexed_docs = list(vectorstore.docstore._dict.values())
    print(f"[PHASE 3] Total indexed chunks in test FAISS store: {len(indexed_docs)}")
    return vectorstore, indexed_docs


def is_chunk_relevant(doc: Document, q_item: Dict[str, Any]) -> bool:
    """
    Determines if retrieved chunk matches expected ground truth.
    Handles single or multi-source targets and entity keyword overlap.
    """
    if q_item["type"] == "unanswerable":
        return False

    chunk_text = doc.page_content.lower()
    source_file = doc.metadata.get("source", "").lower()
    expected_sources = [s.strip().lower() for s in q_item["source_file"].split(";") if s.strip()]

    # Source file match check
    source_matched = any(exp in source_file for exp in expected_sources)
    if not source_matched:
        return False

    # Key entity matching from expected answer
    expected_answer = q_item["expected_answer"].lower()
    stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "of", "for", "with", "at", "by", "on", "from", "that", "this", "which", "per", "via", "then", "under"}
    tokens = [tok.strip(".,;:()$%-[]") for tok in expected_answer.split() if tok.strip(".,;:()$%-[]") not in stopwords and len(tok.strip(".,;:()$%-[]")) > 2]
    
    if not tokens:
        return True

    matches = sum(1 for tok in tokens if tok in chunk_text)
    match_ratio = matches / len(tokens)
    return match_ratio >= 0.25


def evaluate_query(retrieved_docs: List[Document], q_item: Dict[str, Any], top_k: int = 4) -> Tuple[int, float, int]:
    if q_item["type"] == "unanswerable":
        return (0, 0.0, -1)

    top_docs = retrieved_docs[:top_k]
    for rank, doc in enumerate(top_docs, start=1):
        if is_chunk_relevant(doc, q_item):
            return (1, 1.0 / rank, rank)

    return (0, 0.0, -1)


def run_phase3_evaluation():
    vectorstore, indexed_docs = prepare_eval_index_v2()

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    answerable_qs = [q for q in questions if q["type"] != "unanswerable"]
    total_answerable = len(answerable_qs)
    print(f"\n[PHASE 3] Running retrieval benchmark on {len(questions)} total questions ({total_answerable} answerable, {len(questions)-total_answerable} unanswerable)...")

    # Set up retrievers
    faiss_retriever_4 = vectorstore.as_retriever(search_kwargs={"k": 4})
    faiss_retriever_10 = vectorstore.as_retriever(search_kwargs={"k": 10})

    bm25_retriever_4 = BM25Retriever.from_documents(indexed_docs)
    bm25_retriever_4.k = 4

    bm25_retriever_10 = BM25Retriever.from_documents(indexed_docs)
    bm25_retriever_10.k = 10

    hybrid_retriever_4 = EnsembleRetriever(
        retrievers=[bm25_retriever_4, faiss_retriever_4],
        weights=[0.4, 0.6],
    )
    hybrid_retriever_10 = EnsembleRetriever(
        retrievers=[bm25_retriever_10, faiss_retriever_10],
        weights=[0.4, 0.6],
    )

    setups = {
        "A: FAISS Only": {"mode": "faiss_only"},
        "B: BM25 Only": {"mode": "bm25_only"},
        "C: Hybrid (FAISS 0.6 + BM25 0.4)": {"mode": "hybrid"},
        "D: Hybrid + CrossEncoder": {"mode": "hybrid_rerank"},
    }

    per_query_records = []
    summary_results = {}

    question_types = ["paraphrased", "keyword_heavy", "multi_chunk", "near_miss"]

    for setup_name, config in setups.items():
        print(f"\nEvaluating: {setup_name}...")
        type_hits = {t: 0 for t in question_types}
        type_rr = {t: 0.0 for t in question_types}
        type_counts = {t: 0 for t in question_types}
        type_latencies = {t: [] for t in question_types}
        
        all_latencies = []
        overall_hits = 0
        overall_rr = 0.0

        for q in questions:
            q_id = q["id"]
            q_type = q["type"]
            q_text = q["question"]

            t0 = time.perf_counter()
            if config["mode"] == "faiss_only":
                docs = faiss_retriever_4.invoke(q_text)
            elif config["mode"] == "bm25_only":
                docs = bm25_retriever_4.invoke(q_text)
            elif config["mode"] == "hybrid":
                docs = hybrid_retriever_4.invoke(q_text)
            elif config["mode"] == "hybrid_rerank":
                candidates = hybrid_retriever_10.invoke(q_text)
                reranked_pairs = rerank(q_text, candidates)
                docs = [doc for doc, _score in reranked_pairs]

            elapsed_ms = (time.perf_counter() - t0) * 1000
            all_latencies.append(elapsed_ms)

            hit, mrr, rank = evaluate_query(docs, q, top_k=4)

            if q_type != "unanswerable":
                overall_hits += hit
                overall_rr += mrr
                type_hits[q_type] += hit
                type_rr[q_type] += mrr
                type_counts[q_type] += 1
                type_latencies[q_type].append(elapsed_ms)

            per_query_records.append({
                "setup": setup_name,
                "question_id": q_id,
                "type": q_type,
                "question": q_text,
                "hit_at_4": hit if q_type != "unanswerable" else 0,
                "mrr": round(mrr, 4) if q_type != "unanswerable" else 0.0,
                "first_hit_rank": rank if rank != -1 else "Miss",
                "latency_ms": round(elapsed_ms, 2),
                "top1_source": docs[0].metadata.get("source", "None") if docs else "None",
            })

        overall_hit_rate = overall_hits / total_answerable
        overall_mrr = overall_rr / total_answerable
        overall_lat = sum(all_latencies) / len(all_latencies)

        type_breakdown = {}
        for t in question_types:
            c = type_counts[t]
            type_breakdown[t] = {
                "count": c,
                "hit_at_4": round((type_hits[t] / c) * 100, 2) if c > 0 else 0.0,
                "mrr": round(type_rr[t] / c, 4) if c > 0 else 0.0,
                "latency_ms": round(sum(type_latencies[t]) / c, 2) if c > 0 else 0.0,
            }

        summary_results[setup_name] = {
            "overall_hit_at_4_pct": round(overall_hit_rate * 100, 2),
            "overall_mrr": round(overall_mrr, 4),
            "overall_latency_ms": round(overall_lat, 2),
            "by_type": type_breakdown,
        }

    # Save outputs
    df = pd.DataFrame(per_query_records)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[PHASE 3] Saved detailed query results to {RESULTS_CSV.resolve()}")

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"[PHASE 3] Saved summary metrics to {RESULTS_JSON.resolve()}")

    # Print summary tables
    print("\n" + "="*80)
    print(f"{'SETUP':<32} | {'HIT@4 (%)':<10} | {'MRR':<8} | {'AVG LATENCY':<12}")
    print("="*80)
    for setup, data in summary_results.items():
        print(f"{setup:<32} | {data['overall_hit_at_4_pct']:<10.2f} | {data['overall_mrr']:<8.4f} | {data['overall_latency_ms']:<10.2f} ms")
    print("="*80)

    print("\n--- BREAKDOWN BY QUESTION TYPE ---")
    for q_type in question_types:
        print(f"\n[Question Type: {q_type.upper()}]")
        print(f"{'SETUP':<32} | {'HIT@4 (%)':<10} | {'MRR':<8} | {'LATENCY':<12}")
        print("-" * 68)
        for setup, data in summary_results.items():
            t_data = data["by_type"][q_type]
            print(f"{setup:<32} | {t_data['hit_at_4']:<10.2f} | {t_data['mrr']:<8.4f} | {t_data['latency_ms']:<10.2f} ms")

if __name__ == "__main__":
    run_phase3_evaluation()
