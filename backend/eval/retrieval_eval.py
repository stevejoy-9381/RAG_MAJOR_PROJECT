"""
eval/retrieval_eval.py — Rigorous Retrieval Evaluation Suite (Phase 2)
────────────────────────────────────────────────────────────────────────
Evaluates:
  A) FAISS Dense Vector Search Only
  B) BM25 Sparse Keyword Search Only
  C) Hybrid (FAISS + BM25 Ensemble)
  D) Hybrid + CrossEncoder Neural Re-ranking

Computes:
  - Hit Rate@4: Proportion of queries where at least 1 relevant chunk is in top 4
  - MRR (Mean Reciprocal Rank): 1 / rank of first relevant chunk (0 if not in top 4)
  - Average Retrieval Latency (ms)
  - Grid Search over BM25/FAISS weights: (0.2/0.8), (0.4/0.6), (0.5/0.5), (0.8/0.2)

Saves results to eval/results/retrieval.csv and prints comparison tables.
"""

import os
import sys
import json
import time
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure backend root is on Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

import numpy as np

# Set deterministic random seeds
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
from src.retriever import rerank, get_user_index_path
from src.config import EMBEDDING_MODEL, RERANKER_MODEL, RETRIEVAL_CANDIDATES, FINAL_CONTEXT_K

EVAL_USER_ID = "eval_test_runner"
DOCS_DIR = BACKEND_DIR / "eval" / "docs"
QUESTIONS_FILE = BACKEND_DIR / "eval" / "questions.json"
RESULTS_DIR = BACKEND_DIR / "eval" / "results"
RESULTS_CSV = RESULTS_DIR / "retrieval.csv"


def prepare_eval_index() -> Tuple[FAISS, List[Document]]:
    """Ingest test documents in eval/docs into a clean isolated FAISS index."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = BACKEND_DIR / "vectorstore" / EVAL_USER_ID
    if index_path.exists():
        shutil.rmtree(index_path)

    doc_files = [
        "docmind_technical_architecture.txt",
        "cloudcorp_operations_and_policies.txt",
        "enterprise_sla_and_billing.txt",
    ]

    all_chunks: List[Document] = []
    print("\n[EVAL] Ingesting evaluation documents into isolated test store...")
    embedding_model = get_embedding_model()

    for doc_name in doc_files:
        doc_path = DOCS_DIR / doc_name
        if not doc_path.exists():
            raise FileNotFoundError(f"Missing evaluation document: {doc_path}")
        raw_docs = load_document(str(doc_path))
        raw_docs = enrich_metadata(raw_docs, doc_name)
        chunks = chunk_documents(raw_docs)
        all_chunks.extend(chunks)
        print(f"  - {doc_name}: {len(chunks)} chunks created.")

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
    print(f"[EVAL] Total indexed chunks in test FAISS store: {len(indexed_docs)}")
    return vectorstore, indexed_docs


def is_chunk_relevant(doc: Document, q_item: Dict[str, Any]) -> bool:
    """
    Determines if a retrieved chunk is ground-truth relevant.
    Matches against expected answer keywords, source file, and section content.
    """
    if q_item["type"] == "unanswerable":
        return False

    chunk_text = doc.page_content.lower()
    source_file = doc.metadata.get("source", "").lower()
    expected_source = q_item["source_file"].lower()

    # File match check
    if expected_source not in source_file:
        return False

    # Key entity/fact matching from expected answer
    expected_answer = q_item["expected_answer"].lower()
    
    # Extract salient tokens (filter out common stopwords)
    stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "of", "for", "with", "at", "by", "on", "from", "that", "this", "which", "per", "via"}
    expected_tokens = [tok.strip(".,;:()$%-") for tok in expected_answer.split() if tok.strip(".,;:()$%-") not in stopwords and len(tok.strip(".,;:()$%-")) > 2]
    
    if not expected_tokens:
        return True

    # Check match ratio of salient expected tokens
    matches = sum(1 for tok in expected_tokens if tok in chunk_text)
    match_ratio = matches / len(expected_tokens)
    
    # If more than 40% of salient key tokens from expected answer exist in chunk
    return match_ratio >= 0.40


def evaluate_query(
    retrieved_docs: List[Document],
    q_item: Dict[str, Any],
    top_k: int = 4
) -> Tuple[int, float, int]:
    """
    Evaluates retrieved docs for a single query.
    Returns: (hit_at_k (0 or 1), reciprocal_rank (0.0 to 1.0), rank_of_first_hit (1-based or -1))
    """
    if q_item["type"] == "unanswerable":
        return (0, 0.0, -1)

    top_docs = retrieved_docs[:top_k]
    for rank, doc in enumerate(top_docs, start=1):
        if is_chunk_relevant(doc, q_item):
            return (1, 1.0 / rank, rank)

    return (0, 0.0, -1)


def run_retrieval_evaluation():
    """Execute evaluation across Setups A, B, C, D and parameter grid search."""
    vectorstore, indexed_docs = prepare_eval_index()

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Filter answerable questions for Hit Rate and MRR evaluation
    answerable_qs = [q for q in questions if q["type"] != "unanswerable"]
    total_answerable = len(answerable_qs)
    print(f"\n[EVAL] Running evaluation on {len(questions)} total questions ({total_answerable} answerable, {len(questions) - total_answerable} unanswerable)...")

    # Base Retrievers
    faiss_retriever_4 = vectorstore.as_retriever(search_kwargs={"k": 4})
    faiss_retriever_10 = vectorstore.as_retriever(search_kwargs={"k": 10})

    bm25_retriever_4 = BM25Retriever.from_documents(indexed_docs)
    bm25_retriever_4.k = 4

    bm25_retriever_10 = BM25Retriever.from_documents(indexed_docs)
    bm25_retriever_10.k = 10

    # Default Hybrid (BM25 0.4, FAISS 0.6)
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
        "C: Hybrid (FAISS + BM25)": {"mode": "hybrid"},
        "D: Hybrid + CrossEncoder": {"mode": "hybrid_rerank"},
    }

    per_query_results = []
    summary_metrics = {}

    for setup_name, config in setups.items():
        hits = 0
        rr_sum = 0.0
        latencies_ms = []

        print(f"\n[EVAL] Benchmarking Setup: {setup_name}...")

        for q in questions:
            query_str = q["question"]
            t0 = time.perf_counter()

            if config["mode"] == "faiss_only":
                docs = faiss_retriever_4.invoke(query_str)
            elif config["mode"] == "bm25_only":
                docs = bm25_retriever_4.invoke(query_str)
            elif config["mode"] == "hybrid":
                docs = hybrid_retriever_4.invoke(query_str)
            elif config["mode"] == "hybrid_rerank":
                candidates = hybrid_retriever_10.invoke(query_str)
                # Re-rank down to top 4
                reranked_pairs = rerank(query_str, candidates)
                docs = [doc for doc, _score in reranked_pairs]

            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)

            hit, mrr, rank = evaluate_query(docs, q, top_k=4)

            if q["type"] != "unanswerable":
                hits += hit
                rr_sum += mrr

            per_query_results.append({
                "setup": setup_name,
                "question_id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "hit_at_4": hit if q["type"] != "unanswerable" else "N/A",
                "mrr": round(mrr, 4) if q["type"] != "unanswerable" else "N/A",
                "first_hit_rank": rank if rank != -1 else "Miss",
                "latency_ms": round(elapsed_ms, 2),
                "top1_source": docs[0].metadata.get("source", "N/A") if docs else "None",
            })

        avg_hit_rate = hits / total_answerable
        avg_mrr = rr_sum / total_answerable
        avg_latency = sum(latencies_ms) / len(latencies_ms)

        summary_metrics[setup_name] = {
            "Hit Rate@4": f"{avg_hit_rate * 100:.2f}%",
            "MRR": f"{avg_mrr:.4f}",
            "Avg Latency (ms)": f"{avg_latency:.2f} ms",
            "_raw_hit": avg_hit_rate,
            "_raw_mrr": avg_mrr,
            "_raw_lat": avg_latency,
        }

    # ─── Grid Search over BM25 / FAISS Weights ────────────────────────────────
    print("\n[EVAL] Running Grid Search over BM25 / FAISS weights...")
    weight_combinations = [
        (0.2, 0.8),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.8, 0.2),
    ]

    weight_results = []
    best_weight = None
    best_mrr = -1.0

    for bm25_w, faiss_w in weight_combinations:
        grid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever_4, faiss_retriever_4],
            weights=[bm25_w, faiss_w],
        )
        hits = 0
        rr_sum = 0.0
        lats = []

        for q in questions:
            t0 = time.perf_counter()
            docs = grid_retriever.invoke(q["question"])
            elapsed_ms = (time.perf_counter() - t0) * 1000
            lats.append(elapsed_ms)

            hit, mrr, _ = evaluate_query(docs, q, top_k=4)
            if q["type"] != "unanswerable":
                hits += hit
                rr_sum += mrr

        hit_rate = hits / total_answerable
        mrr = rr_sum / total_answerable
        avg_lat = sum(lats) / len(lats)

        if mrr > best_mrr:
            best_mrr = mrr
            best_weight = (bm25_w, faiss_w)

        weight_results.append({
            "Weights (BM25 / FAISS)": f"{bm25_w} / {faiss_w}",
            "Hit Rate@4": f"{hit_rate * 100:.2f}%",
            "MRR": f"{mrr:.4f}",
            "Avg Latency (ms)": f"{avg_lat:.2f} ms",
        })

    # Save detailed per-query results
    df_results = pd.DataFrame(per_query_results)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"\n[EVAL] Per-query evaluation records saved to: {RESULTS_CSV}")

    # Build and print Summary Tables
    print("\n" + "="*80)
    print("PHASE 2: RETRIEVAL EVALUATION RESULTS (Setups A to D)")
    print("="*80)
    df_summary = pd.DataFrame.from_dict(
        {k: {m: v for m, v in vals.items() if not m.startswith("_")} for k, vals in summary_metrics.items()},
        orient="index"
    )
    print(df_summary.to_string())

    print("\n" + "="*80)
    print("HYBRID RETRIEVAL WEIGHT GRID SEARCH")
    print("="*80)
    df_weights = pd.DataFrame(weight_results)
    print(df_weights.to_string(index=False))
    print(f"\nOptimal Weight Configuration: BM25 = {best_weight[0]} / FAISS = {best_weight[1]} (MRR: {best_mrr:.4f})")
    print("="*80 + "\n")

    return summary_metrics, weight_results, best_weight


if __name__ == "__main__":
    run_retrieval_evaluation()
