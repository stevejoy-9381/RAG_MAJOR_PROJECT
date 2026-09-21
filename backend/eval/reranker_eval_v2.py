"""
eval/reranker_eval_v2.py — Phase 4: Reduce Re-ranker Latency
────────────────────────────────────────────────────────────
Compares 4 re-ranking optimization strategies on the 60-question hard set:
  1) Variant A: Re-rank top 10 (PyTorch CrossEncoder - Current Baseline)
  2) Variant B: Re-rank top 6 (PyTorch CrossEncoder)
  3) Variant C: Re-rank top 6 (ONNX INT8-Quantized CrossEncoder)
  4) Variant D: Re-rank top 6 (ONNX INT8 + Score-Gap Threshold)
  5) No Re-ranking: Hybrid Only (Top 4 directly from Ensemble)

Outputs:
  - Hit Rate@4 (%)
  - MRR
  - Latency (ms)
  - Speedup factor vs baseline
  - Plain recommendation on whether re-ranking justifies the latency

Saves results to eval/results_v2/reranker_perf.json and eval/results_v2/reranker_eval.csv.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
import onnxruntime as ort

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

from src.ingest import get_embedding_model
from src.retriever import _get_cross_encoder
from src.config import RETRIEVAL_CANDIDATES, FINAL_CONTEXT_K

EVAL_USER_ID = "eval_v2_runner"
QUESTIONS_FILE = BACKEND_DIR / "eval" / "questions_v2.json"
RESULTS_DIR = BACKEND_DIR / "eval" / "results_v2"
ONNX_INT8_PATH = BACKEND_DIR / "eval" / "models" / "cross_encoder_int8.onnx"


class ONNXCrossEncoderWrapper:
    def __init__(self, model_path: str, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        queries = [p[0] for p in pairs]
        passages = [p[1] for p in pairs]
        inputs = self.tokenizer(
            queries, passages, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run(None, feed)
        logits = outputs[0].squeeze(-1)
        if logits.ndim == 0:
            return [float(logits)]
        return [float(x) for x in logits]


def is_chunk_relevant(doc: Document, q_item: Dict[str, Any]) -> bool:
    if q_item["type"] == "unanswerable":
        return False
    chunk_text = doc.page_content.lower()
    source_file = doc.metadata.get("source", "").lower()
    expected_sources = [s.strip().lower() for s in q_item["source_file"].split(";") if s.strip()]
    if not any(exp in source_file for exp in expected_sources):
        return False

    expected_answer = q_item["expected_answer"].lower()
    stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "of", "for", "with", "at", "by", "on", "from", "that", "this", "which", "per", "via", "then", "under"}
    tokens = [tok.strip(".,;:()$%-[]") for tok in expected_answer.split() if tok.strip(".,;:()$%-[]") not in stopwords and len(tok.strip(".,;:()$%-[]")) > 2]
    if not tokens:
        return True
    matches = sum(1 for tok in tokens if tok in chunk_text)
    return (matches / len(tokens)) >= 0.25


def evaluate_query(retrieved_docs: List[Document], q_item: Dict[str, Any], top_k: int = 4) -> Tuple[int, float, int]:
    if q_item["type"] == "unanswerable":
        return (0, 0.0, -1)
    for rank, doc in enumerate(retrieved_docs[:top_k], start=1):
        if is_chunk_relevant(doc, q_item):
            return (1, 1.0 / rank, rank)
    return (0, 0.0, -1)


def compute_hybrid_candidates_with_scores(
    query: str,
    vectorstore: FAISS,
    bm25_retriever: BM25Retriever,
    k: int = 10,
    faiss_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> Tuple[List[Document], List[float]]:
    """Compute candidates and their combined normalized hybrid scores for threshold evaluation."""
    # 1. FAISS similarity with scores
    faiss_results = vectorstore.similarity_search_with_score(query, k=k)
    # Cosine distance to similarity (normalized)
    faiss_docs = [doc for doc, score in faiss_results]
    faiss_scores = [1.0 / (1.0 + max(0.0, float(score))) for doc, score in faiss_results]

    # 2. BM25 results
    bm25_docs = bm25_retriever.invoke(query)
    
    # Merge and score
    doc_map: Dict[str, Document] = {}
    combined_scores: Dict[str, float] = {}

    for doc, s in zip(faiss_docs, faiss_scores):
        doc_id = doc.page_content.strip()
        doc_map[doc_id] = doc
        combined_scores[doc_id] = combined_scores.get(doc_id, 0.0) + faiss_weight * s

    bm25_rank_weight = 1.0
    for idx, doc in enumerate(bm25_docs[:k]):
        doc_id = doc.page_content.strip()
        doc_map[doc_id] = doc
        # Reciprocal rank contribution
        bm25_s = bm25_rank_weight / (idx + 1)
        combined_scores[doc_id] = combined_scores.get(doc_id, 0.0) + bm25_weight * bm25_s

    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_docs = [doc_map[item[0]] for item in sorted_items[:k]]
    sorted_scores = [item[1] for item in sorted_items[:k]]
    return sorted_docs, sorted_scores


def run_phase4():
    print("\n[PHASE 4] Loading test FAISS vectorstore and benchmark queries...")
    embedding_model = get_embedding_model()
    index_path = BACKEND_DIR / "vectorstore" / EVAL_USER_ID / "faiss_index"
    vectorstore = FAISS.load_local(str(index_path), embedding_model, allow_dangerous_deserialization=True)
    indexed_docs = list(vectorstore.docstore._dict.values())

    bm25_10 = BM25Retriever.from_documents(indexed_docs)
    bm25_10.k = 10

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    answerable_qs = [q for q in questions if q["type"] != "unanswerable"]
    total_ans = len(answerable_qs)

    pytorch_ce = _get_cross_encoder()
    onnx_ce = ONNXCrossEncoderWrapper(str(ONNX_INT8_PATH))

    variants = [
        {"name": "No Re-ranking (Hybrid Top 4)", "mode": "none", "k": 4},
        {"name": "Variant A: Top 10 (PyTorch CrossEncoder)", "mode": "pytorch", "k": 10},
        {"name": "Variant B: Top 6 (PyTorch CrossEncoder)", "mode": "pytorch", "k": 6},
        {"name": "Variant C: Top 6 (ONNX INT8 Quantized)", "mode": "onnx_int8", "k": 6},
        {"name": "Variant D: Top 6 (ONNX INT8 + Gap Threshold 0.20)", "mode": "onnx_int8_threshold", "k": 6, "threshold": 0.20},
    ]

    results_summary = {}
    csv_records = []

    for v in variants:
        v_name = v["name"]
        print(f"\nBenchmarking: {v_name}...")
        hits = 0
        rr_sum = 0.0
        latencies = []
        skipped_rerank_count = 0

        for q in questions:
            q_text = q["question"]
            t0 = time.perf_counter()

            if v["mode"] == "none":
                docs, _ = compute_hybrid_candidates_with_scores(q_text, vectorstore, bm25_10, k=4)
                final_docs = docs[:4]

            elif v["mode"] == "pytorch":
                candidates, _ = compute_hybrid_candidates_with_scores(q_text, vectorstore, bm25_10, k=v["k"])
                pairs = [(q_text, d.page_content) for d in candidates]
                scores = pytorch_ce.predict(pairs)
                scored = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
                final_docs = [d for d, s in scored[:FINAL_CONTEXT_K]]

            elif v["mode"] == "onnx_int8":
                candidates, _ = compute_hybrid_candidates_with_scores(q_text, vectorstore, bm25_10, k=v["k"])
                pairs = [(q_text, d.page_content) for d in candidates]
                scores = onnx_ce.predict(pairs)
                scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                final_docs = [d for d, s in scored[:FINAL_CONTEXT_K]]

            elif v["mode"] == "onnx_int8_threshold":
                candidates, hybrid_scores = compute_hybrid_candidates_with_scores(q_text, vectorstore, bm25_10, k=v["k"])
                # Check score gap between top 1 and top 2
                score_gap = (hybrid_scores[0] - hybrid_scores[1]) if len(hybrid_scores) > 1 else 1.0
                if score_gap >= v["threshold"]:
                    # Skip re-ranking!
                    skipped_rerank_count += 1
                    final_docs = candidates[:FINAL_CONTEXT_K]
                else:
                    pairs = [(q_text, d.page_content) for d in candidates]
                    scores = onnx_ce.predict(pairs)
                    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                    final_docs = [d for d, s in scored[:FINAL_CONTEXT_K]]

            lat_ms = (time.perf_counter() - t0) * 1000
            latencies.append(lat_ms)

            hit, mrr, rank = evaluate_query(final_docs, q, top_k=4)
            if q["type"] != "unanswerable":
                hits += hit
                rr_sum += mrr

            csv_records.append({
                "variant": v_name,
                "question_id": q["id"],
                "type": q["type"],
                "hit_at_4": hit if q["type"] != "unanswerable" else 0,
                "mrr": round(mrr, 4) if q["type"] != "unanswerable" else 0.0,
                "rank": rank,
                "latency_ms": round(lat_ms, 2)
            })

        hit_rate = (hits / total_ans) * 100
        avg_mrr = rr_sum / total_ans
        avg_lat = sum(latencies) / len(latencies)

        results_summary[v_name] = {
            "hit_at_4_pct": round(hit_rate, 2),
            "mrr": round(avg_mrr, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "skipped_rerank_pct": round((skipped_rerank_count / len(questions)) * 100, 2) if v["mode"] == "onnx_int8_threshold" else 0.0
        }

    # Speedup calculation relative to Variant A
    base_lat = results_summary["Variant A: Top 10 (PyTorch CrossEncoder)"]["avg_latency_ms"]
    for k, v in results_summary.items():
        v["speedup_factor"] = round(base_lat / v["avg_latency_ms"], 2) if v["avg_latency_ms"] > 0 else 1.0

    # Save to disk
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "reranker_perf.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    pd.DataFrame(csv_records).to_csv(RESULTS_DIR / "reranker_eval.csv", index=False)
    print(f"\n[PHASE 4] Results saved to {RESULTS_DIR / 'reranker_perf.json'} and {RESULTS_DIR / 'reranker_eval.csv'}")

    print("\n" + "="*95)
    print(f"{'VARIANT':<44} | {'HIT@4 (%)':<10} | {'MRR':<8} | {'AVG LATENCY':<12} | {'SPEEDUP':<8}")
    print("="*95)
    for name, data in results_summary.items():
        print(f"{name:<44} | {data['hit_at_4_pct']:<10.2f} | {data['mrr']:<8.4f} | {data['avg_latency_ms']:<10.2f} ms | {data['speedup_factor']:<6.2f}x")
    print("="*95)


if __name__ == "__main__":
    run_phase4()
