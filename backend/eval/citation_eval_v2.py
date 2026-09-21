"""
eval/citation_eval_v2.py — Phase 5: Citation Accuracy Fix & Benchmark
────────────────────────────────────────────────────────────────────
Benchmarks citation accuracy across all 60 questions (eval/questions_v2.json):
  - BEFORE: Legacy prompt without chunk source labeling, no citation requirement, no fallback.
  - AFTER:  Updated prompt with [Source: filename, Page: N] chunk headers,
            mandatory citation instructions, and post-generation fallback injection.

Reports citation accuracy (%) before and after.
Saves results to eval/results_v2/citations.csv and eval/results_v2/citations_summary.json.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

from src.ingest import get_embedding_model
from src.llm import get_llm
from src.config import FINAL_CONTEXT_K

EVAL_USER_ID = "eval_v2_runner"
QUESTIONS_FILE = BACKEND_DIR / "eval" / "questions_v2.json"
RESULTS_DIR = BACKEND_DIR / "eval" / "results_v2"


def format_context_legacy(docs: List[Document]) -> str:
    """Legacy unlabelled chunk format."""
    return "\n\n".join([doc.page_content.strip()[:300] for doc in docs[:2]])


def format_context_with_labels(docs: List[Document]) -> str:
    """Updated format: each chunk labelled with [Source: filename, Page: N]."""
    formatted = []
    for doc in docs[:2]:
        src = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", doc.metadata.get("page_number", 1))
        formatted.append(f"[Source: {src}, Page: {page}]\n{doc.page_content.strip()[:300]}")
    return "\n\n".join(formatted)


def ensure_citations(generated_answer: str, retrieved_docs: List[Document]) -> Tuple[str, bool]:
    """
    Check if any source filename appears in answer.
    If none appears and answer is not a refusal, automatically append retrieved sources.
    Returns (final_answer, fallback_applied_bool).
    """
    if "i don't know" in generated_answer.lower():
        return generated_answer, False

    doc_sources = []
    for d in retrieved_docs:
        src = d.metadata.get("source", "")
        if src and src not in doc_sources:
            doc_sources.append(src)

    has_citation = any(src.lower() in generated_answer.lower() for src in doc_sources)
    if not has_citation and doc_sources:
        sources_str = ", ".join([f"[Source: {s}]" for s in doc_sources])
        updated_answer = f"{generated_answer.strip()}\n\nSources: {sources_str}"
        return updated_answer, True

    return generated_answer, False


def check_citation_accuracy(answer: str, q_item: Dict[str, Any]) -> int:
    """
    Evaluates whether the response correctly cites the expected source file.
    For unanswerable queries: refusal counts as 1.
    For answerable queries: checks if any expected source filename is present in the answer.
    """
    if q_item["type"] == "unanswerable":
        return 1 if "i don't know" in answer.lower() else 0

    expected_sources = [s.strip().lower() for s in q_item["source_file"].split(";") if s.strip()]
    ans_lower = answer.lower()
    for exp_src in expected_sources:
        if exp_src in ans_lower:
            return 1

    return 0


def run_phase5():
    print("\n[PHASE 5] Initializing FAISS store and Groq LLM...")
    embedding_model = get_embedding_model()
    index_path = BACKEND_DIR / "vectorstore" / EVAL_USER_ID / "faiss_index"
    vectorstore = FAISS.load_local(str(index_path), embedding_model, allow_dangerous_deserialization=True)
    indexed_docs = list(vectorstore.docstore._dict.values())

    bm25 = BM25Retriever.from_documents(indexed_docs)
    bm25.k = 6

    llm = get_llm()
    llm.max_tokens = 90

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total_questions = len(questions)
    print(f"[PHASE 5] Benchmarking citation accuracy on {total_questions} questions (Before vs After)...")

    results_records = []
    before_citations_count = 0
    after_citations_count = 0
    fallbacks_triggered = 0

    legacy_template = """You are a precise document Q&A assistant.
Your job is to answer the user's question using ONLY the context provided below.
If the answer is not present in the context, respond with:
"I don't know — this isn't covered in the uploaded documents."

Context:
────────────────────────────────────
{context}
────────────────────────────────────

Question: {question}

Answer:"""

    updated_template = """You are a precise document Q&A assistant.
Your job is to answer the user's question using ONLY the context provided below.
If the answer is not present in the context, respond with:
"I don't know — this isn't covered in the uploaded documents."

Context:
────────────────────────────────────
{context}
────────────────────────────────────

Question: {question}

Instructions:
- Answer directly, factually, and concisely based strictly on the context above.
- CITATION RULE: Each retrieved chunk is labeled with [Source: filename, Page: N]. You MUST cite the source filename in brackets (e.g. [Source: filename.ext]) directly after each factual claim or statement.
- If multiple chunks support your answer, cite each relevant filename.
- If the answer is not in the context, strictly state: "I don't know — this isn't covered in the uploaded documents."

Answer:"""

    for i, q in enumerate(questions, start=1):
        q_id = q["id"]
        q_text = q["question"]
        q_type = q["type"]

        # Retrieve top 4 hybrid chunks
        faiss_docs = vectorstore.similarity_search(q_text, k=4)
        bm25_docs = bm25.invoke(q_text)
        
        # Deduplicate
        seen = set()
        retrieved_chunks = []
        for d in faiss_docs + bm25_docs:
            h = d.page_content.strip()
            if h not in seen:
                seen.add(h)
                retrieved_chunks.append(d)
                if len(retrieved_chunks) == FINAL_CONTEXT_K:
                    break

        # 1. Condition: BEFORE (Legacy Prompt)
        legacy_prompt = legacy_template.format(
            context=format_context_legacy(retrieved_chunks),
            question=q_text
        )
        try:
            resp_before = llm.invoke(legacy_prompt).content.strip()
        except Exception as e:
            time.sleep(2.0)
            resp_before = llm.invoke(legacy_prompt).content.strip()

        cite_acc_before = check_citation_accuracy(resp_before, q)
        before_citations_count += cite_acc_before

        time.sleep(0.5)  # Rate pacing for Groq

        # 2. Condition: AFTER (Updated Prompt + Post-Generation Fallback)
        updated_prompt = updated_template.format(
            context=format_context_with_labels(retrieved_chunks),
            question=q_text
        )
        try:
            resp_after_raw = llm.invoke(updated_prompt).content.strip()
        except Exception as e:
            time.sleep(2.0)
            resp_after_raw = llm.invoke(updated_prompt).content.strip()

        resp_after, fallback_used = ensure_citations(resp_after_raw, retrieved_chunks)
        if fallback_used:
            fallbacks_triggered += 1

        cite_acc_after = check_citation_accuracy(resp_after, q)
        after_citations_count += cite_acc_after

        results_records.append({
            "question_id": q_id,
            "type": q_type,
            "question": q_text,
            "expected_source": q["source_file"],
            "citation_before": cite_acc_before,
            "citation_after": cite_acc_after,
            "fallback_used": fallback_used,
            "answer_before": resp_before.replace("\n", " ")[:120],
            "answer_after": resp_after.replace("\n", " ")[:120],
        })

        print(f"[{i:02d}/60] Q{q_id:<2} ({q_type:<14}): Before={cite_acc_before} | After={cite_acc_after} (Fallback={fallback_used})")
        time.sleep(0.5)

    before_pct = round((before_citations_count / total_questions) * 100, 2)
    after_pct = round((after_citations_count / total_questions) * 100, 2)
    fallback_pct = round((fallbacks_triggered / total_questions) * 100, 2)

    summary = {
        "total_questions": total_questions,
        "citation_accuracy_before_pct": before_pct,
        "citation_accuracy_after_pct": after_pct,
        "absolute_improvement_pct": round(after_pct - before_pct, 2),
        "post_generation_fallbacks_applied": fallbacks_triggered,
        "fallback_rate_pct": fallback_pct,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "citations_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(results_records).to_csv(RESULTS_DIR / "citations.csv", index=False)
    print(f"\n[PHASE 5] Saved results to {RESULTS_DIR / 'citations.csv'} and {RESULTS_DIR / 'citations_summary.json'}")

    print("\n" + "="*70)
    print(f"{'METRIC':<40} | {'BEFORE':<12} | {'AFTER':<12}")
    print("="*70)
    print(f"{'Overall Citation Accuracy':<40} | {before_pct:<10.2f}% | {after_pct:<10.2f}%")
    print(f"{'Valid Citations Count':<40} | {before_citations_count}/{total_questions:<9} | {after_citations_count}/{total_questions}")
    print(f"{'Post-Generation Fallbacks Triggered':<40} | {'N/A':<12} | {fallbacks_triggered} ({fallback_pct}%)")
    print("="*70)


if __name__ == "__main__":
    run_phase5()
