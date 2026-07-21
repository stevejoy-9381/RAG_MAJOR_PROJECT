"""
src/benchmark.py — Model Benchmarking (Phase 8)
─────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Runs the same questions through multiple Ollama models using the existing
  retrieval + reranking pipeline, records latency and response quality,
  and stores results in SQLite for comparison and reporting.

WHY BENCHMARK?
  When multiple local models are available (e.g. llama3.1:8b, phi3:mini,
  mistral:7b), you want to compare them on YOUR documents — not just
  generic benchmarks. This tool lets you see which model gives better
  answers for your specific content, with concrete latency numbers.

PIPELINE:
  For each question:
    1. Retrieve + rerank documents ONCE (shared across all models)
    2. Build the context string
    3. For each model:
       a. Create an OllamaProvider with that model name
       b. Call provider.chat(messages, stream=False)
       c. Record: answer, latency, approximate token count
    4. Save all results to SQLite

STORAGE:
  Uses the same vectorstore/chat_history.db as conversation_store.py.
  New table: benchmark_runs (one row per model×question pair).
  Results are grouped by run_group_id for easy retrieval.

CONNECTIONS:
  → Called by api.py's /benchmark/run endpoint
  → Uses src/retriever.py for retrieval (same pipeline as /chat)
  → Uses src/llm_provider.OllamaProvider for model-specific calls
"""

import os
import json
import uuid
import time
import sqlite3
import threading
from datetime import datetime, timezone

import httpx

from src.retriever import retrieve_and_rerank, format_sources


# ─── Database ─────────────────────────────────────────────────────────────────

DB_PATH = os.path.join("vectorstore", "chat_history.db")
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection (same pattern as conversation_store)."""
    if not hasattr(_local, "conn"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_benchmark_db() -> None:
    """Create the benchmark_runs table if it doesn't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id              TEXT PRIMARY KEY,
            run_group_id    TEXT NOT NULL,
            user_id         TEXT NOT NULL,
            model           TEXT NOT NULL,
            question        TEXT NOT NULL,
            answer          TEXT NOT NULL,
            latency_seconds REAL NOT NULL,
            token_count     INTEGER NOT NULL,
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_benchmark_user_id
            ON benchmark_runs(user_id);

        CREATE INDEX IF NOT EXISTS idx_benchmark_run_group
            ON benchmark_runs(run_group_id);
    """)
    conn.commit()


# Initialize on import
init_benchmark_db()


# ─── Ollama Model Discovery ──────────────────────────────────────────────────

def list_ollama_models() -> list[str]:
    """
    Query Ollama's local API for all pulled models.

    Calls GET /api/tags on the Ollama server. Returns a list of model
    name strings (e.g. ["llama3.1:8b", "phi3:mini", "mistral:7b"]).

    Returns an empty list if Ollama is not reachable.
    """
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            print(f"[BENCHMARK] Found {len(models)} Ollama models: {models}")
            return models
        return []
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
        print(f"[BENCHMARK] Cannot reach Ollama for model list: {e}")
        return []


# ─── System Prompt ────────────────────────────────────────────────────────────

BENCHMARK_SYSTEM_PROMPT = """\
You are a precise document Q&A assistant.
Use ONLY the provided context to answer. Be concise and factual.
If the answer is not in the context, say "I don't know — not in the documents."
"""


# ─── Benchmark Runner ────────────────────────────────────────────────────────

def run_benchmark(
    user_id: str,
    models: list[str],
    questions: list[str],
) -> dict:
    """
    Run a benchmark: test each model on each question using the same retrieved context.

    For each question:
      1. Retrieve + rerank documents ONCE (fair comparison — same context for all models)
      2. For each model: call Ollama with that model, record latency + answer

    Args:
        user_id:   Authenticated user's ID (for retrieval from their documents)
        models:    List of Ollama model names to benchmark
        questions: List of questions to test

    Returns:
        Dict with 'run_group_id', 'results' (individual), and 'summary' (averages).
    """
    from src.llm_provider import OllamaProvider

    run_group_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = _get_conn()

    all_results = []
    model_stats: dict[str, list[float]] = {m: [] for m in models}

    print(f"\n[BENCHMARK] ═══ Starting benchmark run ═══")
    print(f"[BENCHMARK] Run ID: {run_group_id[:8]}...")
    print(f"[BENCHMARK] Models: {models}")
    print(f"[BENCHMARK] Questions: {len(questions)}")
    print()

    for q_idx, question in enumerate(questions, 1):
        print(f"[BENCHMARK] ─── Question {q_idx}/{len(questions)}: \"{question[:60]}\" ───")

        # Retrieve documents ONCE per question
        try:
            docs = retrieve_and_rerank(question, user_id)
            context_parts = []
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", 0) + 1
                context_parts.append(f"[Source {i}: {src}, Page {page}]\n{doc.page_content}")
            context = "\n\n".join(context_parts)
        except FileNotFoundError:
            print(f"[BENCHMARK] ⚠ No documents found — skipping question")
            continue

        if not context:
            print(f"[BENCHMARK] ⚠ Empty context — skipping question")
            continue

        messages = [
            {"role": "system", "content": BENCHMARK_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context (from your uploaded documents):\n"
                f"{'─'*40}\n{context}\n{'─'*40}\n\n"
                f"Question: {question}"
            )},
        ]

        # Test each model on the same question + context
        for model in models:
            print(f"[BENCHMARK]   Model: {model} ... ", end="", flush=True)

            try:
                provider = OllamaProvider()
                provider._model = model  # Override the model for this run

                t0 = time.time()
                answer = provider.chat(messages, stream=False)
                latency = time.time() - t0

                token_count = len(answer.split())  # approximate

                result = {
                    "id": str(uuid.uuid4()),
                    "run_group_id": run_group_id,
                    "user_id": user_id,
                    "model": model,
                    "question": question,
                    "answer": answer,
                    "latency_seconds": round(latency, 2),
                    "token_count": token_count,
                    "created_at": now,
                }
                all_results.append(result)
                model_stats[model].append(latency)

                # Persist to SQLite
                conn.execute(
                    """INSERT INTO benchmark_runs
                       (id, run_group_id, user_id, model, question, answer,
                        latency_seconds, token_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (result["id"], run_group_id, user_id, model, question,
                     answer, latency, token_count, now),
                )
                conn.commit()

                print(f"{latency:.1f}s, ~{token_count} tokens")

            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "id": str(uuid.uuid4()),
                    "run_group_id": run_group_id,
                    "user_id": user_id,
                    "model": model,
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "latency_seconds": 0,
                    "token_count": 0,
                    "created_at": now,
                })

    # Build summary
    summary = {}
    for model, latencies in model_stats.items():
        if latencies:
            summary[model] = {
                "avg_latency": round(sum(latencies) / len(latencies), 2),
                "min_latency": round(min(latencies), 2),
                "max_latency": round(max(latencies), 2),
                "total_questions": len(latencies),
            }

    print(f"\n[BENCHMARK] ═══ Benchmark complete ═══")
    print(f"[BENCHMARK] Summary:")
    for model, stats in summary.items():
        print(f"[BENCHMARK]   {model}: avg {stats['avg_latency']}s "
              f"(min {stats['min_latency']}s, max {stats['max_latency']}s)")
    print()

    return {
        "run_group_id": run_group_id,
        "results": all_results,
        "summary": summary,
        "created_at": now,
    }


# ─── Query Results ────────────────────────────────────────────────────────────

def get_benchmark_results(user_id: str) -> list[dict]:
    """
    Get all past benchmark runs for this user, grouped by run_group_id.

    Returns a list of run groups, each containing:
      - run_group_id, created_at
      - models tested
      - summary (avg latency per model)
      - individual results
    """
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM benchmark_runs
           WHERE user_id = ?
           ORDER BY created_at DESC""",
        (user_id,),
    ).fetchall()

    if not rows:
        return []

    # Group by run_group_id
    groups: dict[str, list[dict]] = {}
    for row in rows:
        r = dict(row)
        gid = r["run_group_id"]
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(r)

    result = []
    for gid, results in groups.items():
        # Build per-model summary
        model_latencies: dict[str, list[float]] = {}
        for r in results:
            model_latencies.setdefault(r["model"], []).append(r["latency_seconds"])

        summary = {}
        for model, lats in model_latencies.items():
            summary[model] = {
                "avg_latency": round(sum(lats) / len(lats), 2),
                "total_questions": len(lats),
            }

        result.append({
            "run_group_id": gid,
            "created_at": results[0]["created_at"],
            "models": list(model_latencies.keys()),
            "summary": summary,
            "results": results,
        })

    return result
