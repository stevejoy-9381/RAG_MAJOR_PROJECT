"""
test_benchmark.py — Verify the model benchmarking pipeline
─────────────────────────────────────────────────────────────
Run from the backend directory:
  python test_benchmark.py

Prerequisites:
  - At least one PDF uploaded for a user
  - Ollama running locally with at least 1-2 models pulled
    e.g. `ollama pull llama3.1:8b` and `ollama pull phi3:mini`

What it does:
  1. Finds a user with an existing FAISS index
  2. Queries available Ollama models via list_ollama_models()
  3. Executes a benchmark across available models with test questions
  4. Queries stored benchmark results from SQLite
  5. Displays a summary comparison table
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.benchmark import list_ollama_models, run_benchmark, get_benchmark_results

TEST_QUESTIONS = [
    "What are the main findings or conclusions?",
    "Summarize the key methodology or steps.",
]


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
    print(f"\n{'='*70}")
    print(f"  MODEL BENCHMARKING PIPELINE TEST")
    print(f"{'='*70}\n")

    user_id = find_user_with_index()

    print("─── STEP 1: Discover Ollama Models ───")
    models = list_ollama_models()
    if not models:
        print("⚠ No Ollama models found via API. Testing with fallback default model list.")
        models = ["llama3.1:8b"]
    else:
        print(f"✅ Available models: {models}")

    # Use up to 2 models for quick test
    test_models = models[:2]
    print(f"\nSelected models for test: {test_models}")
    print(f"Selected questions ({len(TEST_QUESTIONS)}): {TEST_QUESTIONS}")

    print("\n─── STEP 2: Execute Benchmark ───")
    t0 = time.time()
    res = run_benchmark(user_id, test_models, TEST_QUESTIONS)
    total_time = time.time() - t0

    print(f"\n✅ Benchmark completed in {total_time:.1f}s")
    print(f"Run Group ID: {res['run_group_id']}")

    print("\n─── STEP 3: Per-Model Latency & Token Summary ───")
    print(f"{'Model':<20} {'Avg Latency':<15} {'Total Qs':<10}")
    print(f"{'─'*20} {'─'*15} {'─'*10}")
    for model, stats in res["summary"].items():
        print(f"{model:<20} {stats['avg_latency']:>6.2f}s        {stats['total_questions']:<10}")

    print("\n─── STEP 4: Query Past Results from SQLite ───")
    past_runs = get_benchmark_results(user_id)
    print(f"✅ Retrieved {len(past_runs)} historical benchmark run group(s) from SQLite")

    if past_runs:
        latest = past_runs[0]
        print(f"Latest run ({latest['run_group_id'][:8]}...): {len(latest['results'])} responses stored.")

    print(f"\n{'='*70}")
    print("  BENCHMARK TEST COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_test()
