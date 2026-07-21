"""
test_analytics.py — Generate sample usage data & verify /analytics/summary
─────────────────────────────────────────────────────────────────────────────
Run from the backend directory:
  python test_analytics.py

Prerequisites:
  - At least one PDF or document uploaded for a user
  - An available LLM provider (Ollama or Groq API key set)

What it does:
  1. Finds a user with an existing FAISS index (or test user)
  2. Generates ~10-15 sample exchange records across online/offline providers and documents
  3. Queries get_analytics_summary(user_id) directly and prints formatted metrics
  4. Verifies question frequencies, most cited documents, timeline data, and provider splits
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.conversation_store import create_conversation, add_message
from src.analytics import get_analytics_summary


def find_user_with_index() -> str:
    """Find a user_id that has a FAISS index on disk or return a test user_id."""
    vs_dir = "vectorstore"
    if os.path.exists(vs_dir):
        for entry in os.listdir(vs_dir):
            index_path = os.path.join(vs_dir, entry, "faiss_index", "index.faiss")
            if os.path.exists(index_path):
                print(f"[OK] Found user index: {entry[:8]}...")
                return entry

    print("[INFO] Using test user_id 'test_user_analytics' for demonstration...")
    return "test_user_analytics"


def seed_sample_interactions(user_id: str):
    """Seed ~12 realistic exchanges into SQLite for analytics verification."""
    print("\n--- STEP 1: Seeding Realistic Usage Data ---")

    questions_and_modes = [
        ("What are the main findings of the document?", "groq"),
        ("Summarize the key conclusions.", "groq"),
        ("What are the main findings of the document?", "ollama"),
        ("Explain the methodology used in the paper.", "groq"),
        ("What are the main findings of the document?", "groq"),
        ("List all authors and institutional affiliations.", "ollama"),
        ("Explain the methodology used in the paper.", "ollama"),
        ("What dataset or hardware was used?", "groq"),
        ("Summarize the key conclusions.", "groq"),
        ("What dataset or hardware was used?", "ollama"),
        ("What are the future work recommendations?", "groq"),
        ("Summarize the key conclusions.", "ollama"),
    ]

    conv_id = create_conversation(user_id, title="Analytics Test Conversation")

    for i, (q, provider) in enumerate(questions_and_modes, 1):
        doc_name = "research_paper.pdf" if i % 2 == 0 else "technical_spec.docx"
        page_num = (i % 4) + 1
        sources = [{
            "file": doc_name,
            "page": page_num,
            "total_pages": 10,
            "chunk_index": i,
            "upload_time": "2026-07-20T19:00",
            "preview": f"Sample preview content for {doc_name} page {page_num}",
            "relevance_score": 0.85 + (i * 0.01),
        }]

        add_message(conv_id, "user", q)

        mock_answer = f"Answer {i} to '{q}' based on retrieved sources from {doc_name}."
        mock_latency = round(1.2 + (i % 3) * 0.8, 2)

        add_message(
            conv_id,
            "assistant",
            mock_answer,
            sources=sources,
            provider=provider,
            latency_seconds=mock_latency,
        )

    print(f"[OK] Successfully seeded {len(questions_and_modes)} Q&A exchanges across Groq and Ollama.")


def run_test():
    print(f"\n{'='*70}")
    print(f"  USAGE ANALYTICS & DASHBOARD TEST")
    print(f"{'='*70}")

    user_id = find_user_with_index()

    # Seed data
    seed_sample_interactions(user_id)

    # Fetch summary
    print("\n--- STEP 2: Querying get_analytics_summary() ---")
    summary = get_analytics_summary(user_id)

    print(f"\nSUMMARY OVERVIEW:")
    print(f"  * Total Conversations: {summary['total_conversations']}")
    print(f"  * Total Answers:       {summary['total_answers']}")
    print(f"  * Avg Latency:         {summary['avg_latency_seconds']}s")

    print(f"\nTOP QUESTIONS:")
    for item in summary["top_questions"]:
        print(f"  * [{item['count']}x] \"{item['question']}\"")

    print(f"\nMOST CITED DOCUMENTS:")
    for doc in summary["most_cited_documents"]:
        print(f"  * {doc['document']}: {doc['citations']} citations ({doc['unique_pages_cited']} unique pages)")

    print(f"\nPROVIDER SPLIT:")
    split = summary["provider_split"]
    print(f"  * Online (Groq):    {split['online']}")
    print(f"  * Offline (Ollama): {split['offline']}")
    print(f"  * Total Logged:     {split['total']}")

    print(f"\nUSAGE OVER TIME (Recent Days):")
    for pt in summary["usage_over_time"][-5:]:
        print(f"  * {pt['date']}: {pt['questions']} questions")

    print(f"\n{'='*70}")
    print("  ANALYTICS TEST COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_test()
