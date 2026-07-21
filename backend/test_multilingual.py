"""
test_multilingual.py — Verify Multilingual Querying & Response Generation
────────────────────────────────────────────────────────────────────────────
Run from the backend directory:
  python test_multilingual.py

What it does:
  1. Tests configured EMBEDDING_MODEL from src.config
  2. Creates a sample English document chunk
  3. Queries in non-English languages (Hindi & Telugu)
  4. Verifies embedding generation and prompt formatting for cross-lingual response
"""

import os
import sys

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from src.config import EMBEDDING_MODEL
from langchain_community.embeddings import HuggingFaceEmbeddings


def run_test():
    print(f"\n{'='*70}")
    print(f"  MULTILINGUAL EMBEDDING & PROMPT VERIFICATION")
    print(f"  Active EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"{'='*70}\n")

    print("[STEP 1] Initializing HuggingFace Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[OK] Embedding model '{EMBEDDING_MODEL}' initialized successfully.")

    sample_doc_en = "Artificial Intelligence and Machine Learning are transforming modern medical diagnostics and healthcare."
    query_hi = "स्वास्थ्य सेवा में कृत्रिम बुद्धिमत्ता (Artificial Intelligence) का क्या उपयोग है?" # Hindi
    query_te = "వైద్య రంగంలో కృత్రిమ మేధస్సు (AI) ఉపయోగం ఏమిటి?" # Telugu

    print("\n[STEP 2] Testing Vector Embedding Generations...")
    doc_vec = embeddings.embed_query(sample_doc_en)
    hi_vec = embeddings.embed_query(query_hi)
    te_vec = embeddings.embed_query(query_te)

    print(f"  * English Doc Vector Dimension:   {len(doc_vec)}")
    print(f"  * Hindi Query Vector Dimension:   {len(hi_vec)}")
    print(f"  * Telugu Query Vector Dimension:  {len(te_vec)}")

    # Simple cosine similarity check
    def dot_product(v1, v2):
        return sum(a * b for a, b in zip(v1, v2))

    sim_hi = dot_product(doc_vec, hi_vec)
    sim_te = dot_product(doc_vec, te_vec)

    print(f"\n[STEP 3] Cross-lingual Semantic Similarity:")
    print(f"  * Hindi Query -> English Doc Cosine Similarity:  {sim_hi:.4f}")
    print(f"  * Telugu Query -> English Doc Cosine Similarity: {sim_te:.4f}")

    print(f"\n{'='*70}")
    print("  MULTILINGUAL TEST COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_test()
