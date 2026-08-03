"""
src/config.py — Centralized Configuration & Environment Settings (Phase 11)
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Serves as the single source of truth for shared system configuration,
  environment variables, default model paths, and tuning parameters.

WHY CENTRALIZED CONFIG?
  Previously, EMBEDDING_MODEL was defined independently in ingest.py and
  retriever.py. If one changed without the other, ingestion and retrieval
  would use different vector models with mismatched vector dimensions (e.g.
  384 vs 768), causing deserialization errors or garbage retrieval results.

MULTILINGUAL SUPPORT:
  - Default: "sentence-transformers/all-MiniLM-L6-v2" (384-dim, English optimized)
  - Multilingual option: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (384-dim, 50+ languages)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Embedding Model ─────────────────────────────────────────────────────────
# Single source of truth for both ingestion and retrieval.
# Override via EMBEDDING_MODEL env var in .env file.
# Note: Changing this requires re-ingesting all uploaded documents!
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# ─── Re-ranking Model ─────────────────────────────────────────────────────────
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# ─── Text Chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# ─── Retrieval & Re-ranking Parameters ────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 4))
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", 10))
FINAL_CONTEXT_K = int(os.getenv("FINAL_CONTEXT_K", 4))

BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.4))
FAISS_WEIGHT = float(os.getenv("FAISS_WEIGHT", 0.6))

# ─── Groq LLM Configuration ───────────────────────────────────────────────────
# Single source of truth for Groq cloud LLM provider configuration.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", os.getenv("LLM_MODEL", "qwen/qwen3.6-27b"))
MODEL_NAME = GROQ_MODEL  # Centralized alias

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", os.getenv("MAX_TOKENS", "1024")))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))

# ─── Feature Flags ────────────────────────────────────────────────────────────
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() in ("true", "1", "yes")
QUERY_REWRITING_ENABLED = os.getenv("QUERY_REWRITING_ENABLED", "false").lower() in ("true", "1", "yes")
SHOW_THINKING_PROCESS = os.getenv("SHOW_THINKING_PROCESS", "false").lower() in ("true", "1", "yes")

