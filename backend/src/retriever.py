"""
src/retriever.py — Per-User Hybrid Retrieval + Re-ranking + Multi-Query
─────────────────────────────────────────────────────────────────────────
PIPELINE EVOLUTION:

  Phase 3: EnsembleRetriever returned RETRIEVAL_K docs → used directly.
  Phase 4: Retrieve RETRIEVAL_CANDIDATES → CrossEncoder rerank → FINAL_CONTEXT_K.
  Phase 5: Optional query rewriting → fan-out retrieval for EACH query →
           merge + deduplicate → CrossEncoder rerank → FINAL_CONTEXT_K.

  The Phase 5 multi-query path is triggered when retrieve_and_rerank()
  receives a `queries` list with >1 entry (produced by query_rewriter.py).
  When queries is None or a single item, behavior is identical to Phase 4.

  ESCAPE HATCHES:
    RERANKING_ENABLED=false  → bypass CrossEncoder (Phase 3 behavior)
    QUERY_REWRITING_ENABLED  → controlled in api.py (this module just
                                accepts whatever queries it's given)
"""

from __future__ import annotations
import os
import time

# Prevent transformers from trying to import TensorFlow/Keras at import time.
# We only use PyTorch-based models (CrossEncoder, HuggingFaceEmbeddings).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA

from src.llm import get_llm, get_prompt_template
from src.ingest import get_user_index_path


# ─── Configuration ────────────────────────────────────────────────────────────

from src.llm import get_llm, get_prompt_template
from src.ingest import get_user_index_path
from src.config import (
    EMBEDDING_MODEL, RERANKER_MODEL,
    RETRIEVAL_K, RETRIEVAL_CANDIDATES, FINAL_CONTEXT_K,
    BM25_WEIGHT, FAISS_WEIGHT, RERANKING_ENABLED,
)


# ─── Cache ────────────────────────────────────────────────────────────────────
# Embedding model: shared across all users (one model, identical weights)
# Retriever: per-user dict — each user's key stores their EnsembleRetriever
# CrossEncoder: shared across all users (one model, stateless scoring)
_embedding_model_cache: HuggingFaceEmbeddings | None = None
_cross_encoder_cache = None  # CrossEncoder instance (lazy-imported to avoid TF/Keras)
_retriever_cache: dict[str, EnsembleRetriever] = {}   # {user_id: retriever}


def invalidate_user_cache(user_id: str) -> None:
    """
    Remove this user's cached retriever.
    Called after they upload a new document or delete one.
    The next request rebuilds the retriever from the updated index.
    """
    if user_id in _retriever_cache:
        del _retriever_cache[user_id]
        print(f"[RETRIEVER] Cache invalidated for user '{user_id[:8]}'")


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Load embedding model once, reuse for all users."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        print(f"[RETRIEVER] Loading embedding model...")
        _embedding_model_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[RETRIEVER] Embedding model loaded and cached.")
    return _embedding_model_cache


def _get_cross_encoder() -> CrossEncoder:
    """
    Load the CrossEncoder reranker model once, reuse for all users.

    Uses the same caching pattern as _get_embedding_model().
    The model is ~80MB and runs inference on CPU — fast enough for
    re-scoring 10-20 short document passages per query (~50-100ms).
    """
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        # Lazy import: avoids triggering transformers → TensorFlow → Keras
        # import chain at module load time. Only needed when reranking is
        # actually used (first call to rerank()).
        from sentence_transformers import CrossEncoder
        print(f"[RETRIEVER] Loading CrossEncoder reranker ({RERANKER_MODEL})...")
        _cross_encoder_cache = CrossEncoder(RERANKER_MODEL)
        print("[RETRIEVER] CrossEncoder loaded and cached.")
    return _cross_encoder_cache


def _check_user_index(user_id: str) -> None:
    """Raise a clear error if this user has no FAISS index yet."""
    index_path = get_user_index_path(user_id)
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        raise FileNotFoundError(
            "You haven't uploaded any documents yet. "
            "Upload a PDF first to start asking questions."
        )


def get_hybrid_retriever(user_id: str) -> EnsembleRetriever:
    """
    Return the hybrid BM25 + FAISS retriever for a specific user.

    CACHE HIT:   return cached retriever for this user instantly
    CACHE MISS:  build retriever from this user's FAISS index, cache it

    When RERANKING_ENABLED=true, the retriever fetches RETRIEVAL_CANDIDATES
    docs (a larger pool). The caller is responsible for passing these through
    rerank() to narrow down to FINAL_CONTEXT_K.

    When RERANKING_ENABLED=false, the retriever fetches RETRIEVAL_K docs
    directly (the pre-Phase 4 behavior).
    """
    _check_user_index(user_id)

    if user_id in _retriever_cache:
        return _retriever_cache[user_id]

    # Decide how many candidates to retrieve
    k = RETRIEVAL_CANDIDATES if RERANKING_ENABLED else RETRIEVAL_K

    print(f"[RETRIEVER] Building retriever for user '{user_id[:8]}' (k={k}, reranking={'ON' if RERANKING_ENABLED else 'OFF'})...")
    embedding_model = _get_embedding_model()
    index_path = get_user_index_path(user_id)

    vectorstore = FAISS.load_local(
        index_path, embedding_model,
        allow_dangerous_deserialization=True,
    )

    all_docs = list(vectorstore.docstore._dict.values())
    if not all_docs:
        raise ValueError("Your document index is empty. Re-upload your documents.")

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k

    faiss_ret = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_ret],
        weights=[BM25_WEIGHT, FAISS_WEIGHT],
    )

    _retriever_cache[user_id] = ensemble
    print(f"[RETRIEVER] Retriever cached for user '{user_id[:8]}'.")
    return ensemble


# ─── Re-ranking ───────────────────────────────────────────────────────────────

def rerank(query: str, docs: list) -> list[tuple]:
    """
    Re-rank retrieved documents using a CrossEncoder model.

    HOW IT WORKS:
      The CrossEncoder takes (query, document_text) pairs and produces a
      relevance score for each pair. Unlike bi-encoder embeddings (which
      encode query and document independently), the CrossEncoder reads them
      TOGETHER through a transformer — much more accurate but slower.

      This is why we use it as a second stage: retrieve many cheaply with
      BM25+FAISS, then re-score the top candidates accurately with the
      CrossEncoder.

    Args:
        query: The user's question
        docs: List of LangChain Document objects from the retriever

    Returns:
        List of (doc, score) tuples, sorted by score descending,
        truncated to FINAL_CONTEXT_K. The score is also injected into
        each doc's metadata as 'relevance_score' for downstream use.
    """
    if not docs:
        return []

    cross_encoder = _get_cross_encoder()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc in docs]

    t0 = time.time()
    scores = cross_encoder.predict(pairs)
    rerank_ms = (time.time() - t0) * 1000

    # Combine docs with scores and sort by score descending
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    # Log the reranking results for comparison
    print(f"\n[RERANKER] -- Re-ranking results ({rerank_ms:.0f}ms) --")
    print(f"[RERANKER] Query: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")
    print(f"[RERANKER] {'Rank':<5} {'Score':>8}  {'Source':<30} {'Page':>5}")
    print(f"[RERANKER] {'-'*5} {'-'*8}  {'-'*30} {'-'*5}")
    for i, (doc, score) in enumerate(scored, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0) + 1
        marker = " [OK]" if i <= FINAL_CONTEXT_K else " [X]"
        print(f"[RERANKER] {i:<5} {float(score):>8.4f}  {src:<30} {page:>5}{marker}")
    print(f"[RERANKER] Keeping top {FINAL_CONTEXT_K} of {len(scored)} candidates ({rerank_ms:.0f}ms latency)")
    print()

    # Truncate to FINAL_CONTEXT_K and inject score into metadata
    result = scored[:FINAL_CONTEXT_K]
    for doc, score in result:
        doc.metadata["relevance_score"] = round(float(score), 4)

    return result


def retrieve_with_rewritten_queries(queries: list[str], user_id: str) -> list:
    """
    Fan-out retrieval: run the ensemble retriever for EACH query in the list,
    then merge and deduplicate all results.

    This is the Phase 5 multi-query path. When query rewriting produces
    ["original question", "rewrite 1", "rewrite 2"], we retrieve candidates
    for each, giving the reranker a wider and more diverse candidate pool.

    Deduplication uses the first 200 chars of page_content as a key
    (same approach as format_sources()'s seen set).

    Args:
        queries: List of query strings (original + rewrites)
        user_id: The authenticated user's ID

    Returns:
        Deduplicated list of LangChain Document objects from all queries.
    """
    retriever = get_hybrid_retriever(user_id)
    all_docs = []
    seen = set()

    for i, q in enumerate(queries):
        docs = retriever.invoke(q)
        new_count = 0
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
                new_count += 1
        print(f"[RETRIEVER] Query {i+1}/{len(queries)}: \"{q[:60]}{'...' if len(q) > 60 else ''}\" -> {len(docs)} docs ({new_count} new)")

    print(f"[RETRIEVER] Multi-query total: {len(all_docs)} unique candidates from {len(queries)} queries")
    return all_docs


def retrieve_and_rerank(query: str, user_id: str, queries: list[str] | None = None) -> list:
    """
    Full retrieval pipeline: retrieve candidates -> rerank -> return.

    Phase 5 addition: when `queries` is provided with >1 entry (from
    query rewriting), fans out retrieval to all queries before reranking.
    When queries is None or single, behaves identically to Phase 4.

    Args:
        query:   The original user question (used for reranking scoring)
        user_id: The authenticated user's ID
        queries: Optional list of queries (original + rewrites). If None
                 or single, uses single-query retrieval.

    Returns:
        List of LangChain Document objects (top FINAL_CONTEXT_K after reranking,
        or top RETRIEVAL_K if reranking is disabled).
    """
    # Decide retrieval strategy: multi-query or single-query
    if queries and len(queries) > 1:
        docs = retrieve_with_rewritten_queries(queries, user_id)
    else:
        retriever = get_hybrid_retriever(user_id)
        docs = retriever.invoke(query)

    if not RERANKING_ENABLED:
        print(f"[RETRIEVER] Reranking disabled -- returning {len(docs)} docs directly")
        return docs

    # Log pre-rerank order for comparison
    print(f"\n[RETRIEVER] -- Pre-rerank order ({len(docs)} candidates) --")
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0) + 1
        print(f"[RETRIEVER]   {i}. {src} (Page {page})")

    # Re-rank using the ORIGINAL query (best for cross-encoder scoring)
    ranked = rerank(query, docs)
    return [doc for doc, _score in ranked]


# ─── Legacy functions (kept for backward compatibility) ───────────────────────

def build_qa_chain(user_id: str) -> RetrievalQA:
    """Build the full QA chain for this user's index (for /ask endpoint)."""
    retriever = get_hybrid_retriever(user_id)
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt_template()},
    )
    return qa_chain


def format_sources(docs: list) -> list[dict]:
    """
    Format retrieved documents into clean source citation dicts.

    If the documents have been through reranking, each will have a
    'relevance_score' in their metadata. This is included in the output
    so the frontend can display it.
    """
    sources = []
    seen = set()
    for doc in docs:
        key = doc.page_content[:100]
        if key in seen:
            continue
        seen.add(key)
        meta = doc.metadata
        source_dict = {
            "file":        meta.get("source", "unknown"),
            "page":        meta.get("page", 0) + 1,
            "total_pages": meta.get("total_pages", "?"),
            "chunk_index": meta.get("chunk_index", "?"),
            "upload_time": meta.get("upload_time", ""),
            "preview": (
                doc.page_content[:300] + "..."
                if len(doc.page_content) > 300
                else doc.page_content
            ),
        }
        # Include relevance score if available (from reranking)
        if "relevance_score" in meta:
            source_dict["relevance_score"] = meta["relevance_score"]
        sources.append(source_dict)
    return sources
