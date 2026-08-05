"""
src/retriever.py — Per-User Hybrid Retrieval + Re-ranking + Multi-Query
─────────────────────────────────────────────────────────────────────────
Combines dense FAISS vector search and sparse BM25 keyword matching,
uses SHA-256 candidate deduplication, CrossEncoder re-ranking, and
integrates structured logging with process-wide embedding singletons.
"""

from __future__ import annotations
import os
import time
import hashlib
from typing import List, Dict, Tuple, Optional, Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.schema import Document

from src.llm import get_llm, get_prompt_template
from src.ingest import get_user_index_path, get_embedding_model
from src.config import (
    EMBEDDING_MODEL, RERANKER_MODEL,
    RETRIEVAL_K, RETRIEVAL_CANDIDATES, FINAL_CONTEXT_K,
    BM25_WEIGHT, FAISS_WEIGHT, RERANKING_ENABLED,
)
from src.logger import get_logger, log_event, Timer

logger = get_logger("RETRIEVER")

_cross_encoder_cache = None
_retriever_cache: Dict[str, EnsembleRetriever] = {}


def invalidate_user_cache(user_id: str) -> None:
    """Remove user's cached retriever after document modifications."""
    if user_id in _retriever_cache:
        del _retriever_cache[user_id]
        logger.info(f"Cache invalidated for user '{user_id[:8]}'")


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Delegate to process-wide singleton embedding model in ingest.py."""
    return get_embedding_model()


def _get_cross_encoder():
    """Load CrossEncoder reranker model once and cache."""
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading CrossEncoder reranker model ({RERANKER_MODEL})...")
        _cross_encoder_cache = CrossEncoder(RERANKER_MODEL)
        logger.info("CrossEncoder reranker model ready.")
    return _cross_encoder_cache


def _check_user_index(user_id: str) -> None:
    """Ensure user's FAISS index exists before retrieving."""
    index_path = get_user_index_path(user_id)
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        raise FileNotFoundError(
            "You haven't uploaded any documents yet. "
            "Upload a document first to ask questions."
        )


def get_hybrid_retriever(user_id: str) -> EnsembleRetriever:
    """Return cached or newly built BM25 + FAISS EnsembleRetriever for user."""
    _check_user_index(user_id)

    if user_id in _retriever_cache:
        return _retriever_cache[user_id]

    k = RETRIEVAL_CANDIDATES if RERANKING_ENABLED else RETRIEVAL_K
    logger.info(f"Building hybrid retriever for user '{user_id[:8]}' (k={k}, reranking={'ON' if RERANKING_ENABLED else 'OFF'})")

    embedding_model = _get_embedding_model()
    index_path = get_user_index_path(user_id)

    vectorstore = FAISS.load_local(
        index_path, embedding_model,
        allow_dangerous_deserialization=True,
    )

    all_docs = list(vectorstore.docstore._dict.values())
    if not all_docs:
        raise ValueError("Document index is empty. Please re-upload your documents.")

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
    logger.info(f"Hybrid retriever built and cached for user '{user_id[:8]}'")
    return ensemble


def rerank(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """Re-rank candidate documents using CrossEncoder transformer."""
    if not docs:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]

    t0 = time.time()
    scores = cross_encoder.predict(pairs)
    rerank_ms = (time.time() - t0) * 1000

    scored = list(zip(docs, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"CrossEncoder reranked {len(docs)} candidates in {rerank_ms:.1f}ms for query: '{query[:60]}'")

    result = scored[:FINAL_CONTEXT_K]
    for doc, score in result:
        doc.metadata["relevance_score"] = round(score, 4)

    return result


def retrieve_with_rewritten_queries(queries: List[str], user_id: str) -> List[Document]:
    """Fan-out multi-query retrieval with SHA-256 deduplication."""
    retriever = get_hybrid_retriever(user_id)
    all_docs: List[Document] = []
    seen_hashes = set()

    for i, q in enumerate(queries):
        docs = retriever.invoke(q)
        new_count = 0
        for doc in docs:
            doc_hash = hashlib.sha256(doc.page_content.strip().encode("utf-8")).hexdigest()
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                all_docs.append(doc)
                new_count += 1
        logger.info(f"Query {i+1}/{len(queries)}: '{q[:50]}' -> {len(docs)} docs ({new_count} unique)")

    logger.info(f"Multi-query total: {len(all_docs)} unique candidates from {len(queries)} queries")
    return all_docs


def retrieve_and_rerank(query: str, user_id: str, queries: Optional[List[str]] = None) -> List[Document]:
    """Full RAG retrieval pipeline: fetch candidates -> deduplicate -> rerank -> return top K."""
    t0 = time.time()
    if queries and len(queries) > 1:
        docs = retrieve_with_rewritten_queries(queries, user_id)
    else:
        retriever = get_hybrid_retriever(user_id)
        docs = retriever.invoke(query)
        # Deduplicate single-query docs
        seen = set()
        deduped = []
        for d in docs:
            h = hashlib.sha256(d.page_content.strip().encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                deduped.append(d)
        docs = deduped

    if not RERANKING_ENABLED:
        logger.info(f"Reranking disabled -- returning top {len(docs)} docs directly")
        return docs[:FINAL_CONTEXT_K]

    ranked = rerank(query, docs)
    elapsed = round((time.time() - t0) * 1000, 1)
    logger.info(f"Retrieval & reranking completed in {elapsed}ms: returning {len(ranked)} docs")
    return [doc for doc, _score in ranked]


def build_qa_chain(user_id: str) -> RetrievalQA:
    """Build legacy RetrievalQA chain for user index."""
    retriever = get_hybrid_retriever(user_id)
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt_template()},
    )


def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Format retrieved document chunks into clean source citation dictionaries."""
    sources = []
    seen = set()
    for doc in docs:
        key = hashlib.sha256(doc.page_content.strip().encode("utf-8")).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        meta = doc.metadata
        source_dict = {
            "file": meta.get("source", "unknown"),
            "page": meta.get("page", 0) + 1,
            "total_pages": meta.get("total_pages", "?"),
            "chunk_index": meta.get("chunk_index", "?"),
            "upload_time": meta.get("upload_time", ""),
            "preview": (
                doc.page_content[:300] + "..."
                if len(doc.page_content) > 300
                else doc.page_content
            ),
        }
        if "relevance_score" in meta:
            source_dict["relevance_score"] = meta["relevance_score"]
        sources.append(source_dict)
    return sources
