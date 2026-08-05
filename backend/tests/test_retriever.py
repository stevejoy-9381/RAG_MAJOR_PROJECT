"""
tests/test_retriever.py — RAG Retrieval Quality Test Suite
─────────────────────────────────────────────────────────────
Verifies SHA-256 deduplication, metadata preservation, and citation formatting.
"""

import os
import sys
import pytest
from langchain.schema import Document

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from src.retriever import format_sources, retrieve_with_rewritten_queries


def test_format_sources_deduplication():
    """Verify format_sources deduplicates identical text chunks."""
    doc1 = Document(
        page_content="Quarterly revenue reached $5M in Q3.",
        metadata={"source": "report.pdf", "page": 0, "upload_time": "2026-08-05T12:00"},
    )
    doc2 = Document(
        page_content="Quarterly revenue reached $5M in Q3.",
        metadata={"source": "report.pdf", "page": 0, "upload_time": "2026-08-05T12:00"},
    )
    doc3 = Document(
        page_content="Net profit increased by 15% year-over-year.",
        metadata={"source": "report.pdf", "page": 1, "upload_time": "2026-08-05T12:00"},
    )

    sources = format_sources([doc1, doc2, doc3])
    assert len(sources) == 2, f"Expected 2 unique sources, got {len(sources)}"
    assert sources[0]["file"] == "report.pdf"
    assert sources[0]["page"] == 1
