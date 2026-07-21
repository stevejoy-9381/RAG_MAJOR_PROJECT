"""
src/analytics.py — Usage Analytics & Insights (Phase 9)
──────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Provides analytical queries over past user interactions, document citations,
  provider distribution, and usage timelines stored in SQLite.

FUNCTIONS:
  - get_top_questions(): Groups user questions by text to find most frequent queries.
  - get_most_cited_documents(): Aggregates sources_json citations across messages.
  - get_usage_over_time(): Groups questions/messages by day for the last N days.
  - get_provider_split(): Counts offline (Ollama) vs online (Groq) answers.
  - get_analytics_summary(): Returns all metrics in a single payload.

CONNECTIONS:
  → Called by api.py for GET /analytics/summary endpoint
  → Queries DB_PATH ("vectorstore/chat_history.db")
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timezone, timedelta

DB_PATH = os.path.join("vectorstore", "chat_history.db")
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def get_top_questions(user_id: str, limit: int = 10) -> list[dict]:
    """
    Find the most frequently asked questions for this user.

    Groups user messages by clean text (lowercase, stripped), returning
    the original question text and frequency count.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT m.content, COUNT(*) as frequency
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.user_id = ? AND m.role = 'user'
        GROUP BY LOWER(TRIM(m.content))
        ORDER BY frequency DESC, m.created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()

    return [
        {
            "question": row["content"],
            "count": row["frequency"],
        }
        for row in rows
    ]


def get_most_cited_documents(user_id: str, limit: int = 10) -> list[dict]:
    """
    Aggregate document citations from assistant response metadata.

    Reads sources_json from assistant messages owned by user_id,
    counting total citations and tracking unique pages per document.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT m.sources_json
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.user_id = ? AND m.role = 'assistant' AND m.sources_json IS NOT NULL
        """,
        (user_id,),
    ).fetchall()

    doc_stats: dict[str, dict] = {}  # {filename: {"citations": N, "pages": set()}}

    for row in rows:
        try:
            sources = json.loads(row["sources_json"]) if row["sources_json"] else []
            if not isinstance(sources, list):
                continue
            for src in sources:
                filename = src.get("file", "unknown")
                page = src.get("page", 1)

                if filename not in doc_stats:
                    doc_stats[filename] = {"citations": 0, "pages": set()}

                doc_stats[filename]["citations"] += 1
                doc_stats[filename]["pages"].add(page)
        except Exception:
            continue

    sorted_docs = sorted(
        doc_stats.items(),
        key=lambda x: x[1]["citations"],
        reverse=True,
    )[:limit]

    return [
        {
            "document": filename,
            "citations": stats["citations"],
            "unique_pages_cited": len(stats["pages"]),
        }
        for filename, stats in sorted_docs
    ]


def get_usage_over_time(user_id: str, days: int = 30) -> list[dict]:
    """
    Get daily message activity count for the last N days.

    Fills in missing dates with count=0 so charts display a continuous timeline.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT SUBSTR(m.created_at, 1, 10) as msg_date, COUNT(*) as msg_count
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.user_id = ? AND m.role = 'user'
        GROUP BY SUBSTR(m.created_at, 1, 10)
        ORDER BY msg_date ASC
        """,
        (user_id,),
    ).fetchall()

    activity_map = {row["msg_date"]: row["msg_count"] for row in rows}

    # Generate continuous date range
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days - 1)

    timeline = []
    curr = start_date
    while curr <= end_date:
        d_str = curr.strftime("%Y-%m-%d")
        timeline.append({
            "date": d_str,
            "questions": activity_map.get(d_str, 0),
        })
        curr += timedelta(days=1)

    return timeline


def get_provider_split(user_id: str) -> dict:
    """
    Count messages answered by offline (Ollama) vs online (Groq) LLM providers.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT m.llm_provider_used, COUNT(*) as cnt
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.user_id = ? AND m.role = 'assistant'
        GROUP BY m.llm_provider_used
        """,
        (user_id,),
    ).fetchall()

    split = {"ollama": 0, "groq": 0, "other": 0}
    total = 0

    for row in rows:
        provider = (row["llm_provider_used"] or "other").lower()
        cnt = row["cnt"]
        total += cnt
        if "ollama" in provider:
            split["ollama"] += cnt
        elif "groq" in provider:
            split["groq"] += cnt
        else:
            split["other"] += cnt

    return {
        "offline": split["ollama"],
        "online": split["groq"],
        "other": split["other"],
        "total": total,
    }


def get_analytics_summary(user_id: str) -> dict:
    """
    Get full analytics summary payload for GET /analytics/summary.
    """
    conn = _get_conn()

    # Total conversations count
    conv_count = conn.execute(
        "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
        (user_id,),
    ).fetchone()[0]

    # Total questions & avg latency
    stats = conn.execute(
        """
        SELECT
            COUNT(m.id) as total_messages,
            AVG(m.latency_seconds) as avg_latency
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.user_id = ? AND m.role = 'assistant'
        """,
        (user_id,),
    ).fetchone()

    total_answers = stats["total_messages"] or 0
    avg_latency = round(stats["avg_latency"], 2) if stats["avg_latency"] else 0.0

    return {
        "total_conversations": conv_count,
        "total_answers": total_answers,
        "avg_latency_seconds": avg_latency,
        "top_questions": get_top_questions(user_id, limit=10),
        "most_cited_documents": get_most_cited_documents(user_id, limit=10),
        "usage_over_time": get_usage_over_time(user_id, days=14),
        "provider_split": get_provider_split(user_id),
    }
