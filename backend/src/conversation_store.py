"""
src/conversation_store.py -- Persistent SQLite Conversation Storage
--------------------------------------------------------------------
WHAT THIS FILE DOES:
  Manages durable conversation storage in SQLite. Every chat message
  (user and assistant) is persisted to vectorstore/chat_history.db so
  conversations survive server restarts and can be listed/reopened.

WHY SQLITE (NOT THE IN-MEMORY DICT)?
  Phase 2's chat_memory.py used an in-memory Python dict:
    - Lost on every server restart
    - No way to list or reopen past conversations
    - No persistence for users who log back in later

  SQLite gives us:
    - Zero-setup persistence (single file, no server process)
    - ACID transactions (no partial writes)
    - Per-user isolation enforced at the query level
    - Full conversation history for the frontend sidebar

SCHEMA:
  conversations: one row per conversation (id, user_id, title, timestamps)
  messages: one row per message (id, conversation_id, role, content, etc.)

CONNECTIONS:
  -> Called by api.py for CRUD endpoints and message persistence
  -> chat_memory.py calls get_recent_messages() to rebuild its
     in-memory cache on a cache miss (e.g., after restart)
"""

import os
import json
import uuid
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────

DB_PATH = os.path.join("vectorstore", "chat_history.db")

# Thread-local storage for SQLite connections.
# SQLite connections cannot be shared across threads, so each thread
# gets its own connection via threading.local().
_local = threading.local()


# ─── Database Setup ───────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """
    Get a thread-local SQLite connection.

    WHY THREAD-LOCAL?
      SQLite's default mode disallows sharing a single connection across
      threads (raises ProgrammingError). FastAPI uses a thread pool for
      sync endpoints, so each thread needs its own connection.

      threading.local() gives each thread its own namespace. The first
      call from a new thread creates and caches a connection; subsequent
      calls from the same thread reuse it.
    """
    if not hasattr(_local, "conn"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row  # dict-like access
        _local.conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db() -> None:
    """
    Create the conversations and messages tables if they don't exist.

    Called once at module import time. CREATE TABLE IF NOT EXISTS is
    idempotent, so this is safe to call multiple times.
    """
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            title       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_user_id
            ON conversations(user_id);

        CREATE TABLE IF NOT EXISTS messages (
            id                TEXT PRIMARY KEY,
            conversation_id   TEXT NOT NULL,
            role              TEXT NOT NULL,
            content           TEXT NOT NULL,
            sources_json      TEXT,
            llm_provider_used TEXT,
            latency_seconds   REAL,
            created_at        TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
            ON messages(conversation_id);
    """)

    # Migration check for existing databases missing latency_seconds column
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(messages)")
    columns = [col[1] for col in cursor.fetchall()]
    if "latency_seconds" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN latency_seconds REAL")

    conn.commit()
    print("[CONVERSATION_STORE] SQLite database initialized at", DB_PATH)


# ─── Title Generation ────────────────────────────────────────────────────────

def generate_title(first_message: str) -> str:
    """
    Generate a short conversation title from the first user message.

    Takes the first 6-8 words (up to ~50 characters) and adds ellipsis
    if truncated. No LLM call needed — simple string slicing.

    Examples:
      "What are the main findings of the research paper?"
        -> "What are the main findings of the..."
      "Hi"
        -> "Hi"
      "Explain the architecture of BERT and how it differs from GPT"
        -> "Explain the architecture of BERT and how..."
    """
    words = first_message.strip().split()
    if not words:
        return "New Conversation"

    # Take up to 8 words
    title_words = words[:8]
    title = " ".join(title_words)

    # Trim to ~50 characters if still too long
    if len(title) > 50:
        title = title[:47].rstrip() + "..."
    elif len(words) > 8:
        # Had more words — indicate truncation
        title = title.rstrip(".,;:!?") + "..."

    return title


# ─── CRUD Operations ─────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def create_conversation(user_id: str, title: Optional[str] = None) -> str:
    """
    Create a new conversation for a user.

    Args:
        user_id: The authenticated user's ID (from JWT)
        title: Optional title. If None, defaults to "New Conversation"
               (typically overwritten when the first message arrives)

    Returns:
        The new conversation's UUID string
    """
    conv_id = str(uuid.uuid4())
    now = _now_iso()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conv_id, user_id, title or "New Conversation", now, now),
    )
    conn.commit()
    print(f"[CONVERSATION_STORE] Created conversation '{conv_id[:8]}...' for user '{user_id[:8]}...'")
    return conv_id


def list_conversations(user_id: str) -> list[dict]:
    """
    List all conversations for a user, newest first.

    Returns a list of dicts with: id, title, updated_at, message_count,
    last_message_preview (first 100 chars of the most recent message).

    WHY ORDER BY updated_at DESC?
      The most recently active conversation should appear first in the
      sidebar, matching the UX of ChatGPT, Claude, etc.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT
            c.id,
            c.title,
            c.created_at,
            c.updated_at,
            COUNT(m.id) AS message_count,
            (
                SELECT content FROM messages
                WHERE conversation_id = c.id
                ORDER BY created_at DESC
                LIMIT 1
            ) AS last_message
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id = ?
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        """,
        (user_id,),
    ).fetchall()

    results = []
    for row in rows:
        last_msg = row["last_message"] or ""
        preview = (last_msg[:100] + "...") if len(last_msg) > 100 else last_msg
        results.append({
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"],
            "last_message_preview": preview,
        })

    return results


def get_conversation(conversation_id: str, user_id: str) -> Optional[dict]:
    """
    Get a full conversation with all its messages.

    ISOLATION: Returns None if the conversation doesn't exist OR doesn't
    belong to this user_id. This prevents user A from reading user B's
    conversations by guessing UUIDs.

    Returns:
        {
            "id": "...",
            "title": "...",
            "created_at": "...",
            "updated_at": "...",
            "messages": [
                {"id": "...", "role": "user", "content": "...", ...},
                {"id": "...", "role": "assistant", "content": "...", ...},
            ]
        }
        or None if not found / not owned.
    """
    conn = _get_conn()

    # Verify ownership
    conv = conn.execute(
        "SELECT id, user_id, title, created_at, updated_at "
        "FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    ).fetchone()

    if not conv:
        return None

    # Fetch all messages in chronological order
    messages = conn.execute(
        "SELECT id, role, content, sources_json, llm_provider_used, latency_seconds, created_at "
        "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()

    return {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "messages": [
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "sources": json.loads(m["sources_json"]) if m["sources_json"] else None,
                "llm_provider_used": m["llm_provider_used"],
                "latency_seconds": m["latency_seconds"] if "latency_seconds" in m.keys() else None,
                "created_at": m["created_at"],
            }
            for m in messages
        ],
    }


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
    provider: Optional[str] = None,
    latency_seconds: Optional[float] = None,
) -> str:
    """
    Add a message to an existing conversation.

    Called twice per exchange:
      1. add_message(conv_id, "user", question)
      2. add_message(conv_id, "assistant", answer, sources=..., provider=..., latency_seconds=...)

    Also updates the conversation's updated_at timestamp so it floats
    to the top of the list.
    """
    msg_id = str(uuid.uuid4())
    now = _now_iso()
    sources_json = json.dumps(sources) if sources else None

    conn = _get_conn()
    conn.execute(
        "INSERT INTO messages "
        "(id, conversation_id, role, content, sources_json, llm_provider_used, latency_seconds, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, sources_json, provider, latency_seconds, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    return msg_id


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """
    Delete a conversation and all its messages.

    ISOLATION: Only deletes if the conversation belongs to this user_id.
    ON DELETE CASCADE in the schema handles message cleanup automatically.

    Returns True if a conversation was actually deleted, False if not found/not owned.
    """
    conn = _get_conn()

    # Check ownership first
    row = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    ).fetchone()

    if not row:
        return False

    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    print(f"[CONVERSATION_STORE] Deleted conversation '{conversation_id[:8]}...'")
    return True


def rename_conversation(conversation_id: str, user_id: str, new_title: str) -> bool:
    """
    Rename a conversation's title.

    ISOLATION: Only renames if the conversation belongs to this user_id.
    Returns True if renamed, False if not found/not owned.
    """
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE conversations SET title = ?, updated_at = ? "
        "WHERE id = ? AND user_id = ?",
        (new_title.strip(), _now_iso(), conversation_id, user_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_recent_messages(conversation_id: str, limit: int = 8) -> list[dict]:
    """
    Get the most recent messages from a conversation for cache rebuilding.

    Called by chat_memory.py when it has a cache miss (e.g., after server
    restart). Returns messages in chronological order (oldest first within
    the window).

    Args:
        conversation_id: UUID of the conversation
        limit: Max number of messages to return (default 8 = 4 exchanges)

    Returns:
        List of {"role": "...", "content": "..."} dicts
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content FROM messages "
        "WHERE conversation_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (conversation_id, limit),
    ).fetchall()

    # Reverse to chronological order (the query returns newest-first)
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def conversation_belongs_to_user(conversation_id: str, user_id: str) -> bool:
    """
    Check if a conversation belongs to a specific user.
    Used for quick ownership verification without fetching all messages.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    ).fetchone()
    return row is not None


# ─── Initialize on import ────────────────────────────────────────────────────
init_db()
