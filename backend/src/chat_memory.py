"""
src/chat_memory.py -- In-Memory Sliding-Window Cache for LLM Context
---------------------------------------------------------------------
WHAT THIS FILE DOES:
  Maintains a fast in-memory cache of recent conversation messages,
  used ONLY for building the LLM context window (the "history" sent
  with each request so the model can handle follow-up questions).

  This is NOT the source of truth for conversation data. That role
  belongs to conversation_store.py (SQLite). This cache exists purely
  for performance: reading 4 recent messages from a Python dict is
  faster than hitting SQLite on every single chat request.

CACHE MISS BEHAVIOR:
  If a conversation_id is not in the cache (e.g., after a server
  restart), get_history() automatically rebuilds the cache by loading
  recent messages from SQLite via conversation_store.get_recent_messages().
  This means restarts are transparent to users.

SLIDING WINDOW (unchanged from Phase 2):
  LLMs have a limited context window. We keep only the last
  MAX_EXCHANGES pairs (~600-800 tokens) to leave room for
  retrieved chunks and the answer.

CONNECTIONS:
  -> Called by api.py on every POST /stream and POST /chat request
  -> Reads from conversation_store.py on cache miss
  -> conversation_store.py is the durable source of truth
"""

import time
from typing import Optional


# ─── Configuration ────────────────────────────────────────────────────────────

# How many user/assistant exchanges to remember in the LLM context
# 1 exchange = 1 user message + 1 assistant message
# 4 exchanges = 8 messages = ~600-800 tokens of context overhead
MAX_EXCHANGES = int(4)

# How many seconds of inactivity before a cached session is eligible for cleanup
# 7200 seconds = 2 hours
SESSION_TIMEOUT_SECONDS = 7200

# How many total cached sessions to allow before forcing cleanup
# Prevents memory growth if the server runs for weeks
MAX_SESSIONS = 500


# ─── Session Cache ────────────────────────────────────────────────────────────
#
# Structure (identical to Phase 2, but now a cache rather than source of truth):
# {
#   "conversation-uuid-1": {
#     "messages": [
#       {"role": "user",      "content": "What is BERT?"},
#       {"role": "assistant", "content": "BERT is a transformer model..."},
#     ],
#     "last_active": 1705329811.23,
#     "created_at":  1705329600.00
#   },
#   ...
# }
#
_sessions: dict = {}


# ─── Public API ───────────────────────────────────────────────────────────────

def get_history(conversation_id: str) -> list[dict]:
    """
    Return the recent chat history for a conversation as a messages list.

    CACHE HIT:  return from in-memory cache instantly
    CACHE MISS: load recent messages from SQLite, populate cache, return

    FORMAT (standard OpenAI/Groq messages format):
      [
        {"role": "user",      "content": "What is attention?"},
        {"role": "assistant", "content": "Attention is a mechanism..."},
      ]

    Returns an empty list if the conversation has no messages yet.
    Always applies the sliding window (last MAX_EXCHANGES * 2 messages).
    """
    if conversation_id not in _sessions:
        # Cache miss — rebuild from SQLite
        _rebuild_from_sqlite(conversation_id)

    if conversation_id not in _sessions:
        # Still not found (new conversation with no messages yet)
        return []

    # Update last_active timestamp
    _sessions[conversation_id]["last_active"] = time.time()

    messages = _sessions[conversation_id]["messages"]

    # Apply sliding window: keep only the last N exchanges
    max_messages = MAX_EXCHANGES * 2
    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    return messages


def add_exchange(conversation_id: str, user_message: str, assistant_message: str) -> None:
    """
    Add a completed exchange to the in-memory cache.

    NOTE: This only updates the in-memory cache for fast subsequent reads.
    The durable SQLite write happens separately in api.py via
    conversation_store.add_message(). This separation keeps the cache
    logic simple and avoids double-writing.
    """
    if conversation_id not in _sessions:
        _init_session(conversation_id)

    session = _sessions[conversation_id]

    session["messages"].append({
        "role": "user",
        "content": user_message.strip(),
    })
    session["messages"].append({
        "role": "assistant",
        "content": assistant_message.strip(),
    })

    session["last_active"] = time.time()

    # Trigger cleanup if we're approaching the session cap
    if len(_sessions) > MAX_SESSIONS * 0.9:
        _cleanup_old_sessions()

    print(f"[MEMORY] Cache for '{conversation_id[:8]}...' now has "
          f"{len(session['messages']) // 2} exchange(s)")


def invalidate_session(conversation_id: str) -> None:
    """
    Remove a conversation from the in-memory cache.

    Called when a conversation is deleted via the API. The next
    get_history() call would try to rebuild from SQLite, find nothing,
    and return an empty list.
    """
    if conversation_id in _sessions:
        del _sessions[conversation_id]
        print(f"[MEMORY] Cache invalidated for '{conversation_id[:8]}...'")


def clear_session(session_id: str) -> bool:
    """
    Clear all history for a session (but keep it in cache).

    Kept for backward compatibility with the /sessions/{id} endpoint.
    Returns True if the session existed and was cleared.
    """
    if session_id not in _sessions:
        return False

    _sessions[session_id]["messages"] = []
    _sessions[session_id]["last_active"] = time.time()
    print(f"[MEMORY] Session '{session_id[:8]}...' cleared.")
    return True


def get_session_summary(session_id: str) -> dict:
    """
    Return metadata about a cached session.

    Kept for backward compatibility with the /sessions/{id} endpoint.
    """
    if session_id not in _sessions:
        return {
            "session_id": session_id,
            "exists": False,
            "exchange_count": 0,
            "messages": [],
        }

    messages = _sessions[session_id]["messages"]
    return {
        "session_id": session_id,
        "exists": True,
        "exchange_count": len(messages) // 2,
        "messages": messages,
        "last_active": _sessions[session_id]["last_active"],
    }


def get_active_session_count() -> int:
    """Return the number of active cached sessions (for monitoring/status endpoint)."""
    return len(_sessions)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _init_session(conversation_id: str) -> None:
    """Create a new empty cache entry."""
    now = time.time()
    _sessions[conversation_id] = {
        "messages": [],
        "created_at": now,
        "last_active": now,
    }


def _rebuild_from_sqlite(conversation_id: str) -> None:
    """
    Load recent messages from SQLite into the in-memory cache.

    Called on cache miss (e.g., first request after a server restart).
    Imports conversation_store lazily to avoid circular imports.
    """
    try:
        from src.conversation_store import get_recent_messages

        recent = get_recent_messages(conversation_id, limit=MAX_EXCHANGES * 2)
        if recent:
            _init_session(conversation_id)
            _sessions[conversation_id]["messages"] = recent
            print(f"[MEMORY] Rebuilt cache for '{conversation_id[:8]}...' "
                  f"from SQLite ({len(recent)} messages)")
    except Exception as e:
        # If SQLite read fails, proceed with empty history
        # (degraded but functional — the LLM just won't have context)
        print(f"[MEMORY] Warning: could not rebuild cache from SQLite: {e}")


def _cleanup_old_sessions() -> int:
    """
    Remove cached sessions that have been inactive beyond SESSION_TIMEOUT_SECONDS.

    Prevents memory growth on long-running servers.
    Called automatically when approaching MAX_SESSIONS.

    Returns the number of sessions removed.
    """
    now = time.time()
    cutoff = now - SESSION_TIMEOUT_SECONDS

    expired = [
        sid for sid, data in _sessions.items()
        if data["last_active"] < cutoff
    ]

    for sid in expired:
        del _sessions[sid]

    if expired:
        print(f"[MEMORY] Cleaned up {len(expired)} expired cached session(s). "
              f"Active: {len(_sessions)}")

    return len(expired)
