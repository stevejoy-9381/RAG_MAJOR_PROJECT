"""
api.py — FastAPI Backend (Phase 5: Query Rewriting)
────────────────────────────────────────────────────
WHAT CHANGED FROM PHASE 4:

  NEW ENDPOINTS:
    POST /auth/register     → create a new user account
    POST /auth/login        → authenticate, receive JWT token
    GET  /auth/me           → verify token, return user profile

  ALL DOCUMENT + CHAT ENDPOINTS NOW PROTECTED:
    Every endpoint (except /health, /auth/*) now requires:
      Authorization: Bearer <token>
    in the request header.

    FastAPI's Depends(get_current_user) handles this automatically:
      1. Extracts "Bearer <token>" from the Authorization header
      2. Decodes and verifies the JWT
      3. Returns the user dict to the endpoint function
      4. Returns 401 Unauthorized if the token is missing/invalid/expired

  PER-USER STATE:
    Phase 2: _qa_chain was a global singleton (one chain for all users)
    Phase 3: No global chain. Each endpoint call gets the user_id from
             the JWT and builds/caches the chain for that user.
             _retriever_cache in retriever.py handles caching per user_id.

HOW get_current_user DEPENDENCY WORKS:
  FastAPI's dependency injection system:

  @app.post("/upload")
  async def upload(
      file: UploadFile,
      user: dict = Depends(get_current_user)   ← injected automatically
  ):
      user_id = user["user_id"]   ← from the verified JWT
      ...

  If the token is missing → FastAPI returns 401 before the function runs.
  If the token is valid → FastAPI calls get_current_user(), gets the user dict,
  passes it to the endpoint function as the "user" parameter.

  This is clean, reusable, and impossible to forget — unlike manually
  parsing the Authorization header in every endpoint.
"""

import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import json
import time

# Ensure UTF-8 output encoding on Windows consoles to prevent charmap encoding errors
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import asyncio
import tempfile
import threading
import queue
from pathlib import Path
import traceback
from typing import Optional, Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from dotenv import load_dotenv

from src.auth import create_access_token, decode_access_token
from src.user_store import (
    register_user, authenticate_user, get_user_by_id, get_user_count,
)
from src.ingest import run_ingestion, ALLOWED_EXTENSIONS
from src.retriever import (
    get_hybrid_retriever, build_qa_chain,
    format_sources, invalidate_user_cache,
    retrieve_and_rerank,
)
from src.query_rewriter import rewrite_query
from src.document_store import (
    get_all_documents, get_document_stats,
    is_duplicate, remove_document,
)
from src.chat_memory import (
    get_history, add_exchange, invalidate_session,
    clear_session, get_session_summary, get_active_session_count,
)
from src.llm_provider import get_provider, check_provider_availability
from src.conversation_store import (
    create_conversation, list_conversations, get_conversation,
    add_message, delete_conversation, rename_conversation,
    generate_title, conversation_belongs_to_user,
)
from src.benchmark import (
    list_ollama_models, run_benchmark, get_benchmark_results,
)
from src.analytics import get_analytics_summary

load_dotenv()

# ─── Feature flags config ─────────────────────────────────────────────────────────
from src.config import QUERY_REWRITING_ENABLED, SHOW_THINKING_PROCESS


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document Q&A API — v4",
    description="Authenticated multi-user RAG with per-user document isolation.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    from src.config import GROQ_MODEL, GROQ_API_KEY
    if GROQ_API_KEY or os.getenv("GROQ_API_KEY"):
        print("Groq Provider Connected")
        print(f"Model: {GROQ_MODEL}")



# ─── Auth dependency ──────────────────────────────────────────────────────────

# OAuth2PasswordBearer reads the token from:
#   Authorization: Bearer <token>
# tokenUrl is shown in /docs — the URL clients should POST to for a token.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency: validate JWT and return user dict.

    HOW FASTAPI DEPENDENCY INJECTION WORKS:
      When you add  user: dict = Depends(get_current_user)  to an endpoint,
      FastAPI automatically:
        1. Extracts the token from the Authorization header
        2. Calls this function with that token
        3. Passes the returned dict to the endpoint as "user"
        4. If this function raises HTTPException → the endpoint never runs

    WHY Depends() INSTEAD OF MANUAL TOKEN PARSING?
      - DRY: write auth logic once, use everywhere
      - Testable: swap get_current_user in tests easily
      - Self-documenting: /docs shows which endpoints require auth
      - Cannot be forgotten: forgetting Depends() = unprotected endpoint
        (visible in code review), not a subtle bug

    RAISES:
      401 Unauthorized if token is missing, invalid, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("sub")
    if not user_id:
        raise credentials_exception

    # Verify user still exists (handles deleted accounts)
    user = get_user_by_id(user_id)
    if user is None:
        raise credentials_exception

    return user


# ─── Request / Response models ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str = ""

    class Config:
        json_schema_extra = {
            "example": {"username": "alice", "password": "securepass123"}
        }


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    user_id: str
    message: str


class QuestionRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # Deprecated — use conversation_id
    llm_mode: Optional[Literal["auto", "online", "offline"]] = "auto"
    use_query_rewriting: Optional[bool] = None  # Per-request override (None = use env default)
    show_thinking: Optional[bool] = None  # Per-request override for streaming reasoning tokens


class AnswerResponse(BaseModel):
    answer: str
    sources: list


class ChatResponse(BaseModel):
    answer: str
    sources: list
    conversation_id: str
    provider: str
    reasoning: Optional[str] = None



class RenameRequest(BaseModel):
    title: str


class UploadResponse(BaseModel):
    status: str
    file: str
    pages: int
    chunks: int
    was_duplicate: bool
    message: str
    total_documents: int


class DocumentInfo(BaseModel):
    filename: str
    uploaded_at: str
    pages: int
    chunks: int
    size_kb: float


class DocumentLibraryResponse(BaseModel):
    total_documents: int
    total_pages: int
    total_chunks: int
    documents: list[DocumentInfo]


class BenchmarkRequest(BaseModel):
    models: list[str]
    questions: list[str]



# ─── LLM helpers ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a precise document Q&A assistant.

Your task is to answer questions using ONLY the information provided in the user's context.

Rules:
- Use ONLY the given context
- Do NOT use external knowledge
- Do NOT guess or assume missing information
- If the answer is not found, say:
  "I don't know — this isn't covered in the uploaded documents."
- Answer in the same language the question was asked in, when possible, even if the source context is in a different language.

Response style:
- Be concise and factual
- Use bullet points when helpful
- Keep explanations short and clear

Grounding:
- Base your answer strictly on relevant parts of the context
- If multiple parts are relevant, combine them logically

Do not:
- Add information not present in the context
- Fabricate explanations or details
"""


def _build_messages(question: str, context: str, history: list[dict]) -> list[dict]:
    """Build history-aware messages list for Groq API."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": (
            f"Context (from your uploaded documents):\n"
            f"{'─'*40}\n{context}\n{'─'*40}\n\n"
            f"Question: {question}"
        ),
    })
    return messages


# ─── Auth endpoints (PUBLIC — no Depends(get_current_user)) ──────────────────

@app.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(request: RegisterRequest):
    """
    Register a new user and return a JWT token immediately.

    WHY LOG IN AUTOMATICALLY AFTER REGISTER?
      Better UX — user doesn't have to fill the login form again.
      Standard practice in modern web apps.

    WHAT HAPPENS:
      1. Validate username/password format
      2. Check username is not taken
      3. Hash password with bcrypt and save
      4. Create JWT token
      5. Return token (client stores it, includes in future requests)
    """
    try:
        user = register_user(
            username=request.username,
            password=request.password,
            email=request.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    token = create_access_token({
        "sub": user["user_id"],
        "username": user["username"],
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=user["username"],
        user_id=user["user_id"],
        message=f"Welcome, {user['username']}! Your account has been created.",
    )


@app.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest):
    """
    Authenticate a user and return a JWT token.

    WHAT HAPPENS:
      1. Look up username in registry
      2. Verify password against bcrypt hash
      3. If valid → create and return JWT token
      4. If invalid → 401 Unauthorized (same error for wrong user/password)
    """
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({
        "sub": user["user_id"],
        "username": user["username"],
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=user["username"],
        user_id=user["user_id"],
        message=f"Welcome back, {user['username']}!",
    )


@app.get("/auth/me")
def get_me(user: dict = Depends(get_current_user)):
    """
    Return the authenticated user's profile.
    The frontend calls this on load to verify the stored token is still valid.
    If the token has expired, this returns 401 and the frontend shows the login page.
    """
    return {
        "user_id":   user["user_id"],
        "username":  user["username"],
        "email":     user.get("email", ""),
        "created_at": user.get("created_at", ""),
    }


# ─── Health (public) ──────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "4.0.0",
        "registered_users": get_user_count(),
        "active_sessions": get_active_session_count(),
    }


# ─── Status (protected) ───────────────────────────────────────────────────────

@app.get("/status")
def get_status(user: dict = Depends(get_current_user)):
    """Return this user's document stats, readiness, and LLM provider availability."""
    user_id = user["user_id"]
    stats = get_document_stats(user_id)
    index_path = f"vectorstore/{user_id}/faiss_index/index.faiss"
    return {
        "ready":           os.path.exists(index_path),
        "username":        user["username"],
        "active_sessions": get_active_session_count(),
        "llm_provider_available": check_provider_availability(),
        **stats,
    }


# ─── Document library (protected) ────────────────────────────────────────────

@app.get("/documents", response_model=DocumentLibraryResponse)
def list_documents(user: dict = Depends(get_current_user)):
    """Return this user's document library."""
    stats = get_document_stats(user["user_id"])
    return DocumentLibraryResponse(
        total_documents=stats["total_documents"],
        total_pages=stats["total_pages"],
        total_chunks=stats["total_chunks"],
        documents=[DocumentInfo(**d) for d in stats["documents"]],
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Upload and index a document for the authenticated user.

    Supported formats: .pdf, .docx, .pptx, .txt, .csv
    The file extension is checked against ALLOWED_EXTENSIONS from ingest.py.
    The temp file preserves the original extension so the format dispatcher
    in load_document() can detect the type.
    """
    user_id = user["user_id"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    was_dup = is_duplicate(user_id, file.filename)
    print(f"[API] user='{user['username']}' upload='{file.filename}'")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = run_ingestion(
            file_path=tmp_path,
            user_id=user_id,
            original_filename=file.filename,
        )
        # Invalidate this user's cached retriever so next request rebuilds
        invalidate_user_cache(user_id)

        all_docs = get_all_documents(user_id)
        return UploadResponse(
            status="success",
            file=file.filename,
            pages=result["pages"],
            chunks=result["chunks"],
            was_duplicate=was_dup,
            message=(
                f"✓ {'Updated' if was_dup else 'Indexed'} '{file.filename}'. "
                f"{result['pages']} pages → {result['chunks']} chunks. "
                f"{len(all_docs)} document(s) in your library."
            ),
            total_documents=len(all_docs),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.delete("/documents/{filename}")
def delete_document(
    filename: str,
    user: dict = Depends(get_current_user),
):
    user_id = user["user_id"]
    removed = remove_document(user_id, filename)
    if not removed:
        raise HTTPException(
            status_code=404, detail=f"'{filename}' not found in your library."
        )
    invalidate_user_cache(user_id)
    return {
        "status": "removed",
        "filename": filename,
        "remaining": len(get_all_documents(user_id)),
    }


# ─── Conversation CRUD endpoints (protected) ─────────────────────────────────

@app.get("/conversations")
def api_list_conversations(user: dict = Depends(get_current_user)):
    """List all conversations for the authenticated user, newest first."""
    return list_conversations(user["user_id"])


@app.post("/conversations", status_code=201)
def api_create_conversation(user: dict = Depends(get_current_user)):
    """Create a new empty conversation."""
    conv_id = create_conversation(user["user_id"])
    return {"id": conv_id, "status": "created"}


@app.get("/conversations/{conversation_id}")
def api_get_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
):
    """Get a full conversation with all messages. 404 if not found or not owned."""
    conv = get_conversation(conversation_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@app.delete("/conversations/{conversation_id}")
def api_delete_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a conversation and all its messages."""
    deleted = delete_conversation(conversation_id, user["user_id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    invalidate_session(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}


@app.patch("/conversations/{conversation_id}")
def api_rename_conversation(
    conversation_id: str,
    body: RenameRequest,
    user: dict = Depends(get_current_user),
):
    """Rename a conversation's title."""
    if not body.title or not body.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    renamed = rename_conversation(conversation_id, user["user_id"], body.title)
    if not renamed:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "renamed", "conversation_id": conversation_id, "title": body.title.strip()}


# ─── Streaming endpoint (protected) ──────────────────────────────────────────

def _resolve_conversation_id(request: ChatRequest, user_id: str, question: str) -> tuple[str, bool]:
    """
    Resolve the conversation_id from the request.

    Returns (conversation_id, is_new) where is_new is True if the
    conversation was auto-created from this request.

    Supports backward compat: session_id is accepted as an alias
    for conversation_id if the latter is not provided.
    """
    conv_id = request.conversation_id or request.session_id

    if conv_id:
        # Verify the conversation exists and belongs to this user
        if not conversation_belongs_to_user(conv_id, user_id):
            # If it doesn't exist in SQLite (e.g., old session_id from
            # pre-migration), create a new conversation with that ID
            # so the client's reference still works.
            conv_id = create_conversation(user_id, title=generate_title(question))
            return conv_id, True
        return conv_id, False

    # No conversation_id provided — auto-create
    title = generate_title(question)
    conv_id = create_conversation(user_id, title=title)
    return conv_id, True


@app.post("/stream")
async def stream_answer(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """
    Stream answer tokens for the authenticated user.

    Uses get_provider() to select Ollama or Groq based on request.llm_mode.
    Persists messages to SQLite via conversation_store.
    The final metadata SSE event includes which provider answered
    and the conversation_id (which may be auto-created).
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    user_id = user["user_id"]
    question = request.question.strip()

    # Resolve conversation
    conversation_id, _ = _resolve_conversation_id(request, user_id, question)

    # Resolve provider early so errors surface before streaming starts
    try:
        provider = get_provider(request.llm_mode)
    except (ConnectionError, EnvironmentError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    provider_name = provider.name

    async def generate():
        full_answer = ""
        docs = []

        try:
            loop = asyncio.get_event_loop()

            # Phase 5: optional query rewriting before retrieval
            queries = None
            should_rewrite = request.use_query_rewriting if request.use_query_rewriting is not None else QUERY_REWRITING_ENABLED
            if should_rewrite:
                queries = await loop.run_in_executor(None, rewrite_query, question, provider)

            docs = await loop.run_in_executor(
                None, lambda: retrieve_and_rerank(question, user_id, queries=queries)
            )

            context_parts = []
            for i, doc in enumerate(docs, 1):
                src  = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", 0) + 1
                context_parts.append(f"[Source {i}: {src}, Page {page}]\n{doc.page_content}")
            context = "\n\n".join(context_parts)

            if not context:
                yield f"data: {json.dumps({'type':'token','content':'I could not find relevant sections in your documents for this question.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            history  = get_history(conversation_id)
            messages = _build_messages(question, context, history)

            # ── Stream tokens via the selected provider ───────────────────
            token_queue: queue.Queue = queue.Queue()
            error_box: list = []
            t_start = time.time()
            yield_reasoning = request.show_thinking if request.show_thinking is not None else SHOW_THINKING_PROCESS

            def producer():
                try:
                    for item in provider.chat(messages, stream=True, yield_reasoning=yield_reasoning):
                        token_queue.put(item)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"STREAM ERROR ({provider_name}):", repr(e))
                    error_box.append(str(e))
                finally:
                    token_queue.put(None)

            threading.Thread(target=producer, daemon=True).start()

            while True:
                try:
                    item = token_queue.get(timeout=60)
                except queue.Empty:
                    yield f"data: {json.dumps({'type':'error','content':'Stream timed out.'})}\n\n"
                    break
                if item is None:
                    break
                if error_box:
                    yield f"data: {json.dumps({'type':'error','content':error_box[0]})}\n\n"
                    break

                if yield_reasoning and isinstance(item, tuple):
                    event_type, text = item
                    if event_type == "reasoning":
                        yield f"data: {json.dumps({'type':'reasoning','content':text})}\n\n"
                    else:
                        full_answer += text
                        yield f"data: {json.dumps({'type':'token','content':text})}\n\n"
                else:
                    token_str = item if isinstance(item, str) else item[1]
                    full_answer += token_str
                    yield f"data: {json.dumps({'type':'token','content':token_str})}\n\n"
                await asyncio.sleep(0)


        except FileNotFoundError as e:
            yield f"data: {json.dumps({'type':'error','content':str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','content':f'Error: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if full_answer:
            latency = round(time.time() - t_start, 2)
            # Persist to SQLite (source of truth)
            add_message(conversation_id, "user", question)
            sources = format_sources(docs)
            add_message(conversation_id, "assistant", full_answer,
                        sources=sources, provider=provider_name, latency_seconds=latency)
            # Update in-memory cache for subsequent requests
            add_exchange(conversation_id, question, full_answer)
        else:
            sources = format_sources(docs)

        yield f"data: {json.dumps({'type':'metadata','sources':sources,'conversation_id':conversation_id,'provider':provider_name})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Non-streaming chat (protected) ──────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """
    Non-streaming chat -- uses get_provider() for Ollama/Groq routing.
    Persists messages to SQLite. Returns conversation_id and provider name.
    """
    user_id = user["user_id"]
    question = request.question.strip()

    # Resolve conversation
    conversation_id, _ = _resolve_conversation_id(request, user_id, question)

    # Resolve provider
    try:
        provider = get_provider(request.llm_mode)
    except (ConnectionError, EnvironmentError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        # Phase 5: optional query rewriting before retrieval
        queries = None
        should_rewrite = request.use_query_rewriting if request.use_query_rewriting is not None else QUERY_REWRITING_ENABLED
        if should_rewrite:
            queries = rewrite_query(question, provider)

        docs = retrieve_and_rerank(question, user_id, queries=queries)

        context_parts = []
        for i, doc in enumerate(docs, 1):
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0) + 1
            context_parts.append(f"[Source {i}: {src}, Page {page}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        history  = get_history(conversation_id)
        messages = _build_messages(question, context, history)

        yield_reasoning = request.show_thinking if request.show_thinking is not None else SHOW_THINKING_PROCESS
        t_start  = time.time()
        chat_result = provider.chat(messages, stream=False, yield_reasoning=yield_reasoning)
        if yield_reasoning and isinstance(chat_result, tuple):
            answer, reasoning = chat_result
        else:
            answer, reasoning = (chat_result, None) if isinstance(chat_result, str) else (chat_result[0], chat_result[1])
        latency  = round(time.time() - t_start, 2)

        # Persist to SQLite
        sources = format_sources(docs)
        add_message(conversation_id, "user", question)
        add_message(conversation_id, "assistant", answer,
                    sources=sources, provider=provider.name, latency_seconds=latency)
        # Update in-memory cache
        add_exchange(conversation_id, question, answer)

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            provider=provider.name,
            reasoning=reasoning,
        )

    except (ConnectionError, EnvironmentError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Legacy session endpoints (deprecated — use /conversations) ──────────────

@app.get("/sessions/{session_id}")
def get_session(session_id: str, user: dict = Depends(get_current_user)):
    """Deprecated: use GET /conversations/{id} instead."""
    return get_session_summary(session_id)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Deprecated: use DELETE /conversations/{id} instead."""
    cleared = clear_session(session_id)
    return {"status": "cleared" if cleared else "not_found", "session_id": session_id}


# ─── Benchmark endpoints (protected) ─────────────────────────────────────────

@app.get("/benchmark/models")
def get_benchmark_models(user: dict = Depends(get_current_user)):
    """Return list of pulled Ollama models available on local machine."""
    return {"models": list_ollama_models()}


@app.post("/benchmark/run")
def api_run_benchmark(
    request: BenchmarkRequest,
    user: dict = Depends(get_current_user),
):
    """
    Run benchmark comparing multiple local models on the same set of questions.
    Uses the user's isolated document index for retrieval context.
    """
    if not request.models:
        raise HTTPException(status_code=400, detail="At least one model must be specified.")
    if not request.questions or not any(q.strip() for q in request.questions):
        raise HTTPException(status_code=400, detail="At least one question must be provided.")

    clean_questions = [q.strip() for q in request.questions if q.strip()]

    try:
        return run_benchmark(
            user_id=user["user_id"],
            models=request.models,
            questions=clean_questions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")


@app.get("/benchmark/results")
def api_get_benchmark_results(user: dict = Depends(get_current_user)):
    """Get all past benchmark runs for the authenticated user."""
    return get_benchmark_results(user["user_id"])


# ─── Analytics endpoint (protected) ──────────────────────────────────────────

@app.get("/analytics/summary")
def api_get_analytics_summary(user: dict = Depends(get_current_user)):
    """Return usage analytics summary for the authenticated user."""
    return get_analytics_summary(user["user_id"])




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
