"""
src/llm_provider.py — Dual LLM Provider Abstraction (Ollama + Groq)
────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Provides a unified interface for calling either a local Ollama instance
  or the Groq cloud API. The factory function `get_provider()` handles
  automatic detection and fallback logic.

WHY A PROVIDER ABSTRACTION?
  api.py previously had Groq client calls hardcoded inline. This module
  decouples LLM selection from request handling so you can:
    - Run fully offline with Ollama (no API key needed)
    - Fall back to Groq when Ollama isn't available
    - Let users choose per-request via the `llm_mode` field

CONNECTIONS:
  → Used by api.py's /stream and /chat endpoints
  → Does NOT replace src/llm.py (which provides the LangChain ChatGroq
    wrapper used by retriever.py's build_qa_chain)

PROVIDER ROUTING (get_provider):
  preference="online"  → GroqProvider   (requires GROQ_API_KEY)
  preference="offline" → OllamaProvider (pings first, raises if down)
  preference="auto"/None → try Ollama ping → fallback to Groq → error
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Generator

import httpx
import groq
from groq import Groq
from dotenv import load_dotenv

from src.config import (
    GROQ_API_KEY, GROQ_MODEL, MODEL_NAME,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
)

load_dotenv()

_shared_groq_client: Groq | None = None


def _get_shared_groq_client() -> Groq:
    """Reuse a single Groq client instance across requests for performance."""
    global _shared_groq_client
    if _shared_groq_client is None:
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. "
                "Set it in your .env file or use Ollama (offline mode) instead."
            )
        _shared_groq_client = Groq(api_key=api_key)
        print("Groq Provider Connected")
        print(f"Model: {GROQ_MODEL}")
    return _shared_groq_client



# ─── Abstract Base ────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract base for LLM providers.

    Every provider must implement:
      - name: a short identifier ("ollama" or "groq")
      - chat(): accepts an OpenAI-style messages list, returns either
        a full response string or a generator of token strings.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. 'ollama', 'groq')."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        stream: bool = True,
    ) -> str | Generator[str, None, None]:
        """
        Send a chat completion request.

        Args:
            messages: OpenAI-style messages list
                      [{"role": "system", "content": "..."}, ...]
            stream:   If True, yield tokens one by one (generator).
                      If False, return the full answer as a string.

        Returns:
            str            if stream=False
            Generator[str] if stream=True (yields token strings)
        """
        ...


# ─── Ollama Provider ─────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    Local LLM provider via Ollama (http://localhost:11434).

    HOW IT WORKS:
      Ollama exposes an OpenAI-compatible-ish REST API. We POST to
      /api/chat with the messages list and read back NDJSON lines,
      each containing a token in {"message": {"content": "..."}}.

    CONFIGURATION:
      OLLAMA_HOST  → base URL (default http://localhost:11434)
      OLLAMA_MODEL → model name (default llama3.1:8b)

    STREAMING:
      Ollama streams by default. Each line of the response body is a
      JSON object. We yield the content of each as a token.
    """

    def __init__(self):
        self._host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self._model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    @property
    def name(self) -> str:
        return "ollama"

    def _ping(self, timeout: float = 3.0) -> bool:
        """
        Quick health check — GET the Ollama root endpoint.
        Returns True if Ollama responds, False otherwise.
        """
        try:
            resp = httpx.get(self._host, timeout=timeout)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    def chat(
        self,
        messages: list[dict],
        stream: bool = True,
    ) -> str | Generator[str, None, None]:
        """
        Call Ollama's /api/chat endpoint.

        STREAMING MODE (stream=True):
          Ollama returns NDJSON — one JSON object per line:
            {"message": {"role": "assistant", "content": "Hello"}, "done": false}
            {"message": {"role": "assistant", "content": " world"}, "done": false}
            {"message": {"role": "assistant", "content": ""}, "done": true}
          We yield each content string as a token.

        NON-STREAMING MODE (stream=False):
          We set stream=false in the request body. Ollama returns a single
          JSON object with the full response.
        """
        url = f"{self._host}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
                "num_predict": 1024,
            },
        }

        if stream:
            return self._stream_chat(url, payload)
        else:
            return self._sync_chat(url, payload)

    def _stream_chat(self, url: str, payload: dict) -> Generator[str, None, None]:
        """Stream tokens from Ollama via NDJSON response."""
        # Use a long timeout for generation (tokens can take a while on CPU)
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    def _sync_chat(self, url: str, payload: dict) -> str:
        """Non-streaming call — returns the full answer string."""
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")


# ─── Groq Provider ───────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    """
    Cloud LLM provider via Groq API.

    Uses centralized configuration (qwen/qwen3.6-27b on Groq LPU).

    CONFIGURATION:
      GROQ_API_KEY     → required
      GROQ_MODEL       → model name (default qwen/qwen3.6-27b)
      LLM_TEMPERATURE  → temperature (default 0.2)
    """

    def __init__(self):
        self._client = _get_shared_groq_client()
        self._model = GROQ_MODEL
        self._temperature = LLM_TEMPERATURE
        self._max_tokens = LLM_MAX_TOKENS
        self._top_p = LLM_TOP_P

    @property
    def name(self) -> str:
        return "groq"

    def chat(
        self,
        messages: list[dict],
        stream: bool = True,
        yield_reasoning: bool = False,
    ) -> str | tuple[str, str | None] | Generator[str | tuple[str, str], None, None]:
        """
        Call Groq's chat completions API with detailed error handling.

        STREAMING MODE:
          Uses Groq's streaming API — yields delta content tokens.
          If yield_reasoning=True, yields (type, content) tuples where type is 'reasoning' or 'token'.

        NON-STREAMING MODE:
          Single request, returns full answer (or (answer, reasoning) tuple if yield_reasoning=True).
        """
        if stream:
            return self._stream_chat(messages, yield_reasoning=yield_reasoning)
        else:
            return self._sync_chat(messages, yield_reasoning=yield_reasoning)

    def _stream_chat(
        self,
        messages: list[dict],
        yield_reasoning: bool = False,
    ) -> Generator[str | tuple[str, str], None, None]:
        """Stream tokens from Groq with exception handling and optional reasoning token yielding."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
                stream=True,
            )
            in_think = False
            buffer = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        buffer += delta.content
                        while buffer:
                            if not in_think:
                                if "<think>" in buffer:
                                    pre, post = buffer.split("<think>", 1)
                                    if pre:
                                        yield ("token", pre) if yield_reasoning else pre
                                    buffer = post
                                    in_think = True
                                else:
                                    yield ("token", buffer) if yield_reasoning else buffer
                                    buffer = ""
                                    break
                            else:
                                if "</think>" in buffer:
                                    think_part, post = buffer.split("</think>", 1)
                                    if think_part and yield_reasoning:
                                        yield ("reasoning", think_part)
                                    buffer = post.lstrip("\n")
                                    in_think = False
                                else:
                                    if yield_reasoning and buffer:
                                        yield ("reasoning", buffer)
                                        buffer = ""
                                    break
            if buffer:
                if in_think:
                    if yield_reasoning:
                        yield ("reasoning", buffer)
                else:
                    yield ("token", buffer) if yield_reasoning else buffer
        except groq.AuthenticationError as e:
            print(f"[GROQ ERROR] Invalid API key or unauthorized: {e}")
            raise RuntimeError("Groq API Authentication Error: Invalid API key provided.") from e
        except groq.RateLimitError as e:
            print(f"[GROQ ERROR] Rate limit exceeded: {e}")
            raise RuntimeError("Groq API Rate Limit Exceeded: Please wait a moment and try again.") from e
        except groq.APIConnectionError as e:
            print(f"[GROQ ERROR] Connection error: {e}")
            raise RuntimeError("Groq API Network Error: Could not connect to Groq cloud servers.") from e
        except groq.APITimeoutError as e:
            print(f"[GROQ ERROR] Request timeout: {e}")
            raise RuntimeError("Groq API Timeout: Request to Groq cloud timed out.") from e
        except groq.APIError as e:
            print(f"[GROQ ERROR] API error: {e}")
            raise RuntimeError(f"Groq API Error: {e.message if hasattr(e, 'message') else str(e)}") from e
        except Exception as e:
            print(f"[GROQ ERROR] Unexpected error in streaming chat: {e}")
            raise RuntimeError(f"LLM Provider Error: {str(e)}") from e

    def _sync_chat(
        self,
        messages: list[dict],
        yield_reasoning: bool = False,
    ) -> str | tuple[str, str | None]:
        """Non-streaming call — returns full answer string (and optional reasoning) with exception handling."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
            )
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content is not None:
                    import re
                    reasoning_str = None
                    think_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                    if think_match:
                        reasoning_str = think_match.group(1).strip()
                    clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    if yield_reasoning:
                        return clean_content, reasoning_str
                    return clean_content
            if yield_reasoning:
                return "I apologize, but the model returned an empty response.", None
            return "I apologize, but the model returned an empty response."
        except groq.AuthenticationError as e:
            print(f"[GROQ ERROR] Invalid API key or unauthorized: {e}")
            raise RuntimeError("Groq API Authentication Error: Invalid API key provided.") from e
        except groq.RateLimitError as e:
            print(f"[GROQ ERROR] Rate limit exceeded: {e}")
            raise RuntimeError("Groq API Rate Limit Exceeded: Please wait a moment and try again.") from e
        except groq.APIConnectionError as e:
            print(f"[GROQ ERROR] Connection error: {e}")
            raise RuntimeError("Groq API Network Error: Could not connect to Groq cloud servers.") from e
        except groq.APITimeoutError as e:
            print(f"[GROQ ERROR] Request timeout: {e}")
            raise RuntimeError("Groq API Timeout: Request to Groq cloud timed out.") from e
        except groq.APIError as e:
            print(f"[GROQ ERROR] API error: {e}")
            raise RuntimeError(f"Groq API Error: {e.message if hasattr(e, 'message') else str(e)}") from e
        except Exception as e:
            print(f"[GROQ ERROR] Unexpected error in sync chat: {e}")
            raise RuntimeError(f"LLM Provider Error: {str(e)}") from e





# ─── Factory + Availability ──────────────────────────────────────────────────

def _is_ollama_reachable() -> bool:
    """Quick ping to check if Ollama is running."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        resp = httpx.get(host, timeout=3.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False


def check_provider_availability() -> dict[str, bool]:
    """
    Check which LLM providers are currently available.
    Used by the /status endpoint so the frontend can show/hide the toggle.

    Returns:
        {"ollama": True/False, "groq": True/False}
    """
    return {
        "ollama": _is_ollama_reachable(),
        "groq": bool(GROQ_API_KEY or os.getenv("GROQ_API_KEY")),
    }



def get_provider(preference: str | None = None) -> LLMProvider:
    """
    Factory: return the appropriate LLM provider based on user preference.

    ROUTING LOGIC:
      preference="online"  → GroqProvider (requires GROQ_API_KEY)
      preference="offline" → OllamaProvider (must be reachable)
      preference="auto"/None →
        1. Try Ollama (ping with 3s timeout)
        2. Fall back to Groq if GROQ_API_KEY is set
        3. Raise clear error if neither is available

    Args:
        preference: "online", "offline", "auto", or None

    Returns:
        An LLMProvider instance ready to call .chat()

    Raises:
        ConnectionError: Ollama requested but not reachable
        EnvironmentError: Groq requested but no API key
        RuntimeError: Auto mode and no provider available
    """
    # Normalize: treat "auto" the same as None
    if preference == "auto":
        preference = None

    # ── Explicit online (Groq) ────────────────────────────────────────────
    if preference == "online":
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "Online mode requested but GROQ_API_KEY is not set. "
                "Add it to your .env file or switch to offline/auto mode."
            )
        print("[LLM_PROVIDER] Using Groq (online — explicitly requested)")
        return GroqProvider()

    # ── Explicit offline (Ollama) ─────────────────────────────────────────
    if preference == "offline":
        provider = OllamaProvider()
        if not provider._ping():
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            raise ConnectionError(
                f"Offline mode requested but Ollama is not reachable at {host}. "
                f"Start Ollama with 'ollama serve' or check OLLAMA_HOST in .env."
            )
        print(f"[LLM_PROVIDER] Using Ollama (offline — explicitly requested)")
        return provider

    # ── Auto mode: try Ollama first, fall back to Groq ────────────────────
    ollama = OllamaProvider()
    if ollama._ping():
        print("[LLM_PROVIDER] Using Ollama (auto — local server detected)")
        return ollama

    if os.getenv("GROQ_API_KEY"):
        print("[LLM_PROVIDER] Ollama not reachable, falling back to Groq (auto)")
        return GroqProvider()

    raise RuntimeError(
        "No LLM provider available.\n"
        "  - To use Ollama (local): install and run 'ollama serve'\n"
        "  - To use Groq (cloud):   set GROQ_API_KEY in your .env file\n"
        "At least one provider must be available to answer questions."
    )
