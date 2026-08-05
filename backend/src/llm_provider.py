"""
src/llm_provider.py — Dual LLM Provider Abstraction (Ollama + Groq)
────────────────────────────────────────────────────────────────────
Provides a unified interface for calling local Ollama instance
or Groq cloud API with automatic model-readiness detection, exponential
backoff retries, and seamless auto-failover.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Generator, Optional, Union, Tuple

import httpx
import groq
from groq import Groq
from dotenv import load_dotenv

from src.config import (
    GROQ_API_KEY, GROQ_MODEL, MODEL_NAME,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
)
from src.utils.retry import retry_with_backoff
from src.logger import get_logger, log_event, log_error

logger = get_logger("LLM_PROVIDER")
load_dotenv()

_shared_groq_client: Optional[Groq] = None


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
        logger.info(f"Groq Provider Connected: model={GROQ_MODEL}")
    return _shared_groq_client


# ─── Abstract Base ────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base for LLM providers."""

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
        yield_reasoning: bool = False,
    ) -> Union[str, Tuple[str, Optional[str]], Generator[Union[str, Tuple[str, str]], None, None]]:
        ...


# ─── Ollama Provider ─────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """Local LLM provider via Ollama (http://localhost:11434)."""

    def __init__(self):
        self._host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self._model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    @property
    def name(self) -> str:
        return "ollama"

    def _ping(self, timeout: float = 3.0) -> bool:
        """Check if Ollama server is running AND target model is pulled."""
        try:
            resp = httpx.get(f"{self._host}/api/tags", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                if any(self._model in m or m in self._model for m in models):
                    return True
                logger.warning(f"Ollama running but model '{self._model}' not found in {models}")
                return False
            return False
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    def chat(
        self,
        messages: list[dict],
        stream: bool = True,
        yield_reasoning: bool = False,
    ) -> Union[str, Tuple[str, Optional[str]], Generator[Union[str, Tuple[str, str]], None, None]]:
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
            return self._stream_chat(url, payload, yield_reasoning=yield_reasoning)
        else:
            return self._sync_chat(url, payload, yield_reasoning=yield_reasoning)

    def _stream_chat(
        self, url: str, payload: dict, yield_reasoning: bool = False
    ) -> Generator[Union[str, Tuple[str, str]], None, None]:
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
                            yield ("token", content) if yield_reasoning else content
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    @retry_with_backoff(retries=3, backoff_factor=1.5)
    def _sync_chat(
        self, url: str, payload: dict, yield_reasoning: bool = False
    ) -> Union[str, Tuple[str, Optional[str]]]:
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            if yield_reasoning:
                return content, None
            return content


# ─── Groq Provider ───────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    """Cloud LLM provider via Groq API."""

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
    ) -> Union[str, Tuple[str, Optional[str]], Generator[Union[str, Tuple[str, str]], None, None]]:
        if stream:
            return self._stream_chat(messages, yield_reasoning=yield_reasoning)
        else:
            return self._sync_chat(messages, yield_reasoning=yield_reasoning)

    def _stream_chat(
        self,
        messages: list[dict],
        yield_reasoning: bool = False,
    ) -> Generator[Union[str, Tuple[str, str]], None, None]:
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
            log_error("GROQ", "Authentication error", e)
            raise RuntimeError("Groq API Authentication Error: Invalid API key provided.") from e
        except groq.RateLimitError as e:
            log_error("GROQ", "Rate limit error", e)
            raise RuntimeError("Groq API Rate Limit Exceeded: Please wait a moment and try again.") from e
        except groq.APIConnectionError as e:
            log_error("GROQ", "Connection error", e)
            raise RuntimeError("Groq API Network Error: Could not connect to Groq cloud servers.") from e
        except groq.APITimeoutError as e:
            log_error("GROQ", "Timeout error", e)
            raise RuntimeError("Groq API Timeout: Request to Groq cloud timed out.") from e
        except Exception as e:
            log_error("GROQ", "Unexpected error in streaming", e)
            raise RuntimeError(f"LLM Provider Error: {str(e)}") from e

    @retry_with_backoff(retries=3, backoff_factor=1.5)
    def _sync_chat(
        self,
        messages: list[dict],
        yield_reasoning: bool = False,
    ) -> Union[str, Tuple[str, Optional[str]]]:
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
            fallback = "I apologize, but the model returned an empty response."
            return (fallback, None) if yield_reasoning else fallback
        except Exception as e:
            log_error("GROQ", "Error in sync chat", e)
            raise RuntimeError(f"Groq API Error: {str(e)}") from e


# ─── Factory + Availability ──────────────────────────────────────────────────

def _is_ollama_reachable() -> bool:
    """Quick check if Ollama host is running and model is available."""
    provider = OllamaProvider()
    return provider._ping()


def check_provider_availability() -> dict[str, bool]:
    """Check which LLM providers are currently available."""
    return {
        "ollama": _is_ollama_reachable(),
        "groq": bool(GROQ_API_KEY or os.getenv("GROQ_API_KEY")),
    }


def get_provider(preference: Optional[str] = None) -> LLMProvider:
    """
    Factory: return appropriate LLM provider with failover logic.
    """
    if preference == "auto":
        preference = None

    if preference == "online":
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("Online mode requested but GROQ_API_KEY is not set.")
        logger.info("[LLM_PROVIDER] Using Groq (online mode)")
        return GroqProvider()

    if preference == "offline":
        provider = OllamaProvider()
        if not provider._ping():
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            raise ConnectionError(f"Offline mode requested but Ollama model is not reachable at {host}.")
        logger.info("[LLM_PROVIDER] Using Ollama (offline mode)")
        return provider

    # Auto mode: try Ollama, fall back to Groq
    ollama = OllamaProvider()
    if ollama._ping():
        logger.info("[LLM_PROVIDER] Using Ollama (auto — local model ready)")
        return ollama

    if os.getenv("GROQ_API_KEY"):
        logger.info("[LLM_PROVIDER] Ollama not available, falling back to Groq (auto mode)")
        return GroqProvider()

    raise RuntimeError(
        "No LLM provider available. "
        "Either start Ollama locally or set GROQ_API_KEY in your .env file."
    )
