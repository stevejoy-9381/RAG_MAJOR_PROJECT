"""
tests/test_llm_provider.py — LLM Provider & Failover Test Suite
──────────────────────────────────────────────────────────────────
Verifies provider detection, model-readiness checking, retry decorator,
and auto mode routing.
"""

import os
import sys
import pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from src.llm_provider import get_provider, check_provider_availability, OllamaProvider, GroqProvider
from src.utils.retry import retry_with_backoff


def test_retry_decorator_success_on_retry():
    """Verify that @retry_with_backoff retries on transient errors and succeeds."""
    attempts = 0

    @retry_with_backoff(retries=3, backoff_factor=1.1, exceptions=(ValueError,))
    def flaky_func():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("Rate limit hit — 429")
        return "success"

    result = flaky_func()
    assert result == "success"
    assert attempts == 2


def test_check_provider_availability():
    """Verify structure of provider availability dict."""
    avail = check_provider_availability()
    assert isinstance(avail, dict)
    assert "ollama" in avail
    assert "groq" in avail


def test_get_provider_invalid_mode():
    """Verify error raised when requesting non-existent provider."""
    os.environ.pop("GROQ_API_KEY", None)
    with pytest.raises((EnvironmentError, ConnectionError, RuntimeError)):
        get_provider("offline")
