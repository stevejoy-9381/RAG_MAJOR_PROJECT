"""
src/logger.py — Centralized Structured Logging & Metrics
──────────────────────────────────────────────────────────
Provides a structured logger for timing, diagnostics, metrics,
and traceback logging across ingestion, retrieval, LLM, and API endpoints.
"""

import sys
import os
import time
import logging
import traceback
from typing import Any, Callable, Optional


# Ensure UTF-8 logging on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure standard logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def get_logger(name: str) -> logging.Logger:
    """Return a logger instance bound to a module or component name."""
    return logging.getLogger(name)

logger = get_logger("RAG")

def log_event(component: str, event: str, **kwargs: Any) -> None:
    """Log a structured key-value event."""
    details = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[{component}] {event} {details}".strip())

def log_error(component: str, message: str, exc: Optional[BaseException] = None) -> None:
    """Log an error with complete exception traceback when provided."""
    if exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error(f"[{component}] {message}\nTraceback:\n{tb}")
    else:
        logger.error(f"[{component}] {message}")

class Timer:
    """Context manager to measure execution time of code blocks."""
    def __init__(self, name: str, component: str = "PERF"):
        self.name = name
        self.component = component
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = round((time.time() - self.start_time) * 1000, 2)
        if exc_type is None:
            logger.info(f"[{self.component}] {self.name} completed in {self.elapsed} ms")
        else:
            logger.error(f"[{self.component}] {self.name} failed after {self.elapsed} ms: {exc_val}")
