"""
src/utils/retry.py — Retry Decorator with Exponential Backoff
──────────────────────────────────────────────────────────────
Handles transient LLM provider failures (rate limits, network glitches, timeouts)
with exponential backoff and structured logging.
"""

import time
import functools
from typing import Callable, Type, Tuple, Any

from src.config import MAX_LLM_RETRIES, LLM_RETRY_BACKOFF_FACTOR
from src.logger import get_logger, log_error

logger = get_logger("RETRY")


def retry_with_backoff(
    retries: int = MAX_LLM_RETRIES,
    backoff_factor: float = LLM_RETRY_BACKOFF_FACTOR,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for retrying functions on transient failures.

    Args:
        retries: Maximum number of attempts
        backoff_factor: Multiplier for exponential backoff sleep delay
        exceptions: Tuple of exception types to catch and retry on
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            delay = 1.0
            last_exception = None
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    err_msg = str(e)
                    # Check if error indicates transient/retryable condition
                    is_transient = any(
                        term in err_msg.lower()
                        for term in [
                            "rate limit", "429", "timeout", "timed out",
                            "connection error", "503", "service unavailable",
                            "temporarily unavailable", "try again"
                        ]
                    ) or attempt < retries

                    if not is_transient or attempt == retries:
                        logger.error(
                            f"Call to {func.__name__} failed permanently on attempt {attempt}/{retries}: {e}"
                        )
                        raise e

                    logger.warning(
                        f"Transient error in {func.__name__} (attempt {attempt}/{retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

            if last_exception:
                raise last_exception

        return wrapper
    return decorator
