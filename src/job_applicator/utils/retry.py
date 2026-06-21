"""Retry and backoff utilities."""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Callable
from typing import Any, TypeVar

from job_applicator.utils.logging import get_logger

logger = get_logger("retry")

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator for retrying sync functions with exponential backoff."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay *= 0.5 + random.random()  # noqa: S311 - jitter
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        delay,
                        exc,
                    )
                    import time

                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    exclude: tuple[type[Exception], ...] = (),
) -> Callable[[F], F]:
    """Decorator for retrying async functions with exponential backoff.

    ``exclude`` types are never retried even if they match ``exceptions`` (they
    are a subset to fail fast on — e.g. a circuit-breaker-open error, which a
    retry would only re-hit against the same open breaker).
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exclude:
                    raise
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay *= 0.5 + random.random()  # noqa: S311 - jitter
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
