"""Shared cookie persistence + resilient loading for board scrapers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext
from playwright.async_api import Error as PlaywrightError

from job_applicator.utils.logging import get_logger
from job_applicator.utils.secure_store import write_secret_json

logger = get_logger("cookies")


def save_cookies(path: Path, cookies: Any) -> None:
    """Persist cookies as a ``{"cookies": [...]}`` envelope (atomic, 0600)."""
    write_secret_json(path, {"cookies": cookies})


def read_cookies(path: Path) -> list[Any]:
    """Return the cookie list stored at ``path``, or [] if missing/unreadable."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Failed to read cookies from %s: %s", path, exc)
        return []
    return data.get("cookies", []) if isinstance(data, dict) else []


async def load_cookies(context: BrowserContext, path: Path) -> int:
    """Load cookies from ``path`` into ``context``; return the count added.

    ``context.add_cookies`` is all-or-nothing, so one malformed cookie would void
    the whole batch — fall back to adding them one at a time, skipping bad ones.
    """
    cookies = read_cookies(path)
    if not cookies:
        return 0
    try:
        await context.add_cookies(cookies)
        return len(cookies)
    except (ValueError, PlaywrightError):
        added = 0
        for cookie in cookies:
            try:
                await context.add_cookies([cookie])
                added += 1
            except (ValueError, PlaywrightError) as exc:
                logger.warning("Skipping invalid cookie %r: %s", cookie.get("name", "?"), exc)
        return added
