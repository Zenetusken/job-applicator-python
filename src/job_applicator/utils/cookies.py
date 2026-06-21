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


def _normalize_cookie(entry: Any) -> dict[str, Any] | None:
    """Best-effort conversion of an exported cookie dict to Playwright format.

    Handles common browser-extension exports (e.g. `expirationDate` instead of
    `expires`, `sameSite: "no_restriction"`). Returns None for unusable entries.
    """
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    value = entry.get("value")
    if not name or value is None:
        return None
    out: dict[str, Any] = {"name": str(name), "value": str(value)}
    domain = entry.get("domain")
    if domain:
        out["domain"] = str(domain)
        out["path"] = str(entry.get("path", "/"))
    else:
        out["url"] = str(entry.get("url", "https://www.linkedin.com"))
    exp = entry.get("expires", entry.get("expirationDate"))
    if isinstance(exp, int | float) and not isinstance(exp, bool) and exp > 0:
        out["expires"] = float(exp)
    for key in ("httpOnly", "secure"):
        if key in entry:
            out[key] = bool(entry[key])
    same = entry.get("sameSite")
    if isinstance(same, str):
        mapped = {"no_restriction": "None", "none": "None", "lax": "Lax", "strict": "Strict"}.get(
            same.lower()
        )
        if mapped:
            out["sameSite"] = mapped
    # Chromium rejects a SameSite=None cookie that is not Secure, so an export
    # that omits `secure` would otherwise yield a silently-dropped session cookie.
    if out.get("sameSite") == "None":
        out["secure"] = True
    return out


def _cookiejar_to_playwright(cookie: Any) -> dict[str, Any] | None:
    """Convert a stdlib cookiejar cookie (from browser_cookie3) to Playwright form."""
    raw: dict[str, Any] = {
        "name": cookie.name,
        "value": cookie.value,
        "domain": cookie.domain,
        "path": cookie.path or "/",
        "secure": bool(getattr(cookie, "secure", False)),
    }
    expires = getattr(cookie, "expires", None)
    if expires:
        raw["expires"] = expires
    # cookiejar keeps httpOnly as a nonstandard attr (browser_cookie3 sets it);
    # propagate it so the imported cookie matches the real browser cookie.
    rest = getattr(cookie, "_rest", None) or {}
    if any(str(key).lower() == "httponly" for key in rest):
        raw["httpOnly"] = True
    return _normalize_cookie(raw)
