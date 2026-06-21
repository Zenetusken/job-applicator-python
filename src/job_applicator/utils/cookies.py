"""Shared cookie logic for board scrapers: persistence, resilient loading, format
conversion, browser-store import, and per-board import policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext
from playwright.async_api import Error as PlaywrightError

from job_applicator.exceptions import CookieError
from job_applicator.utils.logging import get_logger
from job_applicator.utils.secure_store import write_secret_json
from job_applicator.utils.url import host_matches

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


def _cookies_from_browser(browser: str, base_domain: str) -> list[dict[str, Any]]:
    """Read a site's cookies directly from a local browser's cookie store.

    Uses browser_cookie3, which decrypts the browser's on-disk cookie database —
    this reaches httpOnly cookies (like LinkedIn ``li_at`` or Cloudflare
    ``cf_clearance``) that page scripts cannot. Only invoked via ``--from-browser``.

    Raises ``CookieError`` (with a user-facing message) on a missing optional
    dependency, an unsupported browser, or an unreadable cookie store — the CLI
    catches it and renders the message; pure utils stay free of typer/console.
    """
    try:
        import browser_cookie3
    except ImportError as exc:
        raise CookieError(
            '--from-browser needs the optional dependency: pip install "job-applicator[browser]"'
        ) from exc

    loaders = {
        "chrome": browser_cookie3.chrome,
        "chromium": browser_cookie3.chromium,
        "brave": browser_cookie3.brave,
        "edge": browser_cookie3.edge,
        "firefox": browser_cookie3.firefox,
    }
    loader = loaders.get(browser.lower())
    if loader is None:
        raise CookieError(f"Unsupported browser '{browser}'. Choose: {', '.join(loaders)}.")
    try:
        jar = loader(domain_name=base_domain)
    except Exception as exc:  # browser_cookie3 raises various OS/keyring/db errors
        raise CookieError(
            f"Could not read {browser} cookies: {exc}. Is {browser} installed and your "
            "login keyring unlocked?"
        ) from exc
    # browser_cookie3's domain filter is a SUBSTRING match, so it can sweep in
    # look-alike hosts (e.g. notlinkedin.com); keep only genuine site cookies.
    return [
        c
        for c in (_cookiejar_to_playwright(ck) for ck in jar)
        if c and host_matches(str(c.get("domain", "")), base_domain)
    ]


@dataclass(frozen=True)
class _SiteSpec:
    """Per-board rules for ``import-cookies``, so the command body stays board-agnostic.

    ``required_cookie`` is a hard gate (absent => refuse, since the session can't
    work without it). ``preferred_cookie`` is a soft signal (absent => warn but
    save). ``session_flags`` enables the LinkedIn ``--li-at``/``--jsessionid``
    seed inputs. ``feed_verify`` runs the post-import logged-in feed check.
    """

    cookie_path: Path
    base_domain: str
    required_cookie: str | None
    preferred_cookie: str | None
    session_flags: bool
    feed_verify: bool


def _site_specs() -> dict[str, _SiteSpec]:
    from job_applicator.scrapers.indeed import IndeedScraper
    from job_applicator.scrapers.linkedin import LinkedInScraper

    return {
        # li_at is the LinkedIn session token: nothing authenticates without it.
        "linkedin": _SiteSpec(
            cookie_path=LinkedInScraper.COOKIE_PATH,
            base_domain="linkedin.com",
            required_cookie="li_at",
            preferred_cookie=None,
            session_flags=True,
            feed_verify=True,
        ),
        # Indeed search is public — no cookie is strictly required. cf_clearance
        # (Cloudflare) is what actually helps a warm session avoid challenges, so
        # it's preferred-not-required; CTK and friends are mere tracking cookies.
        "indeed": _SiteSpec(
            cookie_path=IndeedScraper.COOKIE_PATH,
            base_domain="indeed.com",
            required_cookie=None,
            preferred_cookie="cf_clearance",
            session_flags=False,
            feed_verify=False,
        ),
    }
