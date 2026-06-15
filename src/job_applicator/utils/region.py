"""Best-effort detection of the host's locale and IANA timezone.

The automation browser advertises these so geo-aware sites (Indeed, LinkedIn)
serve the user's real region instead of a hardcoded US default. The timezone is
the reliable signal — the locale is often left at ``en_US`` even outside the US.
"""

from __future__ import annotations

import locale as _locale
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

_DEFAULT_LOCALE = "en-US"
_DEFAULT_TIMEZONE = "America/New_York"
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)


def detect_locale() -> str:
    """Return a BCP-47-ish locale (e.g. ``en-CA``) from the host, else en-US."""
    raw = ""
    try:
        raw = _locale.getlocale()[0] or ""
    except (ValueError, TypeError):
        raw = ""
    if not raw or raw in ("C", "POSIX"):
        raw = os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
    m = re.match(r"([A-Za-z]{2})[_-]([A-Za-z]{2})", raw)
    if m:
        return f"{m.group(1).lower()}-{m.group(2).upper()}"
    return _DEFAULT_LOCALE


def detect_timezone() -> str:
    """Return the host IANA timezone (e.g. ``America/Toronto``), else a default.

    Reads ``/etc/localtime`` (Linux/macOS) then ``/etc/timezone`` (Debian-family).
    This is the most reliable locally-readable geo signal — far more so than the
    locale, which Canadian/UK/etc. users frequently leave at ``en_US``.
    """
    try:
        link = os.readlink("/etc/localtime")
        if "zoneinfo/" in link:
            return link.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    try:
        tz = Path("/etc/timezone").read_text(encoding="utf-8").strip()
        if tz:
            return tz
    except OSError:
        pass
    return _DEFAULT_TIMEZONE


def _platform_ua_token() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Macintosh; Intel Mac OS X 10_15_7"
    if system == "Windows":
        return "Windows NT 10.0; Win64; x64"
    return "X11; Linux x86_64"


def _detect_chrome_major() -> str | None:
    candidates = (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "brave-browser",
    )
    for exe in candidates:
        path = shutil.which(exe)
        if not path:
            continue
        try:
            result = subprocess.run(  # noqa: S603 - fixed args, resolved executable path
                [path, "--version"], capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.search(r"\b(\d{2,})\.\d", result.stdout)
        if match:
            return match.group(1)
    return None


def detect_chrome_user_agent() -> str:
    """Build a UA matching the host's installed Chrome major version.

    Matching the real browser's UA matters for sites that bind clearance cookies
    to the User-Agent (e.g. Cloudflare-fronted Indeed). Falls back to a recent
    default if no local Chrome/Chromium is found.
    """
    major = _detect_chrome_major()
    if not major:
        return _DEFAULT_USER_AGENT
    return (
        f"Mozilla/5.0 ({_platform_ua_token()}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
    )
