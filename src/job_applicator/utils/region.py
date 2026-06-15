"""Best-effort detection of the host's locale and IANA timezone.

The automation browser advertises these so geo-aware sites (Indeed, LinkedIn)
serve the user's real region instead of a hardcoded US default. The timezone is
the reliable signal — the locale is often left at ``en_US`` even outside the US.
"""

from __future__ import annotations

import locale as _locale
import os
import re
from pathlib import Path

_DEFAULT_LOCALE = "en-US"
_DEFAULT_TIMEZONE = "America/New_York"


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
