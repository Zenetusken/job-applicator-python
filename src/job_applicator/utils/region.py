"""Best-effort detection of the host's locale and IANA timezone.

The automation browser advertises these so geo-aware sites (Indeed, LinkedIn)
serve the user's real region instead of a hardcoded US default. The timezone is
the reliable signal — the locale is often left at ``en_US`` even outside the US.

Timezone detection is cross-platform via the ``TZ`` env var, then falls back to
the Unix ``/etc/localtime`` symlink and ``/etc/timezone`` file. Every candidate
is normalised (the ``posix/`` and ``right/`` zoneinfo sub-trees, which ICU and
therefore Playwright reject, are stripped) and validated against the IANA
database before use, so a malformed or non-canonical value can never reach
Playwright and break the browser launch. Hosts where none of these resolve
(notably Windows without ``TZ`` set) fall back to the default; set ``TZ`` or
``browser.timezone`` to pin a region there.
"""

from __future__ import annotations

import functools
import locale as _locale
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_DEFAULT_LOCALE = "en-US"
_DEFAULT_TIMEZONE = "America/New_York"
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)


def _parse_locale(raw: str) -> str:
    """Parse a POSIX/BCP-47 locale string into ``lang`` or ``lang-REGION``.

    Tolerant of forms the host may hand us: encoding suffixes (``.UTF-8``),
    modifiers (``@euro``), numeric region subtags (``es_419``), script subtags
    (``zh_Hans_CN`` -> ``zh-CN``) and language-only values (``en``).
    """
    base = raw.split(".")[0].split("@")[0]
    parts = [p for p in re.split(r"[_-]", base) if p]
    if not parts:
        return _DEFAULT_LOCALE
    lang = parts[0].lower()
    if not (2 <= len(lang) <= 3 and lang.isalpha()):
        return _DEFAULT_LOCALE
    for sub in parts[1:]:
        if len(sub) == 2 and sub.isalpha():
            return f"{lang}-{sub.upper()}"
        if len(sub) == 3 and sub.isdigit():  # UN M.49 region code, e.g. es-419
            return f"{lang}-{sub}"
    return lang


def detect_locale() -> str:
    """Return a BCP-47-ish locale (e.g. ``en-CA``) from the host, else en-US."""
    raw = ""
    try:
        raw = _locale.getlocale()[0] or ""
    except (ValueError, TypeError):
        raw = ""
    if not raw or raw in ("C", "POSIX"):
        raw = os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
    return _parse_locale(raw)


def _normalize_zone(zone: str) -> str:
    """Trim a raw zone string to a canonical IANA name candidate.

    Drops a leading ``:`` (POSIX ``TZ`` notation) and the ``posix/``/``right/``
    zoneinfo sub-tree prefixes — those resolve in the tz database but ICU (hence
    Playwright's ``timezone_id``) rejects them, which would crash the launch.
    """
    z = zone.strip().lstrip(":").strip().strip("/")
    for prefix in ("posix/", "right/"):
        if z.lower().startswith(prefix):
            z = z[len(prefix) :]
    return z


def _is_valid_zone(zone: str) -> bool:
    """True if ``zone`` is a resolvable, concrete IANA timezone name.

    Catches OSError too: a key that resolves to a zoneinfo *directory* (``US``,
    ``America``) raises ``IsADirectoryError`` under the ``tzdata`` backend on some
    platforms, which must be rejected here rather than escape and crash the
    browser launch — the exact failure this gate exists to prevent.
    """
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return False
    return True


def _timezone_candidates() -> Iterator[str]:
    """Yield raw timezone strings from the host, most authoritative first."""
    tz_env = os.environ.get("TZ", "").strip()
    if tz_env:
        yield tz_env
    try:
        link = os.readlink("/etc/localtime")
        if "zoneinfo/" in link:
            yield link.split("zoneinfo/")[-1]
    except OSError:
        pass
    try:
        tz = Path("/etc/timezone").read_text(encoding="utf-8").strip()
        if tz:
            yield tz
    except OSError:
        pass


def detect_timezone() -> str:
    """Return the host IANA timezone (e.g. ``America/Toronto``), else a default.

    Checks ``TZ``, then ``/etc/localtime`` (Linux/macOS), then ``/etc/timezone``
    (Debian-family). Each candidate is normalised and validated against the IANA
    database; the first valid one wins, so a malformed/non-canonical value can
    never reach Playwright. Falls back to a sane default otherwise.
    """
    for candidate in _timezone_candidates():
        norm = _normalize_zone(candidate)
        if norm and _is_valid_zone(norm):
            return norm
    return _DEFAULT_TIMEZONE


def _platform_ua_token() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Macintosh; Intel Mac OS X 10_15_7"
    if system == "Windows":
        return "Windows NT 10.0; Win64; x64"
    return "X11; Linux x86_64"


@functools.lru_cache(maxsize=1)
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
        # Anchor on the browser keyword so e.g. Brave's "Brave Browser 1.71.123
        # Chromium: 130.0..." yields 130, not the leading "71" of its own version.
        match = re.search(r"(?:Chrome|Chromium)[/ :]+(\d{2,})\.", result.stdout)
        if match:
            return match.group(1)
    return None


@functools.lru_cache(maxsize=1)
def detect_chrome_user_agent() -> str:
    """Build a UA matching the host's installed Chrome major version.

    Matching the real browser's UA matters for sites that bind clearance cookies
    to the User-Agent (e.g. Cloudflare-fronted Indeed). Falls back to a recent
    default if no local Chrome/Chromium is found. Cached: the host's browser
    version does not change within a process, and the lookup shells out to the
    browser binary, which we don't want to repeat on every launch.
    """
    major = _detect_chrome_major()
    if not major:
        return _DEFAULT_USER_AGENT
    return (
        f"Mozilla/5.0 ({_platform_ua_token()}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
    )
