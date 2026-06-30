"""Filesystem-path utilities."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path


def safe_filename_slug(text: str) -> str:
    """Create a filesystem-safe slug from ``text``.

    Keeps alphanumerics, hyphens, and underscores; replaces everything else with
    an underscore, then caps the result at 30 characters.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:30]


def set_owner_only(path: Path, mode: int) -> None:
    """Best-effort chmod to owner-only permissions (0o600 for a file, 0o700 for a dir).

    Privacy hygiene, NOT load-bearing: artifacts can hold résumé-derived data / PII, so on a
    shared machine they should be owner-only. A chmod failure (e.g. an exotic filesystem that
    doesn't support it) must never break the write it follows, so OSError is suppressed.
    """
    with suppress(OSError):
        path.chmod(mode)
