"""Filesystem-path utilities."""

from __future__ import annotations


def safe_filename_slug(text: str) -> str:
    """Create a filesystem-safe slug from ``text``.

    Keeps alphanumerics, hyphens, and underscores; replaces everything else with
    an underscore, then caps the result at 30 characters.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:30]
