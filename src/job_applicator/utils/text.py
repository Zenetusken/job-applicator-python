"""Small text-matching helpers shared across document analysis."""

from __future__ import annotations

import re


def contains_word(haystack: str, term: str) -> bool:
    """Return True if ``term`` appears in ``haystack`` as a whole token.

    Uses alphanumeric boundaries rather than naive substring containment so a
    term like ``"api"`` does not match inside ``"therapist"`` and ``"roi"``
    does not match inside ``"android"``. Multi-word phrases (``"system
    design"``) and symbol-bearing terms (``"ci/cd"``, ``"fast-paced"``) are
    matched literally; only the alphanumeric edges are guarded.
    """
    if not term:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
    return re.search(pattern, haystack) is not None
