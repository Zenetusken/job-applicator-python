"""Tests for word-boundary text matching helper (L-5)."""

from __future__ import annotations

import pytest

from job_applicator.utils.text import contains_word


@pytest.mark.parametrize(
    ("haystack", "term", "expected"),
    [
        ("we run a ci/cd pipeline", "ci/cd", True),
        ("fast-paced startup energy", "fast-paced", True),
        ("governance and process improvement here", "process improvement", True),
        ("going from 0 to 1 quickly", "0 to 1", True),
        ("strong api design skills", "api", True),
        # False positives that naive substring matching would wrongly catch:
        ("we ship android apps", "roi", False),
        ("see a good therapist", "api", False),
        ("translate the slate", "sla", False),
        ("questions about seniority", "senior", False),
        ("", "api", False),
        ("anything", "", False),
    ],
)
def test_contains_word(haystack: str, term: str, expected: bool) -> None:
    assert contains_word(haystack, term) is expected
