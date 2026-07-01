"""Tests for skill normalization and hard-negative filtering."""

from __future__ import annotations

import pytest

from job_applicator.skills import HARD_NEGATIVE_SKILLS, is_hard_negative, normalize_skill


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Python 3", "Python"),
        ("python3", "Python"),
        ("Python Programming", "Python"),
        ("nodeJS", "Node.js"),
        ("react js", "React"),
        ("AWS Lambda", "AWS Lambda"),
        ("Terraform", "Terraform"),
        ("cicd", "CI/CD"),
    ],
)
def test_normalize_skill_aliases(raw: str, expected: str) -> None:
    assert normalize_skill(raw) == expected


def test_normalize_skill_unknown_returns_unchanged() -> None:
    assert normalize_skill("Rust") == "Rust"


@pytest.mark.parametrize(
    "term",
    [
        "team player",
        "Communication Skills",
        "Detail-Oriented",
        "Problem Solving",
        "Fast Paced",
        "Remote Work",
    ],
)
def test_is_hard_negative_catches_generic_traits(term: str) -> None:
    assert is_hard_negative(term) is True


def test_is_hard_negative_false_for_technical_skills() -> None:
    assert is_hard_negative("Python") is False
    assert is_hard_negative("AWS Lambda") is False


def test_hard_negative_list_is_frozen() -> None:
    assert "team player" in HARD_NEGATIVE_SKILLS


def test_is_hard_negative_keeps_short_real_skills_drops_noise() -> None:
    """Short REAL skills (C#, Go, R, AI, ML) are NOT hard-negatives — the old `len <= 2` rule
    dropped them on both the résumé and requirement sides, silently losing them from coverage. Pure
    punctuation/empty (noise) and generic traits still ARE hard-negatives."""
    from job_applicator.skills import is_hard_negative

    for s in ("C#", "Go", "R", "AI", "ML", "C", "F#", "SQL", "AWS"):
        assert not is_hard_negative(s), s
    for s in ("•", "-", "|", "", "   ", "Teamwork", "Communication"):
        assert is_hard_negative(s), repr(s)
