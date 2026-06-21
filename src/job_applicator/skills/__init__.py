"""Skill normalization and filtering helpers."""

from __future__ import annotations

from job_applicator.skills.normalization import (
    HARD_NEGATIVE_SKILLS,
    NORMALIZATION_MAP,
    is_hard_negative,
    normalize_skill,
)

__all__ = [
    "HARD_NEGATIVE_SKILLS",
    "NORMALIZATION_MAP",
    "is_hard_negative",
    "normalize_skill",
]
