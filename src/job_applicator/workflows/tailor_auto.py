"""Shared unattended résumé-tailoring helpers.

Batch and TUI tailoring both need the same fail-closed path: source-only
instructions, one strict grounding retry, and the same auto-save integrity gate
used by ``tailor --yes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from job_applicator.workflows.tailor import assert_tailored_auto_saveable
from job_applicator.workflows.tailor_policy import (
    STRICT_GROUNDING_FEEDBACK,
    STRICT_NONINTERACTIVE_INSTRUCTIONS,
    source_only_instructions,
)

if TYPE_CHECKING:
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.embeddings.matching import JobMatcher, MatchResult
    from job_applicator.models import JobListing, ResumeData, StyleGuide, TailoredResume

__all__ = [
    "STRICT_GROUNDING_FEEDBACK",
    "STRICT_NONINTERACTIVE_INSTRUCTIONS",
    "AutoTailorResult",
    "source_only_instructions",
    "tailor_auto_verified_saveable",
]


@dataclass(frozen=True)
class AutoTailorResult:
    """Result of unattended tailoring, including the number of LLM attempts used."""

    tailored: TailoredResume
    attempts: int


async def tailor_auto_verified_saveable(
    *,
    tailor_engine: ResumeTailor,
    resume: ResumeData,
    job: JobListing,
    style_guide: StyleGuide | None = None,
    tone_profile: ToneProfile | None = None,
    matcher: JobMatcher | None = None,
    match_result: MatchResult | None = None,
    user_instructions: str = "",
) -> AutoTailorResult:
    """Generate a verified tailored résumé that is safe for unattended saving.

    The first draft uses the same strict source-only policy as ``tailor --yes``.
    If grounding finds unsupported claims, the helper performs one strict refine
    and then runs the same fail-closed auto-save integrity gate.
    """
    tailored = await tailor_engine.tailor_verified(
        resume=resume,
        job=job,
        user_instructions=source_only_instructions(user_instructions),
        style_guide=style_guide,
        tone_profile=tone_profile,
        matcher=matcher,
        match_result=match_result,
    )
    attempts = 1
    if tailored.grounding_report is not None and not tailored.grounding_report.clean:
        tailored = await tailor_engine.refine_verified(
            resume,
            tailored,
            STRICT_GROUNDING_FEEDBACK,
            job,
            matcher=matcher,
            style_guide=style_guide,
            tone_profile=tone_profile,
        )
        attempts += 1
    assert_tailored_auto_saveable(tailored, resume.raw_text)
    return AutoTailorResult(tailored=tailored, attempts=attempts)
