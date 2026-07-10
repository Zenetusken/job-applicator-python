"""Shared fail-closed helpers for unattended résumé tailoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from job_applicator.workflows.tailor import assert_tailored_auto_saveable

if TYPE_CHECKING:
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.embeddings.matching import JobMatcher, MatchResult
    from job_applicator.models import JobListing, ResumeData, StyleGuide, TailoredResume

__all__ = [
    "AutoTailorResult",
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

    The overlay architecture already enforces the source boundary. The result is passed unchanged
    to the shared auto-save integrity gate; this helper never rewrites generated prose.
    """
    tailored = await tailor_engine.tailor_verified(
        resume=resume,
        job=job,
        user_instructions=user_instructions,
        style_guide=style_guide,
        tone_profile=tone_profile,
        matcher=matcher,
        match_result=match_result,
    )
    assert_tailored_auto_saveable(tailored, resume.raw_text)
    return AutoTailorResult(tailored=tailored, attempts=1)
