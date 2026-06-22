"""Shared helpers that build the LLM-generation inputs — the applicant's ``UserProfile``
and a job's tone — from settings / a job listing.

Extracted from ``cli.py`` so the TUI action layer and the ``workflows`` package don't
import CLI internals (a layering inversion): the CLI, the workflows, and the TUI all
import these from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from job_applicator.models import UserProfile

if TYPE_CHECKING:
    from job_applicator.config import AppSettings
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.models import JobListing


def _detect_tone(job: JobListing) -> ToneProfile:
    """Detect job posting tone deterministically via keyword matching."""
    from job_applicator.documents.tone_detector import ToneDetector

    return ToneDetector().detect(
        title=job.title,
        description=job.description,
        requirements=job.requirements,
    )


def _load_user_profile(settings: AppSettings) -> UserProfile:
    """Load user profile from settings."""
    name_parts = settings.profile_name.split() if settings.profile_name else ["User"]
    return UserProfile(
        first_name=name_parts[0],
        last_name=name_parts[-1] if len(name_parts) > 1 else "",
        email=settings.target.linkedin_email,
        phone="",
        resume_path=settings.resume_path,
    )
