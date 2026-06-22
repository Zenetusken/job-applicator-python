"""UI-agnostic action layer for the TUI.

The operations a user triggers from inside the app (tailor; cover letter next). Pure
async functions that the app's background workers call, so they're unit-testable without
the UI. Account-safe: local files + the LLM only — never a browser, scraper, applicator,
or login.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from job_applicator.documents.artifacts import write_tailored

if TYPE_CHECKING:
    from job_applicator.config import AppSettings
    from job_applicator.models import JobListing, TailoredResume


async def tailor_job(settings: AppSettings, job: JobListing) -> TailoredResume:
    """Tailor the configured résumé for ``job`` (non-interactive, first version) and write
    the artifact; returns the ``TailoredResume`` with ``output_path`` set.

    Raises ``ResumeNotFoundError`` / ``DocumentError`` / ``LLMError`` (all
    ``JobApplicatorError`` subclasses) on failure — the caller surfaces them. LLM + local
    files only; touches no account.
    """
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.factories import _make_runtime

    resume_data = ResumeLoader().load(settings.resume_path)
    engine = ResumeTailor(settings.llm, runtime=_make_runtime(settings))
    tailored = await engine.tailor(resume=resume_data, job=job, user_instructions="")
    write_tailored(settings.ensure_output_dir(), tailored, when=datetime.now())
    return tailored
