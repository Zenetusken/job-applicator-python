"""UI-agnostic action layer for the TUI.

The operations a user triggers from inside the app (tailor, cover letter, search, apply).
Pure async functions the app's background workers call, so they're unit-testable without
the UI. ``tailor_job`` / ``cover_letter_job`` are account-safe (LLM + local files only);
``search_jobs`` / ``apply_job`` are ACCOUNT-TOUCHING (real browser) — each is marked
below, and the UI gates them behind an explicit confirm.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from job_applicator.documents.artifacts import write_cover_letter, write_tailored

if TYPE_CHECKING:
    from job_applicator.config import AppSettings
    from job_applicator.jobs_store import JobStore
    from job_applicator.models import (
        ApplicationResult,
        CoverLetterResult,
        JobListing,
        TailoredResume,
    )
    from job_applicator.scrapers.base import SearchParams


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

    resume_data = await asyncio.to_thread(ResumeLoader().load, settings.resume_path)
    engine = ResumeTailor(settings.llm, runtime=_make_runtime(settings))
    tailored = await engine.tailor(resume=resume_data, job=job, user_instructions="")
    output_dir = await asyncio.to_thread(settings.ensure_output_dir)
    await asyncio.to_thread(write_tailored, output_dir, tailored, when=datetime.now())
    return tailored


async def cover_letter_job(settings: AppSettings, job: JobListing) -> CoverLetterResult:
    """Generate a cover letter for ``job`` from the configured résumé and write the
    artifact; returns the ``CoverLetterResult`` with ``output_path`` set.

    Raises ``JobApplicatorError`` subclasses on failure. LLM + local files only; touches
    no account.
    """
    from job_applicator.cli import _detect_tone, _load_user_profile
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.documents.tone_detector import ToneDetector
    from job_applicator.factories import _make_runtime
    from job_applicator.models import CoverLetterResult

    resume_data = await asyncio.to_thread(ResumeLoader().load, settings.resume_path)
    tone_section = ToneDetector().format_for_prompt(_detect_tone(job))
    generator = CoverLetterGenerator(settings.llm, runtime=_make_runtime(settings))
    letter = await generator.generate(
        job, _load_user_profile(settings), resume_data, tone_section=tone_section
    )
    result = CoverLetterResult(
        job_title=job.title,
        job_company=job.company,
        job_url=str(job.url),
        cover_letter_text=letter,
        attempt=1,
        prompt_version="1.0",
    )
    output_dir = await asyncio.to_thread(settings.ensure_output_dir)
    await asyncio.to_thread(write_cover_letter, output_dir, result, when=datetime.now())
    return result


async def search_jobs(settings: AppSettings, store: JobStore, params: SearchParams) -> int:
    """Scrape ``params`` and persist the results to ``store`` (found stage); returns the
    count.

    ⚠ ACCOUNT-TOUCHING — the only action here that is: it launches a real browser on the
    configured board. The TUI gates it behind an explicit, warned confirm (the search
    modal), and tests never let it construct a real browser.
    """
    from job_applicator.factories import _make_browser, _make_scraper

    site = params.board.value
    async with _make_browser(site, settings) as browser:
        scraper = _make_scraper(site, browser, settings)
        jobs = await scraper.scrape(params)
    for job in jobs:
        store.upsert_job(job, source_query=params.query)
    return len(jobs)


async def apply_job(settings: AppSettings, job: JobListing, *, submit: bool) -> ApplicationResult:
    """Apply to ``job``. Dry-run by default (fills the form, never submits). A real submit
    (``submit=True``) respects the daily cap and skips already-applied jobs — both checked
    BEFORE any browser launches — and is recorded in ``ApplicationState`` on success.

    ⚠ ACCOUNT-TOUCHING: launches a real browser, and on a real submit sends an actual
    application. The TUI gates the real-submit path behind an explicit danger checkbox.
    """
    from datetime import UTC, datetime

    from job_applicator.factories import _make_applicator, _make_browser
    from job_applicator.models import ApplicationResult, ApplicationStatus
    from job_applicator.state import ApplicationState

    state = ApplicationState()
    if submit:  # cap + dedup gates fire before we ever open a browser
        if state.has_applied(str(job.url)):
            return ApplicationResult(
                job=job, status=ApplicationStatus.ALREADY_APPLIED, timestamp=datetime.now(UTC)
            )
        cap = settings.target.max_applications_per_day
        if state.count_today(job.board.value) >= cap:
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.SKIPPED,
                timestamp=datetime.now(UTC),
                notes=f"daily cap ({cap}) reached",
            )

    site = job.board.value
    async with _make_browser(site, settings) as browser:
        applicator = _make_applicator(site, browser, settings)
        result = await applicator.apply(job, submit=submit)
    if submit and result.status == ApplicationStatus.SUBMITTED:
        state.record(result)  # ApplicationState is the authority for "applied"
    return result
