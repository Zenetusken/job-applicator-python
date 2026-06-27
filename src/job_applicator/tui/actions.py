"""UI-agnostic action layer for the TUI.

The operations a user triggers from inside the app (tailor, cover letter, search, apply).
Pure async functions the app's background workers call, so they're unit-testable without
the UI. ``tailor_job`` / ``cover_letter_job`` are account-safe (LLM + local files only);
``search_jobs`` / ``apply_job`` are ACCOUNT-TOUCHING (real browser) — each is marked
below, and the UI gates them behind an explicit confirm.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from job_applicator.documents.artifacts import (
    write_cover_letter,
    write_cover_letter_pdf,
    write_tailored,
    write_tailored_pdf,
)
from job_applicator.documents.job_category import detect_job_category
from job_applicator.models import Format

if TYPE_CHECKING:
    from collections.abc import Callable

    from job_applicator.config import AppSettings
    from job_applicator.embeddings.matching import MatchResult
    from job_applicator.jobs_store import JobStore
    from job_applicator.models import (
        ApplicationResult,
        ATSCompatibilityResult,
        CoverLetterResult,
        JobListing,
        StyleGuide,
        TailoredResume,
    )
    from job_applicator.scrapers.base import SearchParams

logger = logging.getLogger(__name__)


async def _load_style_guide(settings: AppSettings, style_guide_path: str) -> StyleGuide | None:
    """Analyze the configured style-guide path(s) into a ``StyleGuide``.

    Returns ``None`` when no path is configured. Errors are raised as
    ``JobApplicatorError`` subclasses so the caller can toast them.
    """
    if not style_guide_path:
        return None
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.factories import _make_runtime

    generator = CoverLetterGenerator(settings.llm, runtime=_make_runtime(settings))
    return await generator.load_style_guide(style_guide_path)


async def tailor_job(
    settings: AppSettings,
    job: JobListing,
    *,
    style_guide_path: str = "",
    output_format: Format = Format.TXT,
    template: str | None = None,
) -> TailoredResume:
    """Tailor the configured résumé for ``job`` (non-interactive, first version) and write
    the artifact; returns the ``TailoredResume`` with ``output_path`` set.

    When ``style_guide_path`` is set, the tailored résumé mimics that style.
    ``output_format`` controls whether a text, PDF, or both artifacts are written.
    ``template`` overrides the configured résumé PDF template when ``output_format`` is
    ``pdf`` or ``both``.

    Raises ``ResumeNotFoundError`` / ``DocumentError`` / ``LLMError`` (all
    ``JobApplicatorError`` subclasses) on failure — the caller surfaces them. LLM + local
    files only; touches no account.
    """
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.factories import _make_runtime

    resume_data = await asyncio.to_thread(ResumeLoader().load, settings.resume_path)
    style = await _load_style_guide(settings, style_guide_path)
    engine = ResumeTailor(settings.llm, runtime=_make_runtime(settings))
    tailored = await engine.tailor(
        resume=resume_data, job=job, user_instructions="", style_guide=style
    )
    output_dir = await asyncio.to_thread(settings.ensure_output_dir)
    when = datetime.now()
    category = detect_job_category(job)
    effective_template = template or settings.output.resume_template

    if output_format == Format.TXT:
        await asyncio.to_thread(write_tailored, output_dir, tailored, when=when)
    elif output_format == Format.PDF:
        await write_tailored_pdf(
            output_dir,
            tailored,
            settings,
            template=effective_template,
            category=category,
            when=when,
        )
    else:  # both
        await asyncio.to_thread(write_tailored, output_dir, tailored, when=when)
        await write_tailored_pdf(
            output_dir,
            tailored,
            settings,
            template=effective_template,
            category=category,
            when=when,
        )
    return tailored


async def cover_letter_job(
    settings: AppSettings,
    job: JobListing,
    tailored_resume_path: str = "",
    *,
    style_guide_path: str = "",
    output_format: Format = Format.TXT,
    template: str | None = None,
) -> CoverLetterResult:
    """Generate a cover letter for ``job`` from the configured résumé and write the
    artifact; returns the ``CoverLetterResult`` with ``output_path`` set.

    When ``tailored_resume_path`` points at an existing tailored-résumé artifact, the
    letter draws on that TAILORED text (so a cover letter for a tailored job reflects it) —
    best-effort: a read failure falls back to the original résumé.

    When ``style_guide_path`` is set, the letter mimics that writing style.
    ``output_format`` controls whether a text, PDF, or both artifacts are written.
    ``template`` overrides the configured cover-letter PDF template when ``output_format``
    is ``pdf`` or ``both``.

    Raises ``JobApplicatorError`` subclasses on failure. LLM + local files only; touches
    no account.
    """
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.documents.tone_detector import ToneDetector
    from job_applicator.factories import _make_runtime
    from job_applicator.models import CoverLetterResult
    from job_applicator.utils.profile import _detect_tone, _load_user_profile

    resume_data = await asyncio.to_thread(ResumeLoader().load, settings.resume_path)
    tailored_text = ""
    if tailored_resume_path:
        try:
            tailored_text = await asyncio.to_thread(
                Path(tailored_resume_path).read_text, encoding="utf-8"
            )
        except OSError:  # artifact gone/unreadable → fall back to the original résumé
            logger.warning("cover letter: tailored résumé unreadable; using the original")
    tone_section = ToneDetector().format_for_prompt(_detect_tone(job))
    style = await _load_style_guide(settings, style_guide_path)
    generator = CoverLetterGenerator(settings.llm, runtime=_make_runtime(settings))
    letter = await generator.generate(
        job,
        _load_user_profile(settings, resume_name=resume_data.name),
        resume_data,
        style_guide=style,
        tone_section=tone_section,
        tailored_resume_text=tailored_text,
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
    when = datetime.now()
    category = detect_job_category(job)
    effective_template = template or settings.output.cover_letter_template

    if output_format == Format.TXT:
        await asyncio.to_thread(write_cover_letter, output_dir, result, when=when)
    elif output_format == Format.PDF:
        await write_cover_letter_pdf(
            output_dir, result, settings, template=effective_template, category=category, when=when
        )
    else:  # both
        await asyncio.to_thread(write_cover_letter, output_dir, result, when=when)
        await write_cover_letter_pdf(
            output_dir, result, settings, template=effective_template, category=category, when=when
        )
    return result


async def search_jobs(
    settings: AppSettings,
    store: JobStore,
    params: SearchParams,
    on_progress: Callable[[str], None] | None = None,
    on_job: Callable[[JobListing], None] | None = None,
) -> int:
    """Scrape ``params``, score against the résumé if one is set (→ matched, with scores;
    else → found), and persist to ``store``; returns the scraped count.

    ``on_progress(msg)`` (optional) is called at each phase boundary AND per scraped
    card ("Scraping job 7/25…") so the UI shows live progress instead of looking frozen
    during the long scrape/score. (Scoring stays phase-level: it runs in a worker thread,
    where a direct UI update would be unsafe; the batch embed dominates it anyway.)

    ``on_job(job)`` (optional) is called as each listing is scraped — each is persisted to
    the store (as ``found``) immediately, THEN ``on_job`` fires, so a UI repaint from the
    store shows results streaming in instead of all at once. Scoring afterwards re-upserts
    them as ``matched`` (an upgrade; the store never downgrades an already-advanced job).

    ⚠ ACCOUNT-TOUCHING — the only action here that is: it launches a real browser on the
    configured board. The TUI gates it behind an explicit, warned confirm (the search
    modal), and tests never let it construct a real browser. The scoring step is offline.
    """
    from job_applicator.factories import _make_browser, _make_scraper

    def progress(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    def emit(job: JobListing) -> None:
        # Persist FIRST (as found) so a store-driven UI repaint sees it, THEN notify.
        store.upsert_job(job, source_query=params.query)
        if on_job is not None:
            on_job(job)

    site = params.board.value  # wire form for the factories
    site_name = params.board.display_name  # proper casing for the user-facing phase messages
    progress(f"Opening a browser on {site_name}…")
    async with _make_browser(site, settings) as browser:
        scraper = _make_scraper(site, browser, settings)
        progress(f"Searching {site_name} for '{params.query}'…")
        # Per-item progress ("Scraping job 7/25…") replaces the static "Searching…" as
        # each card is processed; on_job streams each listing in as it lands. The scraper
        # runs on this event loop, so the sync UI/store sinks act directly (no
        # call_from_thread) — same pattern as the phase msgs.
        jobs = await scraper.scrape(params, on_progress=on_progress, on_job=emit)

    # Score against the résumé when one is configured, so results arrive ranked with match
    # scores (search → matched). Best-effort: a résumé/embedding failure must never lose
    # the scraped jobs — they are already persisted as found (above / below).
    matches = None
    if settings.resume_path and jobs:
        progress(f"Scoring {len(jobs)} job(s) against your résumé…")
        try:
            matches = await _score_jobs(settings, jobs)
        except Exception:
            logger.warning("search: scoring failed; persisting jobs unscored", exc_info=True)
    if matches is not None:
        for m in matches:
            store.upsert_match(m, source_query=params.query)
    else:
        # No scoring: persist as found. Idempotent for jobs already streamed via emit;
        # this is also the path that persists for a non-streaming scraper (one that never
        # calls on_job — e.g. a test double), so results are never lost either way.
        for job in jobs:
            store.upsert_job(job, source_query=params.query)
    return len(jobs)


async def _score_jobs(settings: AppSettings, jobs: list[JobListing]) -> list[MatchResult]:
    """Load the résumé and rank ``jobs`` against it."""
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.factories import _make_runtime

    resume = await asyncio.to_thread(ResumeLoader().load, settings.resume_path)
    runtime = _make_runtime(settings, name="tui-score")
    matcher = JobMatcher(
        settings.embedding, settings.llm, runtime, grounding_mode=settings.skills.grounding_mode
    )
    return await matcher.rank_jobs(resume, jobs, len(jobs))


async def apply_job(
    settings: AppSettings,
    job: JobListing,
    *,
    submit: bool,
    cover_letter: str | None = None,
) -> ApplicationResult:
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
        # Dedup on the same statuses the CLI uses ({SUBMITTED, ALREADY_APPLIED}) so the TUI
        # doesn't re-attempt a job the applicator already found applied.
        if state.has_applied(
            str(job.url),
            statuses={ApplicationStatus.SUBMITTED, ApplicationStatus.ALREADY_APPLIED},
        ):
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
        result = await applicator.apply(job, cover_letter, submit=submit)
    if submit and result.status == ApplicationStatus.SUBMITTED:
        state.record(result)  # ApplicationState is the authority for "applied"
    return result


async def ats_check(
    settings: AppSettings, tailored_resume_path: str = ""
) -> ATSCompatibilityResult:
    """Run the ATS-compatibility check on the relevant résumé — the tailored artifact when
    ``tailored_resume_path`` points at one, else the configured résumé. Offline (résumé
    parse + heuristic scoring, run off the event loop); touches no account.
    """
    from job_applicator.documents.ats_checker import ATSChecker
    from job_applicator.documents.resume import ResumeLoader

    def _run() -> ATSCompatibilityResult:
        loader = ResumeLoader()
        resume = None
        if tailored_resume_path and Path(tailored_resume_path).exists():
            try:
                text = Path(tailored_resume_path).read_text(encoding="utf-8")
            except (OSError, ValueError):  # unreadable / non-UTF-8 → use the configured one
                text = ""
            if text.strip():
                resume = loader.parse_text(text)
        if (
            resume is None
        ):  # no usable tailored text → the configured résumé (load enforces non-empty)
            resume = loader.load(settings.resume_path)
        return ATSChecker().check(resume)

    return await asyncio.to_thread(_run)
