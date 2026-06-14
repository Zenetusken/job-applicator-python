"""CLI entry point — Typer + Rich for terminal UX."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from job_applicator import __version__
from job_applicator.config import AppSettings
from job_applicator.models import UserProfile
from job_applicator.utils.diff import render_diff
from job_applicator.utils.logging import setup_logging

if TYPE_CHECKING:
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.models import (
        ApplicationResult,
        CoverLetterResult,
        CoverLetterSession,
        JobListing,
        ResumeData,
        StyleGuide,
        TailoredResume,
    )

app = typer.Typer(
    name="job-applicator",
    help="Automated job application tool with AI-powered cover letters.",
    add_completion=False,
)
console = Console()

T = TypeVar("T")


async def _llm_with_retry(  # noqa: UP047 — mypy doesn't support PEP 695 yet
    console: Console,
    operation: Callable[[], Awaitable[T]],
    status_message: str = "Processing...",
    on_fail_choices: str = "[R] Retry or [Q] Quit",
) -> T | None:
    """Execute an async LLM operation with retry on failure.

    Returns the result on success, or None if the user chooses to quit.
    """
    while True:
        try:
            with console.status(status_message):
                return await operation()
        except Exception as exc:
            console.print(f"[red]LLM error: {exc}[/red]")
            choice = console.input(f"[bold cyan]{on_fail_choices}? [/bold cyan]").strip().upper()
            if choice == "Q":
                return None


def _resolve_ocr_mode(ocr_mode: str, force_ocr: bool) -> str:
    """Return effective OCR mode from CLI flags."""
    if force_ocr:
        return "on"
    return ocr_mode


def _run_ats_preflight(resume: ResumeData) -> None:
    """Run ATS compatibility check and warn if issues found."""
    from job_applicator.documents.ats_checker import ATSChecker

    checker = ATSChecker()
    result = checker.check(resume)

    if result.is_compatible:
        return

    console.print(f"\n[yellow]⚠ ATS Compatibility: {result.score:.0%} (Not Compatible)[/yellow]")
    for warning in result.warnings[:3]:
        console.print(f"  [yellow]![/yellow] {warning}")
    console.print(
        "  [dim]Tip: Run 'job-applicator ats-check --resume <path>' for full report[/dim]"
    )
    console.print()


def _run_ats_post_tailor(original_text: str, tailored_text: str) -> None:
    """Compare ATS compatibility before and after tailoring."""
    from job_applicator.documents.ats_checker import ATSChecker
    from job_applicator.documents.resume import ResumeLoader

    checker = ATSChecker()
    loader = ResumeLoader()

    original = loader.parse_text(original_text)
    tailored = loader.parse_text(tailored_text)

    original_result = checker.check(original)
    tailored_result = checker.check(tailored)

    before = original_result.score
    after = tailored_result.score

    if after >= before:
        console.print(
            f"\n[green]ATS Compatibility (before → after): {before:.0%} → {after:.0%} ✓[/green]"
        )
        if after >= 0.6:
            console.print("  [green]✓ All checks passing after tailoring[/green]")
    else:
        console.print(
            f"\n[yellow]⚠ ATS Compatibility (before → after): {before:.0%} → {after:.0%}[/yellow]"
        )
        original_checks = {c["name"]: c["passed"] for c in original_result.checks}
        for check in tailored_result.checks:
            if not check["passed"] and original_checks.get(check["name"], False):
                console.print(f"  [yellow]![/yellow] New issue: {check['details']}")
    console.print()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"job-applicator v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Job Applicator — automated job applications with AI cover letters."""


@app.command()
def search(
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board to search."),
    query: str = typer.Option(..., "--query", "-q", help="Search query."),
    location: str = typer.Option("", "--location", "-l", help="Location filter."),
    remote: bool = typer.Option(False, "--remote", "-r", help="Remote jobs only."),
    max_results: int = typer.Option(25, "--max", "-n", help="Max results."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
) -> None:
    """Search for jobs on a job board."""
    settings = _get_settings(headed)
    setup_logging(settings.log_level)

    async def _run() -> None:
        from job_applicator.browser.manager import BrowserManager
        from job_applicator.models import JobBoard
        from job_applicator.scrapers.base import SearchParams

        board_map = {"linkedin": JobBoard.LINKEDIN, "indeed": JobBoard.INDEED}
        if site not in board_map:
            console.print(f"[red]Unsupported site: {site}[/red]")
            raise typer.Exit(1)

        params = SearchParams(
            query=query,
            location=location,
            remote_only=remote,
            max_results=max_results,
            board=board_map[site],
        )

        async with BrowserManager(settings.browser) as browser:
            if site == "linkedin":
                from job_applicator.scrapers.linkedin import LinkedInScraper

                scraper = LinkedInScraper(browser, settings)
            else:
                console.print(f"[yellow]{site} scraper not yet implemented[/yellow]")
                raise typer.Exit(1)

            with console.status(f"Searching {site} for '{query}'..."):
                jobs = await scraper.scrape(params)

        if not jobs:
            if as_json:
                console.print("[]")
            else:
                console.print("[yellow]No jobs found.[/yellow]")
            return

        if as_json:
            import json

            output = [
                {
                    "title": j.title,
                    "company": j.company,
                    "location": j.location,
                    "url": str(j.url),
                    "description": j.description[:200],
                    "requirements": j.requirements,
                }
                for j in jobs
            ]
            console.print(json.dumps(output, indent=2))
            return

        table = Table(title=f"Found {len(jobs)} jobs")
        table.add_column("Title", style="cyan")
        table.add_column("Company", style="green")
        table.add_column("Location")
        table.add_column("URL", style="blue")

        for job in jobs:
            table.add_row(job.title, job.company, job.location, str(job.url))

        console.print(table)

    asyncio.run(_run())


@app.command()
def apply(
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board."),
    query: str = typer.Option("", "--query", "-q", help="Search query (empty = use saved list)."),
    limit: int = typer.Option(5, "--limit", "-n", help="Max applications."),
    cover_letter: bool = typer.Option(True, "--cover-letter/--no-cover-letter", help="AI cover."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    style_guide: str = typer.Option("", "--style-guide", help="Example to mimic style."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: str = typer.Option(
        "auto",
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
) -> None:
    """Auto-apply to jobs with optional AI cover letters."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    async def _run() -> None:
        from job_applicator.browser.manager import BrowserManager
        from job_applicator.models import JobBoard
        from job_applicator.scrapers.base import SearchParams

        if site == "linkedin":
            from job_applicator.applicators.linkedin import LinkedInApplicator
            from job_applicator.scrapers.linkedin import LinkedInScraper
        else:
            console.print(f"[yellow]{site} not yet implemented[/yellow]")
            raise typer.Exit(1)

        async with BrowserManager(settings.browser) as browser:
            # Search for jobs
            if query:
                scraper = LinkedInScraper(browser, settings) if site == "linkedin" else None
                if not scraper:
                    console.print(f"[yellow]{site} not yet implemented[/yellow]")
                    raise typer.Exit(1)

                params = SearchParams(
                    query=query,
                    max_results=limit,
                    board=JobBoard.LINKEDIN,
                )
                with console.status(f"Searching {site}..."):
                    jobs = await scraper.scrape(params)
            else:
                console.print("[yellow]No query provided. Use --query to search.[/yellow]")
                raise typer.Exit(1)

            if not jobs:
                console.print("[yellow]No jobs found to apply to.[/yellow]")
                return

            # Generate cover letters if requested
            cover_letters: dict[str, str] = {}
            if cover_letter and settings.resume_path:
                from job_applicator.documents.cover_letter import CoverLetterGenerator
                from job_applicator.documents.resume import ResumeLoader

                loader = ResumeLoader()
                resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
                _run_ats_preflight(resume_data)

                user_profile = _load_user_profile(settings)
                generator = CoverLetterGenerator(settings.llm)

                sem = asyncio.Semaphore(3)

                async def _gen_one(
                    job: JobListing,
                ) -> tuple[str, str] | None:
                    async with sem:
                        try:
                            letter = await generator.generate(job, user_profile, resume_data)
                            return str(job.url), letter
                        except Exception as exc:
                            msg = f"Cover letter failed for {job.title}: {exc}"
                            console.print(f"[yellow]{msg}[/yellow]")
                            return None

                with console.status("Generating cover letters (parallel)..."):
                    results_cl = await asyncio.gather(*(_gen_one(j) for j in jobs[:limit]))
                    for entry in results_cl:
                        if entry is not None:
                            url, letter = entry
                            cover_letters[url] = letter

            # Apply to jobs
            applicator = LinkedInApplicator(browser, settings) if site == "linkedin" else None
            if not applicator:
                console.print(f"[yellow]{site} applicator not yet implemented[/yellow]")
                raise typer.Exit(1)

            app_results: list[ApplicationResult] = []
            for job in jobs[:limit]:
                with console.status(f"Applying to {job.title} at {job.company}..."):
                    job_letter = cover_letters.get(str(job.url))
                    ar: ApplicationResult = await applicator.apply(job, job_letter)
                    app_results.append(ar)

            # Display results
            if as_json:
                import json

                output = [
                    {
                        "job": r.job.title,
                        "company": r.job.company,
                        "status": r.status.value,
                        "error": r.error_message,
                        "notes": r.notes,
                    }
                    for r in app_results
                ]
                console.print(json.dumps(output, indent=2))
            else:
                table = Table(title="Application Results")
                table.add_column("Job", style="cyan")
                table.add_column("Company", style="green")
                table.add_column("Status")
                table.add_column("Notes")

                for r in app_results:
                    status_style = {
                        "submitted": "green",
                        "failed": "red",
                        "skipped": "yellow",
                        "pending": "blue",
                    }.get(r.status.value, "white")
                    table.add_row(
                        r.job.title,
                        r.job.company,
                        f"[{status_style}]{r.status.value}[/{status_style}]",
                        r.error_message or r.notes or "",
                    )

                console.print(table)
                submitted = sum(1 for r in app_results if r.status.value == "submitted")
                failed = sum(1 for r in app_results if r.status.value == "failed")
                skipped = sum(1 for r in app_results if r.status.value == "skipped")
                console.print(
                    f"\n[green]{submitted}[/green] submitted, "
                    f"[red]{failed}[/red] failed, "
                    f"[yellow]{skipped}[/yellow] skipped"
                )

    asyncio.run(_run())


@app.command()
def generate_cover_letter(
    job_title: str = typer.Option(..., "--job-title", "-t", help="Job title."),
    company: str = typer.Option(..., "--company", "-c", help="Company name."),
    job_description: str = typer.Option("", "--description", "-d", help="Job description."),
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    style_guide: str = typer.Option("", "--style-guide", help="Style examples."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    ocr_mode: str = typer.Option(
        "auto",
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
) -> None:
    """Generate an AI cover letter for a specific job."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    async def _run() -> None:
        from pydantic import HttpUrl

        from job_applicator.documents.cover_letter import CoverLetterGenerator
        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume or set RESUME_PATH.[/red]")
            raise typer.Exit(1)

        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
        user_profile = _load_user_profile(settings)

        job = JobListing(
            title=job_title,
            company=company,
            description=job_description,
            url=HttpUrl("https://example.com/placeholder"),
            board=JobBoard.LINKEDIN,
        )

        generator = CoverLetterGenerator(settings.llm)

        # Load style guide if provided (supports comma-separated paths)
        style = None
        if settings.style_guide_path:
            paths = [p.strip() for p in settings.style_guide_path.split(",") if p.strip()]

            if len(paths) == 1:
                with console.status("Analyzing writing style..."):
                    style = await generator.load_style_guide(paths[0])
                console.print(f"[green]Style loaded: {style.tone}[/green]")
            elif len(paths) > 1:
                with console.status(f"Analyzing {len(paths)} style examples..."):
                    from job_applicator.documents.style_analyzer import StyleAnalyzer

                    analyzer = StyleAnalyzer(settings.llm)

                    texts = []
                    for path in paths:
                        from pathlib import Path

                        p = Path(path)
                        if await asyncio.to_thread(p.exists):
                            if p.suffix.lower() == ".pdf":
                                resume = loader.load(p, ocr_mode=effective_ocr_mode)
                                texts.append(resume.raw_text)
                            else:
                                texts.append(await asyncio.to_thread(p.read_text, encoding="utf-8"))

                    if texts:
                        style = await analyzer.analyze_multiple(texts)
                        msg = f"Combined style from {len(texts)} examples"
                        console.print(f"[green]{msg}: {style.tone}[/green]")

        with console.status("Generating cover letter..."):
            letter = await generator.generate(job, user_profile, resume_data, style)

        console.print("\n[bold]Generated Cover Letter:[/bold]\n")
        console.print(letter)

    asyncio.run(_run())


@app.command()
def match(
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    jobs_file: str = typer.Option("", "--jobs-file", help="JSON file with job listings."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top matches."),
    min_score: float = typer.Option(0.0, "--min-score", help="Minimum match score (0.0-1.0)."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: str = typer.Option(
        "auto",
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
) -> None:
    """Match resume to job listings using semantic embeddings."""
    settings = _get_settings()
    if resume_path:
        settings.resume_path = resume_path
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    async def _run() -> None:
        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.embeddings.matching import JobMatcher
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        # Load resume
        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
        if not as_json:
            console.print(f"[green]Loaded resume: {resume_data.name}[/green]")
            _run_ats_preflight(resume_data)

        # Load jobs
        jobs: list[JobListing] = []
        if jobs_file:
            import json

            with open(jobs_file) as f:  # noqa: ASYNC230
                data = json.load(f)
                for item in data:
                    jobs.append(JobListing(**item))
        else:
            # Example jobs for demo
            from pydantic import HttpUrl

            jobs = [
                JobListing(
                    title="Python Developer",
                    company="TechCorp",
                    url=HttpUrl("https://example.com/1"),
                    description="Looking for Python developer with FastAPI experience",
                    requirements=["Python", "FastAPI", "PostgreSQL"],
                    board=JobBoard.LINKEDIN,
                ),
                JobListing(
                    title="Backend Engineer",
                    company="StartupXYZ",
                    url=HttpUrl("https://example.com/2"),
                    description="Backend engineer for microservices",
                    requirements=["Python", "Docker", "AWS"],
                    board=JobBoard.LINKEDIN,
                ),
            ]

        if not as_json:
            console.print(f"[green]Loaded {len(jobs)} jobs[/green]")

        # Match
        with console.status("Computing embeddings and matching..."):
            matcher = JobMatcher(settings.embedding)
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)

        # Filter by min score
        if min_score > 0:
            matches = [m for m in matches if m.score >= min_score]

        # JSON output
        if as_json:
            import json

            output = [
                {
                    "rank": i + 1,
                    "score": round(m.score, 4),
                    "title": m.job.title,
                    "company": m.job.company,
                    "url": str(m.job.url),
                    "matched_skills": m.matched_skills,
                    "missing_skills": m.missing_skills,
                    "summary": m.summary,
                }
                for i, m in enumerate(matches)
            ]
            console.print(json.dumps(output, indent=2))
            return

        # Display results
        table = Table(title=f"Top {len(matches)} Job Matches")
        table.add_column("Rank", style="dim")
        table.add_column("Score", style="cyan")
        table.add_column("Job", style="green")
        table.add_column("Company")
        table.add_column("Matched Skills")
        table.add_column("Missing Skills")

        for i, match in enumerate(matches, 1):
            if match.score >= 0.7:
                score_style = "green"
            elif match.score >= 0.5:
                score_style = "yellow"
            else:
                score_style = "red"
            table.add_row(
                str(i),
                f"[{score_style}]{match.score:.0%}[/{score_style}]",
                match.job.title,
                match.job.company,
                ", ".join(match.matched_skills[:3]) or "-",
                ", ".join(match.missing_skills[:3]) or "-",
            )

        console.print(table)

    asyncio.run(_run())


@app.command()
def batch(
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    jobs_file: str = typer.Option("", "--jobs-file", help="JSON file with job listings."),
    query: str = typer.Option(
        "", "--query", "-q", help="Search query (alternative to --jobs-file)."
    ),
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board for --query."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Max jobs to tailor."),
    min_score: float = typer.Option(0.0, "--min-score", help="Skip jobs below this score."),
    cover_letter: bool = typer.Option(
        True, "--cover-letter/--no-cover-letter", help="Generate cover letters."
    ),
    style_guide: str = typer.Option("", "--style-guide", help="Style example file."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: str = typer.Option(
        "auto",
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
) -> None:
    """Batch tailor resumes (and optionally cover letters) for multiple jobs."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    async def _run() -> None:
        import json
        from datetime import datetime as dt
        from pathlib import Path

        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.documents.resume_tailor import ResumeTailor
        from job_applicator.embeddings.matching import JobMatcher, MatchResult
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
        if not as_json:
            console.print(f"[green]Loaded resume: {resume_data.name}[/green]")
        _run_ats_preflight(resume_data)

        jobs: list[JobListing] = []
        if jobs_file:
            try:
                with open(jobs_file) as f:  # noqa: ASYNC230
                    data = json.load(f)
                    for item in data:
                        jobs.append(JobListing(**item))
            except FileNotFoundError:
                console.print(f"[red]Jobs file not found: {jobs_file}[/red]")
                raise typer.Exit(1) from None
            except Exception as exc:
                console.print(f"[red]Error reading jobs file: {exc}[/red]")
                raise typer.Exit(1) from exc
        elif query:
            from job_applicator.browser.manager import BrowserManager
            from job_applicator.scrapers.base import SearchParams

            if site == "linkedin":
                from job_applicator.scrapers.linkedin import LinkedInScraper
            else:
                console.print(f"[yellow]{site} not yet implemented[/yellow]")
                raise typer.Exit(1)

            async with BrowserManager(settings.browser) as browser:
                scraper = LinkedInScraper(browser, settings)
                params = SearchParams(query=query, max_results=top_k * 2, board=JobBoard.LINKEDIN)
                with console.status(f"Searching {site}..."):
                    jobs = await scraper.scrape(params)
        else:
            console.print("[red]Provide --jobs-file or --query.[/red]")
            raise typer.Exit(1)

        if not jobs:
            console.print("[yellow]No jobs found.[/yellow]")
            return

        if not as_json:
            console.print(f"[green]Loaded {len(jobs)} jobs[/green]")

        matcher = JobMatcher(settings.embedding)
        with console.status("Computing match scores..."):
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)

        if min_score > 0:
            before = len(matches)
            matches = [m for m in matches if m.score >= min_score]
            skipped = before - len(matches)
            if skipped and not as_json:
                console.print(
                    f"[yellow]Skipped {skipped} jobs below {min_score:.0%} threshold[/yellow]"
                )

        if not matches:
            console.print("[yellow]No jobs above minimum score threshold.[/yellow]")
            return

        if not as_json:
            console.print(f"[cyan]Tailoring {len(matches)} jobs...[/cyan]")

        style = None
        cl_generator = None
        if settings.style_guide_path:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            cl_generator = CoverLetterGenerator(settings.llm)
            with console.status("Loading style guide..."):
                style = await cl_generator.load_style_guide(settings.style_guide_path)
        elif cover_letter:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            cl_generator = CoverLetterGenerator(settings.llm)

        tailor_engine = ResumeTailor(settings.llm)
        user_profile = _load_user_profile(settings)
        sem = asyncio.Semaphore(3)
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.output_dir
        await asyncio.to_thread(Path(output_dir).mkdir, parents=True, exist_ok=True)

        async def _process_one(match_result: MatchResult) -> dict[str, object]:
            job = match_result.job
            safe_company = "".join(c if c.isalnum() or c in "-_" else "_" for c in job.company)[:30]
            safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in job.title)[:30]

            async with sem:
                result: dict[str, object] = {
                    "title": job.title,
                    "company": job.company,
                    "url": str(job.url),
                }

                try:
                    tailored = await tailor_engine.tailor(
                        resume=resume_data,
                        job=job,
                        user_instructions="",
                        style_guide=style,
                        matcher=matcher,
                    )
                    _run_ats_post_tailor(resume_data.raw_text, tailored.tailored_text)
                    result["match_score"] = round(tailored.match_score, 4)
                    result["semantic_score"] = round(tailored.semantic_score, 4)
                    result["skill_score"] = round(tailored.skill_score, 4)

                    resume_filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
                    resume_path_out = str(Path(output_dir) / resume_filename)
                    await asyncio.to_thread(
                        Path(resume_path_out).write_text, tailored.tailored_text
                    )

                    meta_filename = f"{resume_filename.rsplit('.', 1)[0]}.meta.json"
                    meta_path = str(Path(output_dir) / meta_filename)
                    tailored.output_path = resume_path_out
                    await asyncio.to_thread(
                        Path(meta_path).write_text, tailored.model_dump_json(indent=2)
                    )
                    result["resume_path"] = resume_path_out
                    result["tailored"] = True
                except Exception as exc:
                    result["tailored"] = False
                    result["error"] = str(exc)
                    return result

                if cl_generator is not None:
                    try:
                        letter = await cl_generator.generate(
                            job,
                            user_profile,
                            resume_data,
                            style_guide=style,
                            tailored_resume_text=tailored.tailored_text,
                        )
                        cl_filename = f"cover_letter_{safe_company}_{safe_title}_{timestamp}.txt"
                        cl_path = str(Path(output_dir) / cl_filename)
                        await asyncio.to_thread(Path(cl_path).write_text, letter)
                        result["cover_letter_path"] = cl_path
                        result["cover_letter"] = True
                    except Exception as exc:
                        result["cover_letter"] = False
                        result["cl_error"] = str(exc)

                return result

        with console.status("Processing jobs in parallel..."):
            batch_results = await asyncio.gather(*(_process_one(m) for m in matches))

        summary = {
            "timestamp": timestamp,
            "resume": settings.resume_path,
            "total_jobs": len(jobs),
            "matched": len(matches),
            "results": list(batch_results),
        }
        summary_path = str(Path(output_dir) / f"batch_summary_{timestamp}.json")
        await asyncio.to_thread(Path(summary_path).write_text, json.dumps(summary, indent=2))

        if as_json:
            console.print(json.dumps(summary, indent=2))
        else:
            table = Table(title="Batch Results")
            table.add_column("Job", style="cyan")
            table.add_column("Company", style="green")
            table.add_column("Score")
            table.add_column("Tailored")
            table.add_column("Cover Letter")
            table.add_column("Notes")

            for r in batch_results:
                score_raw = r.get("match_score", 0)
                score_val = float(score_raw) if score_raw else 0.0  # type: ignore[arg-type]
                score_style = (
                    "green" if score_val >= 0.7 else "yellow" if score_val >= 0.5 else "red"
                )
                score_str = (
                    f"[{score_style}]{score_val:.0%}[/{score_style}]"
                    if r.get("tailored")
                    else "[dim]N/A[/dim]"
                )
                table.add_row(
                    str(r.get("title", "")),
                    str(r.get("company", "")),
                    score_str,
                    "✓" if r.get("tailored") else "✗",
                    "✓" if r.get("cover_letter") else ("✗" if cover_letter else "-"),
                    str(r.get("error", r.get("cl_error", ""))),
                )

            console.print(table)
            tailored_ok = sum(1 for r in batch_results if r.get("tailored"))
            cl_ok = sum(1 for r in batch_results if r.get("cover_letter"))
            console.print(
                f"\n[green]{tailored_ok}[/green] tailored, [green]{cl_ok}[/green] cover letters"
            )
            console.print(f"Summary: {summary_path}")

    asyncio.run(_run())


async def _generate_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    resume_data: ResumeData,
    style: StyleGuide | None,
    tone_section: str,
    tailored_resume_text: str,
    session: CoverLetterSession,
    attempt: int = 1,
) -> CoverLetterResult | None:
    """Generate a cover letter via LLM. Returns None on failure."""
    from job_applicator.documents.cover_letter import CoverLetterGenerator

    generator = CoverLetterGenerator(settings.llm)
    try:
        with console.status("Generating cover letter..."):
            letter = await generator.generate(
                job,
                _load_user_profile(settings),
                resume_data,
                style_guide=style,
                tone_section=tone_section,
                tailored_resume_text=tailored_resume_text,
            )
        result = CoverLetterResult(
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            cover_letter_text=letter,
            attempt=attempt,
            prompt_version="1.0",
        )
        session.add_attempt(result)
        return result
    except Exception as exc:
        console.print(f"[red]LLM error: {exc}[/red]")
        return None


async def _save_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    result: CoverLetterResult,
) -> Path:
    """Save cover letter to disk and return the path."""
    from datetime import datetime as dt

    output_dir = Path(settings.output_dir)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    safe_company = job.company.replace(" ", "_").replace("/", "_")
    safe_title = job.title.replace(" ", "_").replace("/", "_")
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    cl_filename = f"cover_letter_{safe_company}_{safe_title}_{timestamp}.txt"
    cl_path = output_dir / cl_filename
    await asyncio.to_thread(cl_path.write_text, result.cover_letter_text, encoding="utf-8")
    result.output_path = str(cl_path)
    cl_meta_path = cl_path.with_suffix(".meta.json")
    await asyncio.to_thread(
        cl_meta_path.write_text, result.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(f"\n[green]Cover letter saved: {cl_path}[/green]")
    return cl_path


async def _refine_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    result: CoverLetterResult,
    user_instructions: str,
    session: CoverLetterSession,
    attempt: int,
    resume_data: ResumeData | None = None,
    style: StyleGuide | None = None,
    tone_section: str = "",
) -> bool:
    """Refine a cover letter with user instructions via LLM."""
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.models import CoverLetterResult as CLResult
    from job_applicator.models import ResumeData

    try:
        generator = CoverLetterGenerator(settings.llm)
        with console.status("Refining cover letter..."):
            refined = await generator.refine(
                job=job,
                resume=resume_data or ResumeData(raw_text=""),
                current_text=result.cover_letter_text,
                user_feedback=user_instructions,
                style_guide=style,
                tone_section=tone_section,
            )
        new_result = CLResult(
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            cover_letter_text=refined,
            user_modifications=user_instructions,
            attempt=attempt + 1,
        )
        session.add_attempt(new_result)
        return True
    except Exception as exc:
        console.print(f"[red]LLM error: {exc}[/red]")
        return False


async def _cover_letter_workflow(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    resume_data: ResumeData,
    style: StyleGuide | None,
    tone_profile: ToneProfile | None,
    tailored_resume_text: str,
) -> Path | None:
    """Generate and save a cover letter with accept/retry workflow.

    Returns the Path to the saved cover letter, or None if skipped.
    """
    from job_applicator.models import CoverLetterSession

    tone_section = ""
    if tone_profile:
        from job_applicator.documents.tone_detector import ToneDetector

        tone_section = ToneDetector().format_for_prompt(tone_profile)

    session = CoverLetterSession(job_title=job.title, job_company=job.company)
    attempt = 0

    result = await _generate_cover_letter(
        console, settings, job, resume_data, style, tone_section, tailored_resume_text, session
    )
    if result is None:
        retry = console.input("[bold cyan][R] Retry or [Q] Skip? [/bold cyan]").strip().upper()
        if retry == "R":
            result = await _generate_cover_letter(
                console,
                settings,
                job,
                resume_data,
                style,
                tone_section,
                tailored_resume_text,
                session,
            )
            if result is None:
                console.print("[red]Cover letter generation failed. Skipping.[/red]")
                return None
        else:
            return None

    while True:
        attempt += 1
        if attempt > 10:
            console.print("[red]Maximum retry limit (10) reached.[/red]")
            break
        if attempt >= 8:
            console.print("[yellow]Warning: approaching retry limit (10 max).[/yellow]")

        result = session.current
        console.print(f"\n[bold blue]--- Cover Letter Attempt #{attempt} ---[/bold blue]")

        console.print("\n[bold]Cover Letter Preview:[/bold]\n")
        console.print(Panel(result.cover_letter_text, title="Cover Letter", border_style="green"))

        if len(session.attempts) > 1:
            render_diff(
                console,
                session.attempts[0].cover_letter_text,
                result.cover_letter_text,
                max_lines=30,
            )

        console.print("\n[bold]What would you like to do?[/bold]")
        action_table = Table(show_header=False, box=None)
        action_table.add_column("Option", style="cyan bold")
        action_table.add_column("Description")
        action_table.add_row("[A] Accept", "Save this cover letter")
        action_table.add_row("[R] Retry", "Regenerate")
        action_table.add_row("[I] Input", "Give custom instructions")
        action_table.add_row("[D] Diff", "Show full diff from first attempt")
        action_table.add_row("[V] History", "Browse previous attempts")
        action_table.add_row("[Q] Skip", "Discard (resume already saved)")
        console.print(action_table)

        choice = (
            console.input("\n[bold cyan]Your choice (A/R/I/D/V/Q): [/bold cyan]").strip().upper()
        )

        if choice == "A":
            return await _save_cover_letter(console, settings, job, result)

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            new_result = await _generate_cover_letter(
                console,
                settings,
                job,
                resume_data,
                style,
                tone_section,
                tailored_resume_text,
                session,
                attempt=attempt + 1,
            )
            if new_result is None:
                console.print("[red]Generation failed. Please try again.[/red]")
            continue

        elif choice == "I":
            user_instructions = console.input(
                "\n[bold]Instructions (e.g., 'emphasize customer service'): [/bold]"
            ).strip()
            if not user_instructions:
                console.print("[yellow]No instructions provided.[/yellow]")
                continue
            refined_ok = await _refine_cover_letter(
                console,
                settings,
                job,
                result,
                user_instructions,
                session,
                attempt,
                resume_data,
                style,
                tone_section,
            )
            if not refined_ok:
                console.print("[red]Refinement failed. Please try again.[/red]")
                continue
            result = session.current
            continue

        elif choice == "D":
            if len(session.attempts) > 1:
                render_diff(
                    console,
                    session.attempts[0].cover_letter_text,
                    result.cover_letter_text,
                    max_lines=0,
                )
            else:
                console.print("[yellow]Only one attempt so far.[/yellow]")
            continue

        elif choice == "V":
            if len(session.attempts) < 2:
                console.print("[yellow]No previous attempts yet.[/yellow]")
                continue
            hist_table = Table(title="Cover Letter History")
            hist_table.add_column("#", style="dim")
            hist_table.add_column("Attempt")
            hist_table.add_column("Preview", style="dim")
            for i, att in enumerate(session.attempts):
                preview = att.cover_letter_text[:60].replace("\n", " ")
                marker = "\u2192" if i == session.current_index else " "
                hist_table.add_row(marker, str(att.attempt), preview + "...")
            console.print(hist_table)
            sel = console.input(
                "\n[bold cyan]Select attempt # (or Enter back): [/bold cyan]"
            ).strip()
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(session.attempts):
                    session.select(idx)
                    console.print(f"[green]Switched to attempt #{session.current.attempt}[/green]")
                else:
                    console.print("[red]Invalid attempt number.[/red]")
            continue

        elif choice == "Q":
            console.print("[yellow]Cover letter skipped. Resume already saved.[/yellow]")
            return None

        else:
            console.print("[red]Invalid choice. Please enter A, R, I, D, V, or Q.[/red]")

    return None


@app.command()
def tailor(
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    job_title: str = typer.Option(..., "--job-title", "-t", help="Job title."),
    company: str = typer.Option(..., "--company", "-c", help="Company name."),
    job_description: str = typer.Option("", "--description", "-d", help="Job description."),
    job_url: str = typer.Option("", "--url", help="Job posting URL."),
    requirements: str = typer.Option(
        "", "--requirements", "-r", help="Comma-separated requirements."
    ),
    location: str = typer.Option("", "--location", "-l", help="Job location."),
    style_guide: str = typer.Option("", "--style-guide", help="Style examples."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    min_score: float = typer.Option(
        0.0, "--min-score", help="Abort if match score is below this threshold (0.0-1.0)."
    ),
    ocr_mode: str = typer.Option(
        "auto",
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
) -> None:
    """Tailor a resume for a specific job with interactive preview."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    async def _run() -> None:
        from pydantic import HttpUrl

        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.documents.resume_tailor import ResumeTailor
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)
        console.print(f"[green]Loaded resume: {resume_data.name}[/green]")
        _run_ats_preflight(resume_data)

        req_list = [r.strip() for r in requirements.split(",") if r.strip()] if requirements else []
        url = HttpUrl(job_url) if job_url else HttpUrl("https://example.com/placeholder")

        job = JobListing(
            title=job_title,
            company=company,
            description=job_description,
            url=url,
            requirements=req_list,
            location=location,
            board=JobBoard.INDEED,
        )

        from job_applicator.documents.tone_detector import ToneDetector

        tone_detector = ToneDetector()
        tone_profile = tone_detector.detect(
            title=job.title,
            description=job.description,
            requirements=job.requirements,
        )
        console.print(
            f"[dim]Detected tone: {tone_profile.primary} "
            f"(confidence: {tone_profile.confidence:.0%})[/dim]"
        )

        style = None
        if settings.style_guide_path:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            generator = CoverLetterGenerator(settings.llm)
            with console.status("Analyzing writing style..."):
                style = await generator.load_style_guide(settings.style_guide_path)
            console.print(f"[green]Style loaded: {style.tone}[/green]")

        tailor_engine = ResumeTailor(settings.llm)
        attempt = 0
        user_instructions = ""

        from job_applicator.models import TailorSession

        session = TailorSession(
            original_text=resume_data.raw_text,
            job_title=job.title,
            job_company=job.company,
        )

        # Pre-ingestion date audit
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        validator = ResumeDateValidator()
        audit = validator.audit(resume_data)

        console.print("\n[bold]📋 CV Date Audit[/bold]")
        audit_table = Table(title="Date Analysis", show_lines=True)
        audit_table.add_column("Section", style="dim")
        audit_table.add_column("Entry", style="bold")
        audit_table.add_column("Start")
        audit_table.add_column("End")
        for entry in audit.entries:
            audit_table.add_row(
                entry.section,
                entry.label,
                entry.start,
                entry.end,
            )
        console.print(audit_table)

        console.print(f"\n[dim]Date range: {audit.earliest_date} → {audit.latest_date}[/dim]")

        if audit.warnings:
            console.print("\n[bold yellow]⚠ Warnings:[/bold yellow]")
            for w in audit.warnings:
                console.print(f"  [yellow]• {w}[/yellow]")

        if audit.staleness_issues:
            console.print("\n[bold red]⚠ Staleness Warnings:[/bold red]")
            for s in audit.staleness_issues:
                console.print(f"  [red]• {s}[/red]")

        if audit.ordering_issues:
            console.print("\n[bold red]⚠ Ordering Issues:[/bold red]")
            for o in audit.ordering_issues:
                console.print(f"  [red]• {o}[/red]")

        if audit.is_stale or audit.ordering_issues:
            console.print(
                "\n[bold yellow]This CV may be outdated or have ordering "
                "issues. Please verify your CV is up to date before "
                "proceeding.[/bold yellow]"
            )
            confirm = (
                console.input("\n[bold cyan]Proceed anyway? (y/n): [/bold cyan]").strip().lower()
            )
            if confirm != "y":
                console.print("[yellow]Aborted. Please update your CV.[/yellow]")
                raise typer.Exit(0)
        else:
            console.print("[green]✓ Dates look coherent and current.[/green]")

        # Pre-tailor match score check
        if min_score > 0:
            from job_applicator.embeddings.matching import JobMatcher

            with console.status("Computing match score..."):
                matcher = JobMatcher(settings.embedding)
                pre_match = matcher.match_resume_to_job(resume_data, job)
            console.print(
                f"[cyan]Match score: {pre_match.score:.0%} (threshold: {min_score:.0%})[/cyan]"
            )
            if pre_match.score < min_score:
                console.print(
                    f"[red]Match score {pre_match.score:.0%} is below threshold "
                    f"{min_score:.0%}. Aborting.[/red]"
                )
                raise typer.Exit(0)

        try:
            with console.status("Tailoring resume..."):
                result = await tailor_engine.tailor(
                    resume_data, job, user_instructions, style, tone_profile
                )
            session.add_attempt(result)
            _run_ats_post_tailor(resume_data.raw_text, result.tailored_text)
        except Exception as exc:
            console.print(f"[red]LLM error: {exc}[/red]")
            console.print("[yellow]Could not generate tailored resume.[/yellow]")
            raise typer.Exit(1) from exc

        while True:
            attempt += 1
            if attempt > 10:
                console.print("[red]Maximum retry limit (10) reached.[/red]")
                break
            if attempt >= 8:
                console.print("[yellow]Warning: approaching retry limit (10 max).[/yellow]")

            console.print(f"\n[bold blue]--- Attempt #{attempt} ---[/bold blue]")

            console.print("\n[bold]Tailored Resume Preview:[/bold]\n")
            console.print(
                Panel(
                    result.tailored_text,
                    title="Tailored Resume",
                    border_style="cyan",
                )
            )
            render_diff(console, session.original_text, result.tailored_text, max_lines=30)

            console.print("\n[bold]Metadata:[/bold]")
            meta_table = Table(show_header=False, box=None)
            meta_table.add_column("Key", style="dim")
            meta_table.add_column("Value")
            meta_table.add_row("Job", f"{job.title} at {job.company}")
            meta_table.add_row("Match Score", f"{result.match_score:.0%}")
            meta_table.add_row(
                "Matched Skills",
                ", ".join(result.matched_skills[:5]) or "—",
            )
            meta_table.add_row(
                "Missing Skills",
                ", ".join(result.missing_skills[:5]) or "—",
            )
            meta_table.add_row("Attempt", str(attempt))
            if result.user_modifications:
                meta_table.add_row("User Input", result.user_modifications)
            console.print(meta_table)

            console.print("\n[bold]Changes Made:[/bold]")
            console.print(result.changes_summary)

            console.print("\n[bold]What would you like to do?[/bold]")
            action_table = Table(show_header=False, box=None)
            action_table.add_column("Option", style="cyan bold")
            action_table.add_column("Description")
            action_table.add_row("[A] Accept", "Save this version as final")
            action_table.add_row("[R] Retry", "Regenerate with same instructions")
            action_table.add_row("[I] Input", "Give custom instructions to refine")
            action_table.add_row("[D] Diff", "Show changes from original resume")
            action_table.add_row("[V] History", "Browse previous attempts")
            action_table.add_row("[S] Section", "Edit a specific section")
            action_table.add_row("[Q] Quit", "Discard and exit")
            console.print(action_table)

            choice = (
                console.input("\n[bold cyan]Your choice (A/R/I/D/V/S/Q): [/bold cyan]")
                .strip()
                .upper()
            )

            if choice == "A":
                from datetime import datetime as dt

                output_dir = Path(settings.output_dir)
                await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

                safe_company = job.company.replace(" ", "_").replace("/", "_")
                safe_title = job.title.replace(" ", "_").replace("/", "_")
                timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
                output_path = output_dir / filename

                await asyncio.to_thread(
                    output_path.write_text, result.tailored_text, encoding="utf-8"
                )
                result.output_path = str(output_path)

                console.print(f"\n[green]Tailored resume saved: {output_path}[/green]")
                console.print(f"[dim]Attempt #{attempt} | Score: {result.match_score:.0%}[/dim]")

                # Offer cover letter generation
                cover_letter_path = None
                cl_choice = (
                    console.input(
                        f"\n[bold cyan]Generate a matching cover letter "
                        f"for {job.title} at {job.company}? (Y/N): [/bold cyan]"
                    )
                    .strip()
                    .upper()
                )

                if cl_choice == "Y":
                    cover_letter_path = await _cover_letter_workflow(
                        console,
                        settings,
                        job,
                        resume_data,
                        style,
                        tone_profile,
                        result.tailored_text,
                    )

                # Write resume meta.json (with or without cover_letter_path)
                if cover_letter_path:
                    result.cover_letter_path = str(cover_letter_path)
                meta_path = output_path.with_suffix(".meta.json")
                await asyncio.to_thread(
                    meta_path.write_text, result.model_dump_json(indent=2), encoding="utf-8"
                )
                console.print(f"[green]Metadata saved: {meta_path}[/green]")

                break

            elif choice == "R":
                console.print("[yellow]Regenerating...[/yellow]")
                user_instructions = ""
                refined: TailoredResume | None = await _llm_with_retry(
                    console,
                    partial(tailor_engine.refine, resume_data, result, "", job),
                    "Tailoring resume...",
                )
                if refined is None:
                    break
                result = refined
                result.attempt = attempt
                session.add_attempt(result)
                continue

            elif choice == "I":
                user_instructions = console.input(
                    "\n[bold]Enter your instructions (e.g., 'emphasize "
                    "customer service', 'add troubleshooting detail'): "
                    "[/bold]"
                ).strip()
                if not user_instructions:
                    console.print("[yellow]No instructions provided, retrying.[/yellow]")
                refined = await _llm_with_retry(
                    console,
                    partial(
                        tailor_engine.refine,
                        resume_data,
                        result,
                        user_instructions,
                        job,
                    ),
                    "Tailoring resume...",
                )
                if refined is None:
                    break
                result = refined
                result.attempt = attempt
                session.add_attempt(result)
                continue

            elif choice == "D":
                render_diff(console, session.original_text, result.tailored_text, max_lines=0)
                continue

            elif choice == "V":
                if len(session.attempts) < 2:
                    console.print("[yellow]No previous attempts yet.[/yellow]")
                    continue
                hist_table = Table(title="Version History")
                hist_table.add_column("#", style="dim")
                hist_table.add_column("Attempt")
                hist_table.add_column("Score", style="cyan")
                hist_table.add_column("Instructions")
                hist_table.add_column("Preview", style="dim")
                for i, att in enumerate(session.attempts):
                    preview = att.tailored_text[:60].replace("\n", " ")
                    marker = "\u2192" if i == session.current_index else " "
                    hist_table.add_row(
                        marker,
                        str(att.attempt),
                        f"{att.match_score:.0%}",
                        att.user_modifications or "\u2014",
                        preview + "...",
                    )
                console.print(hist_table)
                sel = console.input(
                    "\n[bold cyan]Select attempt # to view (or Enter to go back): [/bold cyan]"
                ).strip()
                if sel.isdigit():
                    idx = int(sel) - 1
                    if 0 <= idx < len(session.attempts):
                        session.select(idx)
                        result = session.current
                        console.print(f"[green]Switched to attempt #{result.attempt}[/green]")
                    else:
                        console.print("[red]Invalid attempt number.[/red]")
                continue

            elif choice == "S":
                from job_applicator.documents.resume_tailor import parse_sections

                sections = parse_sections(result.tailored_text)
                if len(sections) <= 1 and sections[0].name == "Full Document":
                    console.print(
                        "[yellow]Could not detect sections. "
                        "Use [I] for full-resume instructions.[/yellow]"
                    )
                    continue

                console.print("\n[bold]Sections:[/bold]")
                sec_table = Table(show_header=False, box=None)
                sec_table.add_column("#", style="cyan")
                sec_table.add_column("Section", style="bold")
                sec_table.add_column("Lines", style="dim")
                for i, sec in enumerate(sections, 1):
                    line_count = sec.text.count("\n") + 1
                    sec_table.add_row(str(i), sec.name, f"{line_count} lines")
                console.print(sec_table)

                sec_choice = console.input(
                    "\n[bold cyan]Section # to edit (or Enter to go back): [/bold cyan]"
                ).strip()
                if not sec_choice.isdigit():
                    continue
                sec_idx = int(sec_choice) - 1
                if sec_idx < 0 or sec_idx >= len(sections):
                    console.print("[red]Invalid section number.[/red]")
                    continue

                target_section = sections[sec_idx]
                console.print(f"\n[dim]Editing: {target_section.name}[/dim]")
                console.print(f"[dim]{target_section.text[:200]}...[/dim]\n")

                sec_instructions = console.input(
                    "[bold]Instructions for this section: [/bold]"
                ).strip()
                if not sec_instructions:
                    console.print("[yellow]No instructions provided.[/yellow]")
                    continue

                user_instructions = (
                    f"ONLY modify the {target_section.name} section. "
                    f"Keep all other sections unchanged.\n\n"
                    f"Current {target_section.name} content:\n{target_section.text}\n\n"
                    f"User instructions for this section: {sec_instructions}"
                )
                refined = await _llm_with_retry(
                    console,
                    partial(
                        tailor_engine.refine,
                        resume_data,
                        result,
                        user_instructions,
                        job,
                    ),
                    "Refining section...",
                )
                if refined is None:
                    break
                result = refined
                result.attempt = attempt
                session.add_attempt(result)
                continue

            elif choice == "Q":
                console.print("[yellow]Discarded. No changes saved.[/yellow]")
                break

            else:
                console.print("[red]Invalid choice. Please enter A, R, I, D, V, S, or Q.[/red]")

    asyncio.run(_run())


@app.command()
def ats_check(
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
    ocr_mode: str = typer.Option(
        "auto",
        "--ocr-mode",
        help="OCR mode: auto (fallback), on (always), off (never).",
    ),
    force_ocr: bool = typer.Option(
        False,
        "--force-ocr",
        help="Force OCR; equivalent to --ocr-mode on.",
    ),
) -> None:
    """Check resume ATS (Applicant Tracking System) compatibility."""
    settings = _get_settings()
    if resume_path:
        settings.resume_path = resume_path
    setup_logging(settings.log_level)
    effective_ocr_mode = _resolve_ocr_mode(ocr_mode, force_ocr)

    from job_applicator.documents.ats_checker import ATSChecker
    from job_applicator.documents.resume import ResumeLoader

    if not settings.resume_path:
        console.print("[red]Resume path required. Use --resume.[/red]")
        raise typer.Exit(1)

    loader = ResumeLoader()
    resume_data = loader.load(settings.resume_path, ocr_mode=effective_ocr_mode)

    if not as_json:
        console.print(f"[green]Loaded resume: {resume_data.name}[/green]")

    checker = ATSChecker()
    result = checker.check(resume_data)

    if as_json:
        import json

        output = {
            "score": result.score,
            "is_compatible": result.is_compatible,
            "checks": result.checks,
            "warnings": result.warnings,
            "suggestions": result.suggestions,
        }
        console.print(json.dumps(output, indent=2))
        return

    # Display results
    color = "green" if result.is_compatible else "red"
    console.print(f"\n[bold {color}]ATS Score: {result.score:.0%}[/bold {color}]")
    status = "Compatible" if result.is_compatible else "Not Compatible"
    console.print(f"[{color}]Status: {status}[/{color}]\n")

    # Check results table
    table = Table(title="ATS Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    for check in result.checks:
        status = "[green]PASS[/green]" if check["passed"] else "[red]FAIL[/red]"
        table.add_row(str(check["name"]), status, str(check["details"]))

    console.print(table)

    # Warnings
    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in result.warnings:
            console.print(f"  [yellow]![/yellow] {warning}")

    # Suggestions
    if result.suggestions:
        console.print("\n[bold cyan]Suggestions:[/bold cyan]")
        for suggestion in result.suggestions:
            console.print(f"  [cyan]*[/cyan] {suggestion}")


@app.command()
def config_init() -> None:
    """Create a sample config.toml file."""
    config_content = """# Job Applicator Configuration

# Profile
profile_name = "default"
resume_path = "/path/to/your/resume.pdf"
output_dir = "output"
log_level = "INFO"

# Browser
[browser]
headless = true
slow_mo = 0
timeout_ms = 30000

# LLM (for AI cover letters)
[llm]
api_base = "http://localhost:8000/v1"
api_key = "not-needed-for-local"
model = "Qwen/Qwen3-8B-AWQ"
max_tokens = 1024
temperature = 0.7

# Targets
[target]
max_applications_per_day = 20
delay_between_applications_s = 2.0
# linkedin_email = "your-email@example.com"
# linkedin_password = "your-password"
"""
    config_path = Path("config.toml")
    if config_path.exists():
        console.print("[yellow]config.toml already exists. Skipping.[/yellow]")
        return

    config_path.write_text(config_content)
    console.print("[green]Created config.toml[/green]")
    console.print("Edit it with your credentials, or set environment variables.")


def _get_settings(headed: bool = False) -> AppSettings:
    """Build AppSettings, overriding headless if --headed."""
    settings = AppSettings()
    if headed:
        settings.browser.headless = False
    return settings


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
