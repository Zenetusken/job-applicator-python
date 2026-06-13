"""CLI entry point — Typer + Rich for terminal UX."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from job_applicator import __version__
from job_applicator.config import AppSettings
from job_applicator.models import UserProfile
from job_applicator.utils.diff import render_diff as _render_diff
from job_applicator.utils.logging import setup_logging

if TYPE_CHECKING:
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.models import JobListing, ResumeData, StyleGuide

app = typer.Typer(
    name="job-applicator",
    help="Automated job application tool with AI-powered cover letters.",
    add_completion=False,
)
console = Console()


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
            console.print("[yellow]No jobs found.[/yellow]")
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
) -> None:
    """Auto-apply to jobs with optional AI cover letters."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)

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
                resume_data = loader.load(settings.resume_path)

                user_profile = _load_user_profile(settings)
                generator = CoverLetterGenerator(settings.llm)

                with console.status("Generating cover letters..."):
                    for job in jobs[:limit]:
                        try:
                            letter = await generator.generate(job, user_profile, resume_data)
                            cover_letters[str(job.url)] = letter
                        except Exception as exc:
                            msg = f"Cover letter failed for {job.title}: {exc}"
                            console.print(f"[yellow]{msg}[/yellow]")

            # Apply to jobs
            applicator = LinkedInApplicator(browser, settings) if site == "linkedin" else None
            if not applicator:
                console.print(f"[yellow]{site} applicator not yet implemented[/yellow]")
                raise typer.Exit(1)

            results = []
            for job in jobs[:limit]:
                with console.status(f"Applying to {job.title} at {job.company}..."):
                    job_letter = cover_letters.get(str(job.url))
                    result = await applicator.apply(job, job_letter)
                    results.append(result)

            # Display results
            table = Table(title="Application Results")
            table.add_column("Job", style="cyan")
            table.add_column("Company", style="green")
            table.add_column("Status")
            table.add_column("Notes")

            for r in results:
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
            submitted = sum(1 for r in results if r.status.value == "submitted")
            failed = sum(1 for r in results if r.status.value == "failed")
            skipped = sum(1 for r in results if r.status.value == "skipped")
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
) -> None:
    """Generate an AI cover letter for a specific job."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)

    async def _run() -> None:
        from pydantic import HttpUrl

        from job_applicator.documents.cover_letter import CoverLetterGenerator
        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume or set RESUME_PATH.[/red]")
            raise typer.Exit(1)

        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path)
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
                        if p.exists():
                            if p.suffix.lower() == ".pdf":
                                resume = loader.load(p)
                                texts.append(resume.raw_text)
                            else:
                                texts.append(p.read_text(encoding="utf-8"))

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
) -> None:
    """Match resume to job listings using semantic embeddings."""
    settings = _get_settings()
    if resume_path:
        settings.resume_path = resume_path
    setup_logging(settings.log_level)

    async def _run() -> None:
        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.embeddings.matching import JobMatcher
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        # Load resume
        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path)
        console.print(f"[green]Loaded resume: {resume_data.name}[/green]")

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

        console.print(f"[green]Loaded {len(jobs)} jobs[/green]")

        # Match
        with console.status("Computing embeddings and matching..."):
            matcher = JobMatcher(settings.embedding)
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)

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
    from job_applicator.documents.cover_letter import CoverLetterGenerator, strip_thinking_process
    from job_applicator.models import CoverLetterResult, CoverLetterSession

    generator = CoverLetterGenerator(settings.llm)
    tone_section = ""
    if tone_profile:
        from job_applicator.documents.tone_detector import ToneDetector

        tone_section = ToneDetector().format_for_prompt(tone_profile)

    session = CoverLetterSession(job_title=job.title, job_company=job.company)
    attempt = 0

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
            attempt=1,
        )
        session.add_attempt(result)
    except Exception as exc:
        console.print(f"[red]LLM error: {exc}[/red]")
        retry = console.input("[bold cyan][R] Retry or [Q] Skip? [/bold cyan]").strip().upper()
        if retry == "R":
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
                    attempt=1,
                )
                session.add_attempt(result)
            except Exception:
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
            from job_applicator.utils.diff import render_diff

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
            from datetime import datetime as dt

            output_dir = Path(settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_company = job.company.replace(" ", "_").replace("/", "_")
            safe_title = job.title.replace(" ", "_").replace("/", "_")
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            cl_filename = f"cover_letter_{safe_company}_{safe_title}_{timestamp}.txt"
            cl_path = output_dir / cl_filename
            cl_path.write_text(result.cover_letter_text, encoding="utf-8")
            result.output_path = str(cl_path)
            cl_meta_path = cl_path.with_suffix(".meta.json")
            cl_meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"\n[green]Cover letter saved: {cl_path}[/green]")
            return cl_path

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
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
                new_result = CoverLetterResult(
                    job_title=job.title,
                    job_company=job.company,
                    job_url=str(job.url),
                    cover_letter_text=letter,
                    attempt=attempt + 1,
                )
                session.add_attempt(new_result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
            continue

        elif choice == "I":
            user_instructions = console.input(
                "\n[bold]Instructions (e.g., 'emphasize customer service'): [/bold]"
            ).strip()
            if not user_instructions:
                console.print("[yellow]No instructions provided.[/yellow]")
            try:
                with console.status("Refining cover letter..."):
                    refine_prompt = (
                        f"User wants changes to this cover letter.\n\n"
                        f"Job: {job.title} at {job.company}\n\n"
                        f"Current cover letter:\n{result.cover_letter_text}\n\n"
                        f"User feedback: {user_instructions}\n\n"
                        f"Return the complete updated cover letter."
                    )
                    from litellm import acompletion

                    model = (
                        f"openai/{settings.llm.model}"
                        if settings.llm.api_base
                        else settings.llm.model
                    )
                    response = await acompletion(
                        model=model,
                        api_base=settings.llm.api_base,
                        api_key=settings.llm.api_key,
                        messages=[{"role": "user", "content": refine_prompt}],
                        max_tokens=settings.llm.max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    refined = strip_thinking_process(response.choices[0].message.content)
                new_result = CoverLetterResult(
                    job_title=job.title,
                    job_company=job.company,
                    job_url=str(job.url),
                    cover_letter_text=refined,
                    user_modifications=user_instructions,
                    attempt=attempt + 1,
                )
                session.add_attempt(new_result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
            continue

        elif choice == "D":
            from job_applicator.utils.diff import render_diff

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
) -> None:
    """Tailor a resume for a specific job with interactive preview."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)

    async def _run() -> None:
        from pydantic import HttpUrl

        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.documents.resume_tailor import ResumeTailor
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path)
        console.print(f"[green]Loaded resume: {resume_data.name}[/green]")

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
                str(entry.get("section", "")),
                str(entry.get("label", "")),
                str(entry.get("start", "")),
                str(entry.get("end", "")),
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

        try:
            with console.status("Tailoring resume..."):
                result = await tailor_engine.tailor(resume_data, job, user_instructions, style)
            session.add_attempt(result)
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
            _render_diff(console, session.original_text, result.tailored_text, max_lines=30)

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
                output_dir.mkdir(parents=True, exist_ok=True)

                safe_company = job.company.replace(" ", "_").replace("/", "_")
                safe_title = job.title.replace(" ", "_").replace("/", "_")
                timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
                output_path = output_dir / filename

                output_path.write_text(result.tailored_text, encoding="utf-8")
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
                meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                console.print(f"[green]Metadata saved: {meta_path}[/green]")

                break

            elif choice == "R":
                console.print("[yellow]Regenerating...[/yellow]")
                user_instructions = ""
                try:
                    with console.status("Tailoring resume..."):
                        result = await tailor_engine.refine(resume_data, result, "", job)
                    result.attempt = attempt
                    session.add_attempt(result)
                except Exception as exc:
                    console.print(f"[red]LLM error: {exc}[/red]")
                    retry_choice = (
                        console.input("[bold cyan][R] Retry or [Q] Quit? [/bold cyan]")
                        .strip()
                        .upper()
                    )
                    if retry_choice == "Q":
                        break
                    continue
                continue

            elif choice == "I":
                user_instructions = console.input(
                    "\n[bold]Enter your instructions (e.g., 'emphasize "
                    "customer service', 'add troubleshooting detail'): "
                    "[/bold]"
                ).strip()
                if not user_instructions:
                    console.print("[yellow]No instructions provided, retrying.[/yellow]")
                try:
                    with console.status("Tailoring resume..."):
                        result = await tailor_engine.refine(
                            resume_data, result, user_instructions, job
                        )
                    result.attempt = attempt
                    session.add_attempt(result)
                except Exception as exc:
                    console.print(f"[red]LLM error: {exc}[/red]")
                    retry_choice = (
                        console.input("[bold cyan][R] Retry or [Q] Quit? [/bold cyan]")
                        .strip()
                        .upper()
                    )
                    if retry_choice == "Q":
                        break
                    continue
                continue

            elif choice == "D":
                _render_diff(console, session.original_text, result.tailored_text, max_lines=0)
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
                try:
                    with console.status("Refining section..."):
                        result = await tailor_engine.refine(
                            resume_data, result, user_instructions, job
                        )
                    result.attempt = attempt
                    session.add_attempt(result)
                except Exception as exc:
                    console.print(f"[red]LLM error: {exc}[/red]")
                    retry_choice = (
                        console.input("[bold cyan][R] Retry or [Q] Quit? [/bold cyan]")
                        .strip()
                        .upper()
                    )
                    if retry_choice == "Q":
                        break
                    continue
                continue

            elif choice == "Q":
                console.print("[yellow]Discarded. No changes saved.[/yellow]")
                break

            else:
                console.print("[red]Invalid choice. Please enter A, R, I, D, V, S, or Q.[/red]")

    asyncio.run(_run())


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
