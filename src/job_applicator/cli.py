"""CLI entry point — Typer + Rich for terminal UX."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from job_applicator import __version__
from job_applicator.config import AppSettings
from job_applicator.models import UserProfile
from job_applicator.utils.logging import setup_logging

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
                        if p.exists():  # noqa: ASYNC240
                            if p.suffix.lower() == ".pdf":
                                resume = loader.load(p)
                                texts.append(resume.raw_text)
                            else:
                                texts.append(p.read_text(encoding="utf-8"))  # noqa: ASYNC240

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
