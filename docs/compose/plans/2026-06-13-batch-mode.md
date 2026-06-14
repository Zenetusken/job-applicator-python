# Batch Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `batch` CLI command that runs match→tailor→cover-letter pipeline non-interactively across multiple jobs, with parallel execution and summary output.

**Architecture:** New `batch` command in `cli.py` that reuses `JobMatcher`, `ResumeTailor`, and `CoverLetterGenerator` with `Semaphore(3)` concurrency. No new modules — all components already exist. Jobs come from `--jobs-file` (JSON) or `--query` (live search). Pipeline: match+rank → filter by `--min-score` → take `--top-k` → parallel tailor+cover-letter → save per-job files + `batch_summary.json`.

**Tech Stack:** Python 3.12, Typer, Rich, asyncio, existing `ResumeTailor`/`CoverLetterGenerator`/`JobMatcher`

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `src/job_applicator/cli.py` | Modify | Add `batch` command (~120 lines) |
| `tests/unit/test_batch.py` | Create | Unit tests for batch command |
| `tests/test_batch_live.py` | Create | Live integration test |

---

### Task 1: Add `batch` command to CLI

**Covers:** Full batch pipeline

**Files:**
- Modify: `src/job_applicator/cli.py` (add after `match` command, before `generate-cover-letter`)

- [ ] **Step 1: Add the `batch` command function**

Insert after the `match` command's closing `asyncio.run(_run())` and before the `tailor` command:

```python
@app.command()
def batch(
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    jobs_file: str = typer.Option("", "--jobs-file", help="JSON file with job listings."),
    query: str = typer.Option("", "--query", "-q", help="Search query (alternative to --jobs-file)."),
    site: str = typer.Option("linkedin", "--site", "-s", help="Job board for --query."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Max jobs to tailor."),
    min_score: float = typer.Option(0.0, "--min-score", help="Skip jobs below this score."),
    cover_letter: bool = typer.Option(
        True, "--cover-letter/--no-cover-letter", help="Generate cover letters."
    ),
    style_guide: str = typer.Option("", "--style-guide", help="Style example file."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
    as_json: bool = typer.Option(False, "--json", help="Output results as JSON."),
) -> None:
    """Batch tailor resumes (and optionally cover letters) for multiple jobs."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)

    async def _run() -> None:
        import json

        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.documents.resume_tailor import ResumeTailor
        from job_applicator.embeddings.matching import JobMatcher, MatchResult
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        # Load resume
        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path)
        if not as_json:
            console.print(f"[green]Loaded resume: {resume_data.name}[/green]")

        # Load jobs
        jobs: list[JobListing] = []
        if jobs_file:
            try:
                with open(jobs_file) as f:  # noqa: ASYNC230
                    data = json.load(f)
                    for item in data:
                        jobs.append(JobListing(**item))
            except FileNotFoundError:
                console.print(f"[red]Jobs file not found: {jobs_file}[/red]")
                raise typer.Exit(1)
            except Exception as exc:
                console.print(f"[red]Error reading jobs file: {exc}[/red]")
                raise typer.Exit(1)
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

        # Match and rank
        matcher = JobMatcher(settings.embedding)
        with console.status("Computing match scores..."):
            matches = matcher.rank_jobs(resume_data, jobs, top_k=top_k)

        # Filter by min_score
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

        # Load style guide and cover letter generator once
        from datetime import datetime as dt
        from pathlib import Path

        style = None
        cl_generator = None
        if cover_letter:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            cl_generator = CoverLetterGenerator(settings.llm)
            if settings.style_guide_path:
                with console.status("Loading style guide..."):
                    style = await cl_generator.load_style_guide(settings.style_guide_path)

        # Parallel pipeline
        tailor_engine = ResumeTailor(settings.llm)
        user_profile = _load_user_profile(settings)
        sem = asyncio.Semaphore(3)
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        async def _process_one(
            match_result: MatchResult,
        ) -> dict:
            """Process a single job: tailor + optional cover letter."""
            job = match_result.job
            safe_company = "".join(c if c.isalnum() or c in "-_" else "_" for c in job.company)[:30]
            safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in job.title)[:30]

            async with sem:
                result = {"title": job.title, "company": job.company, "url": str(job.url)}

                # Tailor
                try:
                    tailored = await tailor_engine.tailor(
                        resume=resume_data,
                        job=job,
                        user_instructions="",
                        style_guide=style,
                        matcher=matcher,
                    )
                    result["match_score"] = round(tailored.match_score, 4)
                    result["semantic_score"] = round(tailored.semantic_score, 4)
                    result["skill_score"] = round(tailored.skill_score, 4)

                    # Save tailored resume
                    resume_filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
                    resume_path_out = f"{output_dir}/{resume_filename}"
                    await asyncio.to_thread(
                        Path(resume_path_out).write_text,
                        tailored.tailored_text,
                    )

                    # Save meta
                    meta_path = f"{output_dir}/{resume_filename.rsplit('.', 1)[0]}.meta.json"
                    tailored.output_path = resume_path_out
                    await asyncio.to_thread(
                        Path(meta_path).write_text,
                        tailored.model_dump_json(indent=2),
                    )
                    result["resume_path"] = resume_path_out
                    result["tailored"] = True
                except Exception as exc:
                    result["tailored"] = False
                    result["error"] = str(exc)
                    return result

                # Cover letter
                if cover_letter and cl_generator is not None:
                    try:
                        letter = await cl_generator.generate(
                            job,
                            user_profile,
                            resume_data,
                            style_guide=style,
                            tailored_resume_text=tailored.tailored_text,
                        )
                        cl_filename = (
                            f"cover_letter_{safe_company}_{safe_title}_{timestamp}.txt"
                        )
                        cl_path = f"{output_dir}/{cl_filename}"
                        await asyncio.to_thread(
                            Path(cl_path).write_text, letter
                        )
                        result["cover_letter_path"] = cl_path
                        result["cover_letter"] = True
                    except Exception as exc:
                        result["cover_letter"] = False
                        result["cl_error"] = str(exc)

                return result

        # Run all jobs in parallel
        with console.status("Processing jobs in parallel..."):
            batch_results = await asyncio.gather(
                *(_process_one(m) for m in matches)
            )

        # Write batch summary
        summary = {
            "timestamp": timestamp,
            "resume": settings.resume_path,
            "total_jobs": len(jobs),
            "matched": len(matches),
            "results": list(batch_results),
        }
        summary_path = f"{output_dir}/batch_summary_{timestamp}.json"
        await asyncio.to_thread(
            Path(summary_path).write_text,
            json.dumps(summary, indent=2),
        )

        # Display results
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
                score = r.get("match_score", 0)
                score_style = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
                score_str = f"[{score_style}]{score:.0%}[/{score_style}]" if r.get("tailored") else "[dim]N/A[/dim]"
                table.add_row(
                    r["title"],
                    r["company"],
                    score_str,
                    "✓" if r.get("tailored") else "✗",
                    "✓" if r.get("cover_letter") else ("✗" if cover_letter else "-"),
                    r.get("error", r.get("cl_error", "")),
                )

            console.print(table)
            tailored_ok = sum(1 for r in batch_results if r.get("tailored"))
            cl_ok = sum(1 for r in batch_results if r.get("cover_letter"))
            console.print(
                f"\n[green]{tailored_ok}[/green] tailored, "
                f"[green]{cl_ok}[/green] cover letters"
            )
            console.print(f"Summary: {summary_path}")

    asyncio.run(_run())
```

- [ ] **Step 2: Run lint and typecheck**

```bash
ruff check src/job_applicator/cli.py
ruff format src/job_applicator/cli.py
mypy src/job_applicator/cli.py --ignore-missing-imports
```

Expected: Clean (or only pre-existing errors).

---

### Task 2: Write unit tests for batch command

**Covers:** Batch pipeline logic

**Files:**
- Create: `tests/unit/test_batch.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for the batch CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.models import JobBoard, JobListing, ResumeData


@pytest.fixture
def sample_jobs_file(tmp_path: Path) -> Path:
    """Create a sample jobs JSON file."""
    jobs = [
        {
            "title": "Python Developer",
            "company": "TechCorp",
            "url": "https://example.com/1",
            "description": "Python, FastAPI",
            "requirements": ["Python", "FastAPI"],
            "board": "linkedin",
        },
        {
            "title": "Backend Engineer",
            "company": "StartupXYZ",
            "url": "https://example.com/2",
            "description": "Django, PostgreSQL",
            "requirements": ["Django", "PostgreSQL"],
            "board": "linkedin",
        },
    ]
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps(jobs))
    return jobs_file


@pytest.fixture
def sample_resume_file(tmp_path: Path) -> Path:
    """Create a sample resume file."""
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "John Doe\njohn@example.com\n555-0123\n\n"
        "Summary:\nPython developer\n\n"
        "Skills:\nPython, FastAPI, Django\n"
    )
    return resume


class TestBatchCommand:
    """Tests for the batch CLI command."""

    def test_batch_command_exists(self):
        """Batch command is registered in the CLI."""
        from job_applicator.cli import app

        commands = {cmd.name for cmd in app.registered_commands}
        assert "batch" in commands

    def test_batch_loads_jobs_from_file(self, sample_jobs_file: Path):
        """Batch command loads jobs from --jobs-file JSON."""
        data = json.loads(sample_jobs_file.read_text())
        assert len(data) == 2
        assert data[0]["title"] == "Python Developer"
        assert data[0]["board"] == "linkedin"

    def test_batch_job_file_format_matches_model(self, sample_jobs_file: Path):
        """Jobs from file deserialize into JobListing correctly."""
        data = json.loads(sample_jobs_file.read_text())
        for item in data:
            job = JobListing(**item)
            assert job.title
            assert job.company
            assert job.board in (JobBoard.LINKEDIN, JobBoard.INDEED)

    def test_batch_requires_resume(self):
        """Batch exits if no resume provided."""
        from typer.testing import CliRunner
        from job_applicator.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["batch", "--jobs-file", "nonexistent.json"])
        assert result.exit_code != 0 or "resume" in result.output.lower()

    def test_batch_requires_jobs_or_query(self, sample_resume_file: Path):
        """Batch exits if neither --jobs-file nor --query provided."""
        from typer.testing import CliRunner
        from job_applicator.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["batch", "--resume", str(sample_resume_file)])
        assert result.exit_code != 0 or "jobs-file" in result.output.lower() or "query" in result.output.lower()

    def test_batch_min_score_filter(self):
        """Jobs below min_score are filtered out."""
        from job_applicator.embeddings.matching import MatchResult

        matches = [
            MatchResult(
                job=JobListing(
                    title="Good", company="A", url="https://example.com/1",
                    board=JobBoard.LINKEDIN,
                ),
                score=0.7,
                matched_skills=["Python"],
                missing_skills=[],
                summary="good match",
            ),
            MatchResult(
                job=JobListing(
                    title="Bad", company="B", url="https://example.com/2",
                    board=JobBoard.LINKEDIN,
                ),
                score=0.3,
                matched_skills=[],
                missing_skills=["Python"],
                summary="poor match",
            ),
        ]
        min_score = 0.5
        filtered = [m for m in matches if m.score >= min_score]
        assert len(filtered) == 1
        assert filtered[0].job.title == "Good"

    def test_batch_top_k_limits_results(self):
        """--top-k limits the number of jobs processed."""
        from job_applicator.embeddings.matching import MatchResult

        matches = [
            MatchResult(
                job=JobListing(
                    title=f"Job {i}", company=f"Co {i}",
                    url=f"https://example.com/{i}",
                    board=JobBoard.LINKEDIN,
                ),
                score=0.8 - i * 0.1,
                matched_skills=[],
                missing_skills=[],
                summary="",
            )
            for i in range(10)
        ]
        top_k = 3
        assert len(matches[:top_k]) == 3

    def test_batch_summary_json_structure(self, tmp_path: Path):
        """batch_summary.json has correct structure."""
        summary = {
            "timestamp": "20260613_120000",
            "resume": "/path/to/resume.pdf",
            "total_jobs": 5,
            "matched": 3,
            "results": [
                {
                    "title": "Python Dev",
                    "company": "TechCorp",
                    "url": "https://example.com/1",
                    "match_score": 0.75,
                    "semantic_score": 0.45,
                    "skill_score": 0.30,
                    "tailored": True,
                    "resume_path": "output/tailored_TechCorp_Python_Dev_20260613_120000.txt",
                    "cover_letter": True,
                    "cover_letter_path": "output/cover_letter_TechCorp_Python_Dev_20260613_120000.txt",
                }
            ],
        }
        summary_path = tmp_path / "batch_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        loaded = json.loads(summary_path.read_text())
        assert loaded["total_jobs"] == 5
        assert loaded["results"][0]["tailored"] is True
        assert loaded["results"][0]["match_score"] == 0.75
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/unit/test_batch.py -v
```

Expected: All 8 tests pass.

---

### Task 3: Write live integration test

**Covers:** End-to-end batch pipeline with real LLM

**Files:**
- Create: `tests/test_batch_live.py`

- [ ] **Step 1: Write the live test**

```python
#!/usr/bin/env python3
"""Live integration test for batch mode."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()


async def test_batch_live():
    """Run a real batch pipeline end-to-end."""
    console.print("[bold cyan]LIVE BATCH MODE TEST[/]\n")

    from job_applicator.config import LLMConfig, EmbeddingConfig
    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.models import JobBoard, JobListing, ResumeData

    # Create test resume
    resume_text = """John Doe
john@example.com
555-0123

Summary:
Senior Python developer with 6 years of experience.

Skills:
Python, FastAPI, Django, PostgreSQL, Docker, AWS, Redis, asyncio

Experience:
Senior Python Developer | TechStart Inc. | 2021-Present
- Built microservices handling 10M+ daily requests

Python Developer | CodeBase Corp | 2019-2021
- Developed RESTful APIs using Django REST Framework
"""

    resume = ResumeData(
        raw_text=resume_text,
        name="John Doe",
        email="john@example.com",
        phone="555-0123",
        summary="Senior Python developer",
        skills=["Python", "FastAPI", "Django", "PostgreSQL", "Docker", "AWS", "Redis", "asyncio"],
    )

    # Create test jobs
    jobs = [
        JobListing(
            title="Senior Python Developer", company="TechCorp",
            url="https://example.com/1", description="Python, FastAPI, AWS",
            requirements=["Python", "FastAPI", "AWS"], location="Remote",
            board=JobBoard.LINKEDIN,
        ),
        JobListing(
            title="Backend Engineer", company="StartupXYZ",
            url="https://example.com/2", description="Django, PostgreSQL, Docker",
            requirements=["Django", "PostgreSQL", "Docker"], location="SF",
            board=JobBoard.LINKEDIN,
        ),
        JobListing(
            title="Marketing Manager", company="AdCo",
            url="https://example.com/3", description="SEO, social media",
            requirements=["SEO", "Social Media"], location="NYC",
            board=JobBoard.LINKEDIN,
        ),
    ]

    # Step 1: Match and rank
    console.print("[bold]Step 1: Match and rank[/]")
    matcher = JobMatcher(EmbeddingConfig())
    matches = matcher.rank_jobs(resume, jobs, top_k=3)
    for m in matches:
        console.print(f"  {m.job.title}: {m.score:.3f}")

    # Step 2: Filter by min_score
    min_score = 0.4
    filtered = [m for m in matches if m.score >= min_score]
    console.print(f"\n[bold]Step 2: Filter (min_score={min_score})[/]")
    console.print(f"  {len(filtered)}/{len(matches)} jobs above threshold")

    # Step 3: Parallel tailoring
    console.print("\n[bold]Step 3: Parallel tailoring[/]")
    tailor_engine = ResumeTailor(LLMConfig())
    sem = asyncio.Semaphore(3)

    async def tailor_one(match_result):
        async with sem:
            try:
                result = await tailor_engine.tailor(
                    resume=resume, job=match_result.job, matcher=matcher,
                    user_instructions="",
                )
                return {
                    "title": match_result.job.title,
                    "company": match_result.job.company,
                    "match_score": round(result.match_score, 4),
                    "semantic_score": round(result.semantic_score, 4),
                    "skill_score": round(result.skill_score, 4),
                    "tailored": True,
                    "tailored_length": len(result.tailored_text),
                }
            except Exception as e:
                return {
                    "title": match_result.job.title,
                    "company": match_result.job.company,
                    "tailored": False,
                    "error": str(e),
                }

    import time
    start = time.monotonic()
    results = await asyncio.gather(*(tailor_one(m) for m in filtered))
    elapsed = time.monotonic() - start

    # Step 4: Verify results
    console.print(f"\n[bold]Step 4: Verify ({elapsed:.1f}s)[/]")
    checks = []
    for r in results:
        ok = r.get("tailored", False)
        checks.append(ok)
        icon = "✓" if ok else "✗"
        score = r.get("match_score", 0)
        console.print(f"  {icon} {r['title']} at {r['company']}: score={score}, len={r.get('tailored_length', 0)}")

    # Step 5: Summary JSON
    summary = {
        "timestamp": "live_test",
        "total_jobs": len(jobs),
        "matched": len(filtered),
        "results": results,
    }
    console.print(f"\n[bold]Step 5: Summary JSON[/]")
    console.print(json.dumps(summary, indent=2))

    all_ok = all(checks)
    console.print(f"\n{'✓ ALL PASS' if all_ok else '✗ SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    passed = asyncio.run(test_batch_live())
    sys.exit(0 if passed else 1)
```

- [ ] **Step 2: Run the live test**

```bash
.venv/bin/python tests/test_batch_live.py
```

Expected: All jobs tailored, summary JSON printed.

---

### Task 4: Final verification

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: All tests pass (existing + new).

- [ ] **Step 2: Run lint and typecheck**

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/job_applicator/ --ignore-missing-imports
```

Expected: Clean.
