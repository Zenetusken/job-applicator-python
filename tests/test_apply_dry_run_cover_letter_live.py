#!/usr/bin/env python3
"""Live end-to-end test for dry-run cover-letter generation.

Exercises the real `apply` CLI command with:
- Real vLLM cover-letter generation
- A real résumé file on disk
- The real jobs-store pathing
- Mocked browser/applicator/state so no real job board is contacted

Marked ``live`` because it calls the local LLM endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.jobs_store import JobStore
from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    DryRunValidation,
    JobBoard,
    JobListing,
)

pytestmark = pytest.mark.live


def _extract_json_array(output: str) -> list[object]:
    """Extract the final JSON array from CLI output that may contain log lines."""
    lines = output.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("["):
            return json.loads("\n".join(lines[i:]))
    raise ValueError("No JSON array found in output")


RESUME_TEXT = """\
Jane Doe
jane.doe@example.com
555-0123

Summary
-------
Senior software engineer with 8 years of experience building scalable web
services, developer tools, and data pipelines. Strong advocate for clean code,
code review, and automated testing.

Skills
------
Python, FastAPI, Django, PostgreSQL, Docker, Kubernetes, AWS, TypeScript, React

Experience
----------
Senior Software Engineer | TechCorp | 2020-present
- Led a team of 5 engineers rebuilding the core billing service in Python/FastAPI.
- Reduced API latency by 40% through caching and query optimization.
- Introduced pytest-based testing, raising coverage from 60% to 92%.

Software Engineer | StartupX | 2017-2020
- Built CI/CD pipelines with GitHub Actions and Docker.
- Maintained PostgreSQL and Redis services on AWS.

Education
---------
B.S. Computer Science, Example University
"""


def _store_job(tmp_path: object) -> str:
    """Insert a fake job into the isolated test store and return its URL."""
    job_url = "https://example.com/jobs/12345"
    job = JobListing(
        title="Senior Python Engineer",
        company="Example Corp",
        url=job_url,
        description="We need a senior Python engineer with FastAPI and PostgreSQL experience.",
        location="Remote",
        board=JobBoard.LINKEDIN,
    )
    JobStore().upsert_job(job, source_query="live test")
    return job_url


def _mock_applicator() -> MagicMock:
    """Return an applicator mock that returns the cover letter in the result."""

    async def _apply(job, cover_letter, submit=False):  # type: ignore[no-untyped-def]
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.SKIPPED,
            cover_letter=cover_letter,
            dry_run=DryRunValidation(
                reached_submit=True,
                easy_apply_button_found=True,
                cover_letter_field_found=cover_letter is not None,
            ),
            notes="DRY RUN: form prepared but not submitted.",
        )

    applicator = MagicMock()
    applicator.apply = AsyncMock(side_effect=_apply)
    return applicator


def test_apply_dry_run_generates_real_cover_letter(tmp_path: Path) -> None:
    """Dry run with --resume calls the live LLM and returns a real cover letter."""
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text(RESUME_TEXT, encoding="utf-8")
    job_url = _store_job(tmp_path)

    browser_cm = MagicMock()
    browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    browser_cm.__aexit__ = AsyncMock(return_value=False)

    state = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})
    applicator = _mock_applicator()

    with (
        patch.object(cli, "_make_browser", return_value=browser_cm),
        patch.object(cli, "_make_scraper", return_value=MagicMock()),
        patch.object(cli, "_make_applicator", return_value=applicator),
        patch("job_applicator.workflows.apply.ApplicationState", return_value=state),
    ):
        result = CliRunner().invoke(
            cli.app,
            [
                "apply",
                "--from",
                job_url,
                "--resume",
                str(resume_path),
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    data = _extract_json_array(result.output)
    assert len(data) == 1
    cover_letter = data[0]["cover_letter"]
    assert cover_letter is not None
    assert len(cover_letter) > 100, f"cover letter suspiciously short: {cover_letter!r}"
    assert "Example Corp" in cover_letter or "Python" in cover_letter


def test_apply_dry_run_no_cover_letter_skips_llm(tmp_path: Path) -> None:
    """--no-cover-letter in dry run produces no cover letter and makes no LLM call."""
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text(RESUME_TEXT, encoding="utf-8")
    job_url = _store_job(tmp_path)

    browser_cm = MagicMock()
    browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    browser_cm.__aexit__ = AsyncMock(return_value=False)

    state = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})
    applicator = _mock_applicator()

    with (
        patch.object(cli, "_make_browser", return_value=browser_cm),
        patch.object(cli, "_make_scraper", return_value=MagicMock()),
        patch.object(cli, "_make_applicator", return_value=applicator),
        patch("job_applicator.workflows.apply.ApplicationState", return_value=state),
    ):
        result = CliRunner().invoke(
            cli.app,
            [
                "apply",
                "--from",
                job_url,
                "--resume",
                str(resume_path),
                "--no-cover-letter",
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    data = _extract_json_array(result.output)
    assert len(data) == 1
    assert data[0]["cover_letter"] is None


def test_apply_dry_run_console_shows_cover_letter_note(tmp_path: Path) -> None:
    """Console table shows the cover-letter length note in dry run."""
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text(RESUME_TEXT, encoding="utf-8")
    job_url = _store_job(tmp_path)

    browser_cm = MagicMock()
    browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    browser_cm.__aexit__ = AsyncMock(return_value=False)

    state = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})
    applicator = _mock_applicator()

    with (
        patch.object(cli, "_make_browser", return_value=browser_cm),
        patch.object(cli, "_make_scraper", return_value=MagicMock()),
        patch.object(cli, "_make_applicator", return_value=applicator),
        patch("job_applicator.workflows.apply.ApplicationState", return_value=state),
    ):
        result = CliRunner().invoke(
            cli.app,
            [
                "apply",
                "--from",
                job_url,
                "--resume",
                str(resume_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "cover letter:" in result.output.lower()
