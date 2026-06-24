#!/usr/bin/env python3
"""Live end-to-end tests for the universal multi-file style-guide feature.

These tests exercise the real CLI commands with live vLLM calls and real disk
I/O to empirically validate that style-guide loading, OCR-mode forwarding,
``--no-cover-letter`` respect, JSON-stdout isolation, and interactive refine
style preservation are all wired correctly.

All tests are marked ``live`` and are skipped automatically when the vLLM
endpoint at ``localhost:8000`` is not reachable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.documents.resume import ResumeLoader
from job_applicator.documents.resume_tailor import ResumeTailor
from job_applicator.jobs_store import JobStore
from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    DryRunValidation,
    JobBoard,
    JobListing,
    StyleGuide,
)

pytestmark = pytest.mark.live


RESUME_TEXT = """\
Jane Doe
jane.doe@example.com
555-0123

Summary
-------
Senior software engineer with 8 years of experience building scalable web
services, developer tools, and data pipelines.

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


STYLE_GUIDE_TEXT = """\
Dear Hiring Manager,

I am writing to express my strong interest in the Senior Backend Engineer
position at CloudScale Inc. With over seven years of experience designing
resilient distributed systems, I bring a track record of turning ambiguous
requirements into production-grade services.

In my current role at TechStart, I lead the platform team responsible for the
core API infrastructure. We reduced p99 latency by 45% after migrating to a
Kafka-backed event architecture and introduced automated canary deployments
that cut production incidents by half. I thrive in environments where system
reliability and developer velocity are equally valued.

What draws me to CloudScale is your commitment to open-source tooling and your
transparent engineering culture. I would welcome the opportunity to contribute
my expertise in Python, FastAPI, and cloud-native infrastructure to your team.

Thank you for considering my application. I look forward to discussing how my
background aligns with your goals.

Sincerely,
Alex Morgan
"""


JOB_DESCRIPTION = """\
We are seeking a Senior Python Engineer to build scalable APIs and data pipelines.

Requirements:
- 5+ years of Python experience
- FastAPI or Django
- PostgreSQL, Docker, AWS
- Experience with CI/CD and testing

Nice to have: Kubernetes, asyncio
"""


def _write_resume(tmp_path: Path) -> Path:
    path = tmp_path / "resume.txt"
    path.write_text(RESUME_TEXT, encoding="utf-8")
    return path


def _write_style_guide(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "style.txt"
    path.write_text(STYLE_GUIDE_TEXT + extra, encoding="utf-8")
    return path


def _write_jobs_file(tmp_path: Path) -> Path:
    path = tmp_path / "jobs.json"
    jobs = [
        {
            "title": "Senior Python Engineer",
            "company": "CloudScale Inc.",
            "url": "https://example.com/jobs/1",
            "description": JOB_DESCRIPTION,
            "requirements": ["Python", "FastAPI", "PostgreSQL"],
            "location": "Remote",
            "board": "linkedin",
        },
        {
            "title": "Backend Engineer",
            "company": "DataFlow",
            "url": "https://example.com/jobs/2",
            "description": "Build Django APIs and PostgreSQL schemas.",
            "requirements": ["Python", "Django", "PostgreSQL"],
            "location": "Remote",
            "board": "linkedin",
        },
    ]
    path.write_text(json.dumps(jobs), encoding="utf-8")
    return path


def _extract_json(stdout: str) -> Any:
    """Return the first complete JSON object or array printed to stdout.

    Uses brace/bracket depth tracking so nested objects inside a larger JSON
    document are not mistaken for the root value.
    """
    lines = stdout.splitlines()
    for start, line in enumerate(lines):
        stripped = line.lstrip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            continue

        text = ""
        depth = 0
        in_string = False
        escaped = False
        for ch in "\n".join(lines[start:]):
            text += ch
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            break
        continue
    raise ValueError(f"No JSON found in stdout:\n{stdout}")


def _store_job(tmp_path: Path) -> str:
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
    JobStore().upsert_job(job, source_query="live style-guide test")
    return job_url


def _mock_applicator() -> MagicMock:
    """Return an applicator mock that returns a dry-run result."""

    async def _apply(
        job: JobListing,
        cover_letter: str | None,
        submit: bool = False,
    ) -> ApplicationResult:
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


@pytest.fixture(autouse=True)
def _isolated_style_cache_and_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep style-analysis cache and batch/tailor output inside the test tmp dir."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("JOB_APPLICATOR_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("JOB_APPLICATOR_LLM_MAX_TOKENS", "1024")


class TestGenerateCoverLetterStyleGuide:
    """Live CLI tests for ``generate-cover-letter --style-guide``."""

    def test_generate_cover_letter_with_style_guide_and_clean_json_stdout(
        self, tmp_path: Path
    ) -> None:
        """Style guide is loaded and JSON stdout contains only the result."""
        resume_path = _write_resume(tmp_path)
        style_path = _write_style_guide(tmp_path, extra="\n\nLive test token: alpha-1\n")
        runner = CliRunner()

        result = runner.invoke(
            cli.app,
            [
                "generate-cover-letter",
                "--resume",
                str(resume_path),
                "--job-title",
                "Senior Python Engineer",
                "--company",
                "CloudScale Inc.",
                "--description",
                JOB_DESCRIPTION,
                "--style-guide",
                str(style_path),
                "--json",
            ],
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        data = _extract_json(result.stdout)
        assert "cover_letter" in data
        letter = data["cover_letter"]
        assert isinstance(letter, str) and len(letter) > 100

        # Progress / style messages must be on stderr, never on stdout.
        assert "Style loaded" in result.stderr
        assert "Analyzing writing style" not in result.stdout
        assert "Style loaded" not in result.stdout
        assert "Generating cover letter" not in result.stdout

    def test_generate_cover_letter_without_style_guide_succeeds(self, tmp_path: Path) -> None:
        """The command works without a style guide and still emits clean JSON."""
        resume_path = _write_resume(tmp_path)
        runner = CliRunner()

        result = runner.invoke(
            cli.app,
            [
                "generate-cover-letter",
                "--resume",
                str(resume_path),
                "--job-title",
                "Senior Python Engineer",
                "--company",
                "CloudScale Inc.",
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        data = _extract_json(result.stdout)
        assert data["cover_letter"]
        assert "Style loaded" not in result.stderr


class TestBatchStyleGuide:
    """Live CLI tests for ``batch --style-guide`` and ``--no-cover-letter``."""

    def test_batch_with_style_guide_generates_cover_letters(self, tmp_path: Path) -> None:
        """Style guide is loaded and cover letters are produced for every job."""
        resume_path = _write_resume(tmp_path)
        jobs_path = _write_jobs_file(tmp_path)
        style_path = _write_style_guide(tmp_path, extra="\n\nLive test token: beta-1\n")
        runner = CliRunner()

        result = runner.invoke(
            cli.app,
            [
                "batch",
                "--resume",
                str(resume_path),
                "--jobs-file",
                str(jobs_path),
                "--style-guide",
                str(style_path),
                "--top-k",
                "2",
                "--json",
            ],
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        summary = _extract_json(result.stdout)
        results = summary["results"]
        assert len(results) >= 1

        assert "Style loaded" in result.stderr
        for r in results:
            assert r["tailored"] is True
            assert r["cover_letter"] is True
            assert Path(str(r["cover_letter_path"])).exists()

    def test_batch_no_cover_letter_with_style_guide_skips_cover_letters(
        self, tmp_path: Path
    ) -> None:
        """``--no-cover-letter`` suppresses cover letters even when style guide is given."""
        resume_path = _write_resume(tmp_path)
        jobs_path = _write_jobs_file(tmp_path)
        style_path = _write_style_guide(tmp_path, extra="\n\nLive test token: gamma-1\n")
        runner = CliRunner()

        result = runner.invoke(
            cli.app,
            [
                "batch",
                "--resume",
                str(resume_path),
                "--jobs-file",
                str(jobs_path),
                "--style-guide",
                str(style_path),
                "--top-k",
                "2",
                "--no-cover-letter",
                "--json",
            ],
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        summary = _extract_json(result.stdout)
        results = summary["results"]
        assert len(results) >= 1

        assert "Style loaded" in result.stderr
        for r in results:
            assert r["tailored"] is True
            assert "cover_letter_path" not in r
            assert r.get("cover_letter") is not True

        output_dir = tmp_path / "out"
        assert not any(p.name.startswith("cover_letter_") for p in output_dir.iterdir())

    def test_batch_style_guide_ocr_mode_is_forwarded(self, tmp_path: Path) -> None:
        """``--force-ocr`` is propagated into the style-guide loader."""
        resume_path = _write_resume(tmp_path)
        jobs_path = _write_jobs_file(tmp_path)
        style_path = _write_style_guide(tmp_path, extra="\n\nLive test token: delta-1\n")
        captured: dict[str, str] = {}
        original_load = ResumeLoader.load

        def _tracking_load(self: ResumeLoader, path: str | Path, ocr_mode: str = "auto") -> Any:
            if str(path) == str(style_path):
                captured["style_ocr_mode"] = ocr_mode
            return original_load(self, path, ocr_mode=ocr_mode)

        runner = CliRunner()
        with patch.object(ResumeLoader, "load", _tracking_load):
            result = runner.invoke(
                cli.app,
                [
                    "batch",
                    "--resume",
                    str(resume_path),
                    "--jobs-file",
                    str(jobs_path),
                    "--style-guide",
                    str(style_path),
                    "--top-k",
                    "1",
                    "--force-ocr",
                    "--json",
                ],
            )

        assert result.exit_code == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        assert captured.get("style_ocr_mode") == "on"
        summary = _extract_json(result.stdout)
        assert summary["results"][0]["tailored"] is True


class TestTailorStyleGuide:
    """Live CLI tests for ``tailor --style-guide`` and refine style preservation."""

    def test_tailor_json_with_style_guide(self, tmp_path: Path) -> None:
        """Tailor loads the style guide and emits only JSON on stdout."""
        resume_path = _write_resume(tmp_path)
        style_path = _write_style_guide(tmp_path, extra="\n\nLive test token: epsilon-1\n")
        runner = CliRunner()

        result = runner.invoke(
            cli.app,
            [
                "tailor",
                "--resume",
                str(resume_path),
                "--job-title",
                "Senior Python Engineer",
                "--company",
                "CloudScale Inc.",
                "--description",
                JOB_DESCRIPTION,
                "--style-guide",
                str(style_path),
                "--json",
            ],
        )

        assert result.exit_code == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        data = _extract_json(result.stdout)
        assert "tailored_text" in data
        assert len(data["tailored_text"]) > 100

        assert "Style loaded" in result.stderr
        assert "Style loaded" not in result.stdout

    def test_tailor_interactive_refine_preserves_style_guide(self, tmp_path: Path) -> None:
        """[R]etry and [A]ccept in the interactive loop keep the style guide."""
        resume_path = _write_resume(tmp_path)
        style_path = _write_style_guide(tmp_path, extra="\n\nLive test token: zeta-1\n")
        captured: dict[str, StyleGuide | None] = {"style_guide": None}
        original_refine = ResumeTailor.refine

        async def _tracking_refine(self: ResumeTailor, *args: Any, **kwargs: Any) -> Any:
            captured["style_guide"] = kwargs.get("style_guide")
            return await original_refine(self, *args, **kwargs)

        runner = CliRunner()
        with patch.object(ResumeTailor, "refine", _tracking_refine):
            result = runner.invoke(
                cli.app,
                [
                    "tailor",
                    "--resume",
                    str(resume_path),
                    "--job-title",
                    "Senior Python Engineer",
                    "--company",
                    "CloudScale Inc.",
                    "--description",
                    JOB_DESCRIPTION,
                    "--style-guide",
                    str(style_path),
                ],
                input="R\nA\nN\n",
            )

        assert result.exit_code == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "Tailored resume saved" in result.stdout
        assert captured["style_guide"] is not None
        assert isinstance(captured["style_guide"], StyleGuide)
        assert captured["style_guide"].tone


class TestApplyStyleGuide:
    """Live CLI tests for ``apply --style-guide`` dry-run behavior."""

    def test_apply_dry_run_with_style_guide_json_stdout_clean(self, tmp_path: Path) -> None:
        """Dry-run ``apply`` loads style guide and keeps JSON on stdout only."""
        resume_path = _write_resume(tmp_path)
        style_path = _write_style_guide(tmp_path, extra="\n\nLive test token: eta-1\n")
        job_url = _store_job(tmp_path)

        browser_cm = MagicMock()
        browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
        browser_cm.__aexit__ = AsyncMock(return_value=False)
        state = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})
        applicator = _mock_applicator()

        runner = CliRunner()
        with (
            patch.object(cli, "_make_browser", return_value=browser_cm),
            patch.object(cli, "_make_scraper", return_value=MagicMock()),
            patch.object(cli, "_make_applicator", return_value=applicator),
            patch("job_applicator.workflows.apply.ApplicationState", return_value=state),
        ):
            result = runner.invoke(
                cli.app,
                [
                    "apply",
                    "--from",
                    job_url,
                    "--resume",
                    str(resume_path),
                    "--style-guide",
                    str(style_path),
                    "--json",
                ],
            )

        assert result.exit_code == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        data = _extract_json(result.stdout)
        assert isinstance(data, list) and len(data) == 1
        assert data[0]["cover_letter"]
        assert len(data[0]["cover_letter"]) > 100

        # Dry-run banner and style messages belong on stderr.
        assert "Dry run" in result.stderr
        assert "Style loaded" in result.stderr
        assert "Style loaded" not in result.stdout
        assert "Analyzing writing style" not in result.stdout
