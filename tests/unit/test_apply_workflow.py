"""Characterization tests for the apply command's per-job apply loop.

Tests-first guard before extracting the apply loop to a workflow module. Drives the real
`apply` command with mocked browser/scraper/applicator/state, pinning the dry-run +
submit behavior (per-job apply, skip-already-applied, daily cap, results, validate-exit).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    ATSCompatibilityResult,
    DryRunValidation,
    JobBoard,
    JobListing,
    ResumeData,
    UserProfile,
)


def _jobs(n: int = 2) -> list[JobListing]:
    return [
        JobListing(
            title=f"Dev{i}",
            company=f"Co{i}",
            url=f"https://example.com/{i}",
            board=JobBoard.LINKEDIN,
        )
        for i in range(1, n + 1)
    ]


def _drive(
    args: list[str],
    *,
    jobs: list[JobListing] | None = None,
    apply_fn: Callable[..., ApplicationResult] | None = None,
    state: MagicMock | None = None,
):
    """Drive the `apply` command with mocked browser/scraper/applicator/state."""
    import job_applicator.cli as cli

    jobs = jobs if jobs is not None else _jobs(2)
    scraper = MagicMock()
    scraper.scrape = AsyncMock(return_value=jobs)

    applicator = MagicMock()
    if apply_fn is not None:
        applicator.apply = AsyncMock(side_effect=apply_fn)
    else:

        async def _default_apply(job, letter, submit=False):  # type: ignore[no-untyped-def]
            return ApplicationResult(job=job, status=ApplicationStatus.PENDING)

        applicator.apply = AsyncMock(side_effect=_default_apply)

    browser_cm = MagicMock()
    browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    browser_cm.__aexit__ = AsyncMock(return_value=False)

    st = state or MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})

    with (
        patch.object(cli, "_make_browser", return_value=browser_cm),
        patch.object(cli, "_make_scraper", return_value=scraper),
        patch.object(cli, "_make_applicator", return_value=applicator),
        patch("job_applicator.workflows.apply.ApplicationState", return_value=st),
    ):
        result = CliRunner().invoke(cli.app, ["apply", *args])
    return result, applicator, st


def test_apply_dry_run_applies_each_job_without_submitting() -> None:
    """Dry run (default): every job is applied with submit=False; no state writes."""
    result, applicator, st = _drive(["-q", "python", "-n", "2"])
    assert result.exit_code == 0, result.output
    assert applicator.apply.await_count == 2
    for call in applicator.apply.await_args_list:
        assert call.kwargs.get("submit") is False
    st.record.assert_not_called()  # dry run never records


def test_apply_no_jobs_found_skips_apply() -> None:
    """An empty search result short-circuits before the apply loop."""
    result, applicator, _ = _drive(["-q", "python"], jobs=[])
    assert result.exit_code == 0, result.output
    assert "No jobs found" in result.output
    applicator.apply.assert_not_awaited()


def test_apply_validate_fails_when_dry_run_misses_submit() -> None:
    """--validate exits non-zero when a dry run does not reach the Submit step."""

    def _miss(job, letter, submit=False):  # type: ignore[no-untyped-def]
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.PENDING,
            dry_run=DryRunValidation(reached_submit=False),
        )

    result, _applicator, _ = _drive(["-q", "python", "-n", "1", "--validate"], apply_fn=_miss)
    assert result.exit_code == 1, result.output
    assert "Validation failed" in result.output


def test_apply_json_output_lists_each_result() -> None:
    """--json emits a machine-readable array of per-job outcomes."""
    import json

    result, _applicator, _ = _drive(["-q", "python", "-n", "2", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output[result.output.index("[") :])
    assert len(data) == 2
    assert {d["status"] for d in data} == {"pending"}


def test_apply_submit_records_each_application() -> None:
    """--submit applies with submit=True and records each outcome in local state."""
    result, applicator, st = _drive(["-q", "python", "-n", "2", "--submit", "--no-cover-letter"])
    assert result.exit_code == 0, result.output
    assert applicator.apply.await_count == 2
    for call in applicator.apply.await_args_list:
        assert call.kwargs.get("submit") is True
    assert st.record.call_count == 2


def test_apply_submit_skips_already_applied() -> None:
    """On submit, jobs the local state marks as already-applied are skipped (no apply call)."""
    st = MagicMock(**{"has_applied.return_value": True, "count_today.return_value": 0})
    result, applicator, _ = _drive(
        ["-q", "python", "-n", "2", "--submit", "--no-cover-letter"], state=st
    )
    assert result.exit_code == 0, result.output
    applicator.apply.assert_not_awaited()
    assert "already applied" in result.output.lower()


def test_apply_submit_stops_at_daily_cap() -> None:
    """On submit, the daily application cap short-circuits the apply loop."""
    st = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 9999})
    result, applicator, _ = _drive(["-q", "python", "--submit", "--no-cover-letter"], state=st)
    assert result.exit_code == 0, result.output
    applicator.apply.assert_not_awaited()
    assert "cap reached" in result.output.lower()


def _drive_cover_letter(
    args: list[str],
    *,
    jobs: list[JobListing] | None = None,
    cover_letter_text: str = "Generated cover letter text",
    resume_data: ResumeData | None = None,
) -> tuple[Any, MagicMock, MagicMock]:
    """Drive `apply` with cover-letter generation mocked.

    Returns ``(result, applicator, cl_generator_mock)``.
    """
    import job_applicator.cli as cli

    jobs = jobs if jobs is not None else _jobs(2)
    scraper = MagicMock()
    scraper.scrape = AsyncMock(return_value=jobs)

    applicator = MagicMock()

    async def _default_apply(job, letter, submit=False):  # type: ignore[no-untyped-def]
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.SUBMITTED if submit else ApplicationStatus.PENDING,
            cover_letter=letter,
        )

    applicator.apply = AsyncMock(side_effect=_default_apply)

    browser_cm = MagicMock()
    browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    browser_cm.__aexit__ = AsyncMock(return_value=False)

    st = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})

    if resume_data is None:
        resume_data = ResumeData(
            raw_text="Jane Doe\njane@example.com\nSkills: Python",
            name="Jane Doe",
            email="jane@example.com",
            skills=["Python"],
        )

    user_profile = UserProfile(
        first_name="Jane",
        last_name="Doe",
        email="jane@example.com",
        phone="",
        resume_path="/fake/resume.pdf",
    )

    ats_result = ATSCompatibilityResult(score=1.0)

    loader = MagicMock()
    loader.load.return_value = resume_data

    cl_generator = MagicMock()
    cl_generator.generate = AsyncMock(return_value=cover_letter_text)

    with (
        patch.object(cli, "_make_browser", return_value=browser_cm),
        patch.object(cli, "_make_scraper", return_value=scraper),
        patch.object(cli, "_make_applicator", return_value=applicator),
        patch("job_applicator.workflows.apply.ApplicationState", return_value=st),
        patch("job_applicator.documents.resume.ResumeLoader", return_value=loader),
        patch(
            "job_applicator.documents.cover_letter.CoverLetterGenerator",
            return_value=cl_generator,
        ),
        patch.object(cli, "_load_user_profile", return_value=user_profile),
        patch.object(cli, "_run_ats_preflight", return_value=ats_result),
    ):
        result = CliRunner().invoke(cli.app, ["apply", *args])
    return result, applicator, cl_generator


def test_apply_dry_run_generates_cover_letter() -> None:
    """Dry run with --resume generates a cover letter and passes it to the applicator."""
    result, applicator, cl_generator = _drive_cover_letter(
        ["-q", "python", "-n", "1", "--resume", "/fake/resume.pdf"]
    )
    assert result.exit_code == 0, result.output
    cl_generator.generate.assert_awaited_once()
    applicator.apply.assert_awaited_once()
    call = applicator.apply.await_args
    assert call.kwargs.get("submit") is False
    cover_letter = call.args[1] if len(call.args) > 1 else call.kwargs.get("cover_letter")
    assert cover_letter == "Generated cover letter text"


def test_apply_dry_run_no_cover_letter_flag_skips_generation() -> None:
    """--no-cover-letter prevents any cover-letter generation in dry run."""
    result, applicator, cl_generator = _drive_cover_letter(
        ["-q", "python", "-n", "1", "--resume", "/fake/resume.pdf", "--no-cover-letter"]
    )
    assert result.exit_code == 0, result.output
    cl_generator.generate.assert_not_awaited()
    applicator.apply.assert_awaited_once()
    call = applicator.apply.await_args
    cover_letter = call.args[1] if len(call.args) > 1 else call.kwargs.get("cover_letter")
    assert cover_letter is None


def _extract_json_array(output: str) -> list[object]:
    """Extract the final JSON array from CLI output that may contain log lines."""
    lines = output.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("["):
            return json.loads("\n".join(lines[i:]))
    raise ValueError("No JSON array found in output")


def test_apply_json_output_includes_cover_letter() -> None:
    """--json dry-run output contains the generated cover letter text."""
    result, _applicator, _cl_generator = _drive_cover_letter(
        ["-q", "python", "-n", "1", "--resume", "/fake/resume.pdf", "--json"],
        cover_letter_text="Preview cover letter",
    )
    assert result.exit_code == 0, result.output
    data = _extract_json_array(result.output)
    assert len(data) == 1
    assert data[0]["cover_letter"] == "Preview cover letter"


def test_apply_dry_run_cover_letter_note_in_table() -> None:
    """Console table shows a cover-letter length note during dry run."""
    result, _applicator, _cl_generator = _drive_cover_letter(
        ["-q", "python", "-n", "1", "--resume", "/fake/resume.pdf"],
        cover_letter_text="Preview cover letter",
    )
    assert result.exit_code == 0, result.output
    assert "cover letter:" in result.output.lower()
