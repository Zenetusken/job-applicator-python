"""Characterization tests for the apply command's per-job apply loop.

Tests-first guard before extracting the apply loop to a workflow module. Drives the real
`apply` command with mocked browser/scraper/applicator/state, pinning the dry-run +
submit behavior (per-job apply, skip-already-applied, daily cap, results, validate-exit).
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    DryRunValidation,
    JobBoard,
    JobListing,
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
        patch.object(cli, "ApplicationState", return_value=st),
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
