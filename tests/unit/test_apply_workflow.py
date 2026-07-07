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

import pytest
import typer
from typer.testing import CliRunner

from job_applicator.config import AppSettings
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


@pytest.fixture(autouse=True)
def _pin_apply_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the inter-application pacing delay to 0 for this file — keeps the real asyncio.sleep
    out of the suite runtime and decoupled from the dev box's ./config.toml (which may set a long
    delay). Pacing tests override it to a known value in their own body."""
    monkeypatch.setenv("JOB_APPLICATOR_TARGET_DELAY_BETWEEN_APPLICATIONS_S", "0")


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


def test_apply_json_output_includes_dry_run_evidence_fields() -> None:
    """Dry-run validation evidence is exposed in JSON without changing the top-level shape."""
    import json

    def _with_evidence(job, letter, submit=False):  # type: ignore[no-untyped-def]
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.PENDING,
            dry_run=DryRunValidation(
                reached_submit=True,
                resume_uploaded=True,
                resume_upload_accepted=False,
                form_validation_errors=["Please upload a resume."],
            ),
        )

    result, _applicator, _ = _drive(["-q", "python", "-n", "1", "--json"], apply_fn=_with_evidence)

    assert result.exit_code == 0, result.output
    data = json.loads(result.output[result.output.index("[") :])
    assert data[0]["dry_run"]["resume_uploaded"] is True
    assert data[0]["dry_run"]["resume_upload_accepted"] is False
    assert data[0]["dry_run"]["resume_upload_evidence"] == ""
    assert data[0]["dry_run"]["form_validation_errors"] == ["Please upload a resume."]


def test_apply_table_marks_actionable_dry_run_evidence() -> None:
    """Human output stays compact but calls out unconfirmed upload and form errors."""

    def _with_evidence(job, letter, submit=False):  # type: ignore[no-untyped-def]
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.PENDING,
            dry_run=DryRunValidation(
                reached_submit=True,
                resume_uploaded=True,
                resume_upload_accepted=False,
                form_validation_errors=["Please upload a resume."],
            ),
        )

    result, _applicator, _ = _drive(["-q", "python", "-n", "1"], apply_fn=_with_evidence)

    assert result.exit_code == 0, result.output
    assert "submit" in result.output
    assert "upload unconfirmed" in result.output
    assert "form errors: 1" in result.output


def test_apply_json_empty_result_emits_empty_array() -> None:
    """--json + an empty search result must emit valid JSON ([]) on stdout, not the human
    'No jobs found' text (CLAUDE.md: --json output is PURE parseable stdout)."""
    import json

    result, _applicator, _ = _drive(["-q", "python", "--json"], jobs=[])
    assert result.exit_code == 0, result.output
    assert "No jobs found" not in result.output  # fails before the fix (plain text on stdout)
    assert json.loads(result.output[result.output.index("[") :]) == []


def test_apply_submit_records_each_application() -> None:
    """--submit applies with submit=True and records each outcome in local state."""
    result, applicator, st = _drive(
        ["-q", "python", "-n", "2", "--submit", "--yes", "--no-cover-letter"]
    )
    assert result.exit_code == 0, result.output
    assert applicator.apply.await_count == 2
    for call in applicator.apply.await_args_list:
        assert call.kwargs.get("submit") is True
    assert st.record.call_count == 2


def test_apply_submit_skips_already_applied() -> None:
    """On submit, jobs the local state marks as already-applied are skipped (no apply call)."""
    st = MagicMock(**{"has_applied.return_value": True, "count_today.return_value": 0})
    result, applicator, _ = _drive(
        ["-q", "python", "-n", "2", "--submit", "--yes", "--no-cover-letter"], state=st
    )
    assert result.exit_code == 0, result.output
    applicator.apply.assert_not_awaited()
    assert "already applied" in result.output.lower()


def test_apply_submit_stops_at_daily_cap() -> None:
    """On submit, the daily application cap short-circuits the apply loop."""
    st = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 9999})
    result, applicator, _ = _drive(
        ["-q", "python", "--submit", "--yes", "--no-cover-letter"], state=st
    )
    assert result.exit_code == 0, result.output
    applicator.apply.assert_not_awaited()
    assert "cap reached" in result.output.lower()


def test_apply_submit_paces_between_applications(monkeypatch: pytest.MonkeyPatch) -> None:
    """On submit, consecutive real applications are paced by delay_between_applications_s —
    one inter-application sleep between two applies, and none trailing the last."""
    # Pin the delay via env (outranks the dev box's real ./config.toml, which may set a
    # non-default value) so the asserted 2.0 literal stays meaningful.
    monkeypatch.setenv("JOB_APPLICATOR_TARGET_DELAY_BETWEEN_APPLICATIONS_S", "2.0")
    with patch("job_applicator.workflows.apply.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result, applicator, _ = _drive(
            ["-q", "python", "-n", "2", "--submit", "--yes", "--no-cover-letter"]
        )
    assert result.exit_code == 0, result.output
    assert applicator.apply.await_count == 2
    sleep.assert_awaited_once_with(2.0)  # the configured gap, once, between the two applications


def test_apply_dry_run_does_not_pace_between_applications(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry runs are previews already paced by the per-step delays inside apply(); the
    inter-application sleep is submit-gated, so it must NOT fire on a dry run."""
    monkeypatch.setenv("JOB_APPLICATOR_TARGET_DELAY_BETWEEN_APPLICATIONS_S", "2.0")
    with patch("job_applicator.workflows.apply.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result, _applicator, _ = _drive(["-q", "python", "-n", "2", "--no-cover-letter"])
    assert result.exit_code == 0, result.output
    sleep.assert_not_awaited()


def test_apply_submit_without_yes_aborts_noninteractive() -> None:
    """--submit without --yes in a non-interactive context (CI / piped stdin) must REFUSE to send
    real applications: exit 1, a clear stderr message, and no apply call. With the dry-run default,
    this gate means a mistyped command or a script can't fire applications at the real account."""
    result, applicator, _ = _drive(["-q", "python", "--submit", "--no-cover-letter"])
    assert result.exit_code == 1, result.output
    assert "refus" in result.stderr.lower()
    applicator.apply.assert_not_awaited()


def test_apply_submit_interactive_decline_aborts() -> None:
    """An interactive --submit run the user DECLINES at the confirmation prompt sends nothing."""
    with (
        patch("job_applicator.cli._stdin_is_interactive", return_value=True),
        patch("typer.confirm", return_value=False),
    ):
        result, applicator, _ = _drive(["-q", "python", "--submit", "--no-cover-letter"])
    assert result.exit_code == 1, result.output
    applicator.apply.assert_not_awaited()


def test_apply_submit_interactive_accept_proceeds() -> None:
    """An interactive --submit run the user CONFIRMS proceeds to apply each job."""
    with (
        patch("job_applicator.cli._stdin_is_interactive", return_value=True),
        patch("typer.confirm", return_value=True),
    ):
        result, applicator, _ = _drive(["-q", "python", "-n", "2", "--submit", "--no-cover-letter"])
    assert result.exit_code == 0, result.output
    assert applicator.apply.await_count == 2


def test_apply_submit_json_requires_yes() -> None:
    """--submit --json must not prompt (it would corrupt JSON stdout): it REQUIRES --yes, refusing
    otherwise even when stdin is a TTY."""
    with patch("job_applicator.cli._stdin_is_interactive", return_value=True):
        result, applicator, _ = _drive(["-q", "python", "--submit", "--json", "--no-cover-letter"])
    assert result.exit_code == 1, result.output
    applicator.apply.assert_not_awaited()


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
    cl_generator.generate_verified = AsyncMock(return_value=cover_letter_text)

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
    cl_generator.generate_verified.assert_awaited_once()
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
    cl_generator.generate_verified.assert_not_awaited()
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


def test_apply_style_guide_flows_to_generator() -> None:
    """--style-guide is loaded by the shared helper and passed into generate()."""
    import job_applicator.cli as cli

    jobs = _jobs(1)
    scraper = MagicMock()
    scraper.scrape = AsyncMock(return_value=jobs)

    applicator = MagicMock()

    async def _default_apply(job, letter, submit=False):  # type: ignore[no-untyped-def]
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.PENDING,
            cover_letter=letter,
        )

    applicator.apply = AsyncMock(side_effect=_default_apply)

    browser_cm = MagicMock()
    browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    browser_cm.__aexit__ = AsyncMock(return_value=False)

    st = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})

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

    mock_style = MagicMock()
    mock_style.tone = "professional"

    cl_generator = MagicMock()
    cl_generator.load_style_guide = AsyncMock(return_value=mock_style)
    cl_generator.generate_verified = AsyncMock(return_value="Styled cover letter")

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
        result = CliRunner().invoke(
            cli.app,
            [
                "apply",
                "-q",
                "python",
                "-n",
                "1",
                "--resume",
                "/fake/resume.pdf",
                "--style-guide",
                "/fake/style.txt",
            ],
        )

    assert result.exit_code == 0, result.output
    cl_generator.load_style_guide.assert_awaited_once_with("/fake/style.txt", ocr_mode="auto")
    cl_generator.generate_verified.assert_awaited_once()
    call = cl_generator.generate_verified.await_args
    assert call.kwargs.get("style_guide") is mock_style


def test_apply_style_guide_messages_go_to_stderr_not_stdout() -> None:
    """Progress/status messages about style loading must not corrupt --json stdout."""
    import job_applicator.cli as cli

    jobs = _jobs(1)
    scraper = MagicMock()
    scraper.scrape = AsyncMock(return_value=jobs)

    applicator = MagicMock()

    async def _default_apply(job, letter, submit=False):  # type: ignore[no-untyped-def]
        return ApplicationResult(
            job=job,
            status=ApplicationStatus.PENDING,
            cover_letter=letter,
        )

    applicator.apply = AsyncMock(side_effect=_default_apply)

    browser_cm = MagicMock()
    browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    browser_cm.__aexit__ = AsyncMock(return_value=False)

    st = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})

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

    mock_style = MagicMock()
    mock_style.tone = "professional"

    cl_generator = MagicMock()
    cl_generator.load_style_guide = AsyncMock(return_value=mock_style)
    cl_generator.generate_verified = AsyncMock(return_value="Styled cover letter")

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
        result = CliRunner().invoke(
            cli.app,
            [
                "apply",
                "-q",
                "python",
                "-n",
                "1",
                "--resume",
                "/fake/resume.pdf",
                "--style-guide",
                "/fake/style.txt",
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    # Stdout should be parseable JSON starting at the first non-whitespace character.
    stdout = result.stdout
    stripped = stdout.lstrip()
    assert stripped.startswith("["), f"stdout started with non-JSON: {stripped[:80]!r}"
    # The info/status messages should be on stderr, not stdout.
    assert "Style loaded" in result.stderr
    assert "Dry run" in result.stderr
    data = _extract_json_array(stdout)
    assert len(data) == 1


async def _run_loop(
    app_settings: AppSettings,
    jobs: list[JobListing],
    *,
    submit: bool,
    cover_letter_failures: set[str] | None = None,
    record_side_effect: Exception | None = None,
) -> MagicMock:
    """Call _apply_to_jobs directly with a mock applicator + state; return the applicator mock."""
    from job_applicator.workflows import apply as apply_mod

    app_settings.target.delay_between_applications_s = 0  # no real pacing sleep in the unit suite
    applicator = MagicMock()

    async def _apply(
        job: JobListing, letter: str | None, submit: bool = False
    ) -> ApplicationResult:
        status = ApplicationStatus.SUBMITTED if submit else ApplicationStatus.PENDING
        return ApplicationResult(job=job, status=status)

    applicator.apply = AsyncMock(side_effect=_apply)
    st = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})
    if record_side_effect is not None:
        st.record.side_effect = record_side_effect

    with patch.object(apply_mod, "ApplicationState", return_value=st):
        await apply_mod._apply_to_jobs(
            jobs,
            applicator,
            {},
            app_settings,
            "linkedin",
            len(jobs),
            submit=submit,
            validate=False,
            as_json=False,
            console=MagicMock(),
            reporter=None,
            cover_letter_failures=cover_letter_failures,
        )
    return applicator


@pytest.mark.asyncio
async def test_apply_skips_submit_when_cover_letter_failed(app_settings: AppSettings) -> None:
    """NEW-1 (fail-loud): on a real --submit, a job whose REQUESTED cover letter FAILED to generate
    is skipped (not applied), never silently submitted letterless — a real application missing its
    intended letter is spent + unrecoverable, unlike a skipped job."""
    jobs = _jobs(2)
    applicator = await _run_loop(
        app_settings, jobs, submit=True, cover_letter_failures={str(jobs[0].url)}
    )
    applied = {str(c.args[0].url) for c in applicator.apply.await_args_list}
    assert str(jobs[0].url) not in applied  # failed-letter job NOT submitted
    assert str(jobs[1].url) in applied  # the other job still applied


@pytest.mark.asyncio
async def test_apply_dry_run_proceeds_when_cover_letter_failed(app_settings: AppSettings) -> None:
    """The fail-loud skip is SUBMIT-only: a dry run sends no real application, so a failed-letter
    job still runs (failure surfaced, nothing account-spending)."""
    jobs = _jobs(2)
    applicator = await _run_loop(
        app_settings, jobs, submit=False, cover_letter_failures={str(jobs[0].url)}
    )
    assert applicator.apply.await_count == 2  # both applied in dry run


@pytest.mark.asyncio
async def test_apply_loop_stops_on_record_failure_no_cap_bypass(app_settings: AppSettings) -> None:
    """NEW-2 (fail-CLOSED): a StateError from state.record must STOP the loop, never continue. Under
    WAL a read can succeed while a write fails, freezing count_today — so continuing would bypass
    the daily cap and send a real apply to EVERY job. The loop breaks after the first unrecorded
    apply (bounded to one), never mass-applies."""
    from job_applicator.state import StateError

    jobs = _jobs(5)  # 5 jobs, but the cap must NOT be bypassed when record keeps failing
    applicator = await _run_loop(
        app_settings, jobs, submit=True, record_side_effect=StateError("database is locked")
    )
    assert applicator.apply.await_count == 1  # stopped after the first apply, never reached 5


@pytest.mark.asyncio
async def test_apply_failed_letter_job_surfaced_as_failed(
    app_settings: AppSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    """A skipped failed-letter job is surfaced as a FAILED result in --json (not silently dropped),
    so a scripted --submit --json run can detect the letter never went out."""
    from job_applicator.workflows import apply as apply_mod

    app_settings.target.delay_between_applications_s = 0
    jobs = _jobs(1)
    applicator = MagicMock()

    async def _apply(
        job: JobListing, letter: str | None, submit: bool = False
    ) -> ApplicationResult:
        return ApplicationResult(job=job, status=ApplicationStatus.SUBMITTED)

    applicator.apply = AsyncMock(side_effect=_apply)
    st = MagicMock(**{"has_applied.return_value": False, "count_today.return_value": 0})
    with patch.object(apply_mod, "ApplicationState", return_value=st):
        with pytest.raises(typer.Exit):
            await apply_mod._apply_to_jobs(
                jobs,
                applicator,
                {},
                app_settings,
                "linkedin",
                1,
                submit=True,
                validate=False,
                as_json=True,
                console=MagicMock(),
                reporter=None,
                cover_letter_failures={str(jobs[0].url)},
            )
    data = json.loads(capsys.readouterr().out)
    assert data[0]["status"] == "failed"
    assert "cover letter" in (data[0]["error"] or "").lower()
    applicator.apply.assert_not_awaited()  # the real apply never happened
