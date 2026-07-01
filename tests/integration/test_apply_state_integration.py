"""Integration tests for the apply loop against a REAL ``ApplicationState``.

The unit suite (``tests/unit/test_apply_workflow.py``) drives the loop with a ``MagicMock``
state whose ``count_today``/``has_applied`` return canned constants — so it can prove the loop
*checks* the cap, but never that real SQLite persistence produces the right emergent behavior as
applications accumulate. These tests close that seam: they call ``_apply_to_jobs`` directly with a
fake applicator but a real ``ApplicationState`` (pointed at the shared tmp DB by the autouse
``_isolate_local_state`` conftest fixture), and assert on the persisted rows.

The killer property: with ``max_applications_per_day=2`` and three SUBMITTED jobs, the real count
climbs 0→1→2 across the loop's own ``record``/``count_today`` calls and stops the third — a
progression a static mock cannot express. Also covered: pre-loop cap from prior state, cross-run
URL dedup, board-scoped cap isolation, dry-run persisting nothing, a submitted-row round-trip, and
the fail-closed stop when a mid-loop ``record`` raises ``StateError`` (a real WAL write failure).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from job_applicator.config import AppSettings, BrowserConfig, LLMConfig, TargetConfig
from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    JobBoard,
    JobListing,
)
from job_applicator.state import ApplicationState, StateError
from job_applicator.workflows.apply import _apply_to_jobs

_LINKEDIN = JobBoard.LINKEDIN.value


class _FakeApplicator:
    """Minimal stand-in for ``BaseApplicator`` — records calls, returns a fixed status."""

    def __init__(self, status: ApplicationStatus = ApplicationStatus.SUBMITTED) -> None:
        self._status = status
        self.calls: list[tuple[str, bool]] = []

    async def apply(
        self, job: JobListing, letter: str | None, submit: bool = False
    ) -> ApplicationResult:
        self.calls.append((str(job.url), submit))
        return ApplicationResult(job=job, status=self._status)


def _jobs(n: int, *, board: JobBoard = JobBoard.LINKEDIN) -> list[JobListing]:
    return [
        JobListing(
            title=f"Dev{i}",
            company=f"Co{i}",
            url=f"https://example.com/{board.value}/{i}",
            board=board,
        )
        for i in range(1, n + 1)
    ]


def _settings(tmp_path: Path, *, cap: int) -> AppSettings:
    """A real AppSettings with an explicit daily cap and zero pacing (init kwargs outrank env)."""
    return AppSettings(
        profile_name="Test User",
        resume_path="",
        output_dir=str(tmp_path / "out"),
        browser=BrowserConfig(headless=True, slow_mo=0, timeout_ms=5000),
        llm=LLMConfig(api_base="http://localhost:8000/v1", model="test-model"),
        target=TargetConfig(max_applications_per_day=cap, delay_between_applications_s=0.0),
    )


async def _run_loop(
    jobs: list[JobListing],
    applicator: _FakeApplicator,
    settings: AppSettings,
    *,
    submit: bool,
    site: str = _LINKEDIN,
) -> str:
    """Drive ``_apply_to_jobs`` with output captured to a string. Returns the console text."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    await _apply_to_jobs(
        jobs,
        applicator,  # type: ignore[arg-type]  # duck-typed fake applicator
        {},
        settings,
        site,
        len(jobs),
        submit=submit,
        validate=False,
        as_json=False,
        console=console,
        reporter=None,
    )
    return buf.getvalue()


def _submitted(job: JobListing) -> ApplicationResult:
    return ApplicationResult(job=job, status=ApplicationStatus.SUBMITTED)


async def test_real_cap_stops_loop_when_count_reaches_limit(tmp_path: Path) -> None:
    """cap=2, three SUBMITTED jobs: the real count climbs 0→1→2 and stops the third.

    A static mock (count_today == constant) cannot express this progression — only real
    persistence, where each record() bumps what the next count_today() sees, can.
    """
    applicator = _FakeApplicator(ApplicationStatus.SUBMITTED)
    output = await _run_loop(_jobs(3), applicator, _settings(tmp_path, cap=2), submit=True)

    # Exactly two applications happened; the third was stopped by the real count.
    assert len(applicator.calls) == 2, applicator.calls
    assert "cap reached" in output.lower()

    # And exactly two SUBMITTED rows are durably persisted.
    state = ApplicationState()
    assert state.count_today(board=_LINKEDIN) == 2


async def test_preloop_cap_from_prior_real_state_skips_entire_loop(tmp_path: Path) -> None:
    """Two SUBMITTED rows already in the store (cap=2): the pre-loop check sees the real count
    and skips the loop entirely — nothing is applied."""
    seed = ApplicationState()
    for job in _jobs(2):
        seed.record(_submitted(job))

    applicator = _FakeApplicator(ApplicationStatus.SUBMITTED)
    # Distinct URLs so this isn't an already-applied skip — it's the pre-loop cap.
    fresh_jobs = [
        j.model_copy(update={"url": f"https://example.com/fresh/{i}"})
        for i, j in enumerate(_jobs(2))
    ]
    output = await _run_loop(fresh_jobs, applicator, _settings(tmp_path, cap=2), submit=True)

    assert applicator.calls == []
    # Assert the PRE-LOOP branch specifically ("Skipping apply loop.") — not the generic
    # "cap reached", which the in-loop check also prints ("Stopping.") on iteration 1.
    assert "skipping apply loop" in output.lower()
    assert ApplicationState().count_today(board=_LINKEDIN) == 2  # unchanged


async def test_real_cross_run_dedup_skips_already_recorded_url(tmp_path: Path) -> None:
    """A URL recorded by a first loop run is really skipped on a second run (no re-apply)."""
    settings = _settings(tmp_path, cap=20)
    [job_a] = _jobs(1)

    first = _FakeApplicator(ApplicationStatus.SUBMITTED)
    await _run_loop([job_a], first, settings, submit=True)
    assert len(first.calls) == 1

    job_b = job_a.model_copy(update={"url": "https://example.com/linkedin/2", "title": "DevB"})
    second = _FakeApplicator(ApplicationStatus.SUBMITTED)
    output = await _run_loop([job_a, job_b], second, settings, submit=True)

    # job_a is skipped via the real state store; only job_b is applied.
    applied_urls = [url for url, _ in second.calls]
    assert applied_urls == [str(job_b.url)]
    assert "already applied" in output.lower()


async def test_cross_board_cap_isolation(tmp_path: Path) -> None:
    """A SUBMITTED row on another board does NOT consume this board's cap.

    Seed one SUBMITTED Indeed row, then run the LinkedIn loop with cap=1: the LinkedIn count
    starts at 0 (the Indeed row is board-scoped out), so the LinkedIn job still applies.
    """
    seed = ApplicationState()
    [indeed_job] = _jobs(1, board=JobBoard.INDEED)
    seed.record(_submitted(indeed_job))  # a prior SUBMITTED Indeed application

    applicator = _FakeApplicator(ApplicationStatus.SUBMITTED)
    output = await _run_loop(_jobs(1), applicator, _settings(tmp_path, cap=1), submit=True)

    # The LinkedIn job applied despite cap=1, because the board-scoped count started at 0 —
    # the Indeed row didn't consume the LinkedIn cap. Both rows are counted, board-scoped.
    assert len(applicator.calls) == 1, output
    final = ApplicationState()
    assert final.count_today(board=_LINKEDIN) == 1
    assert final.count_today(board=JobBoard.INDEED.value) == 1


async def test_dry_run_persists_nothing(tmp_path: Path) -> None:
    """A dry run (submit=False) applies each job but writes zero rows to the real store."""
    applicator = _FakeApplicator(ApplicationStatus.PENDING)
    await _run_loop(_jobs(2), applicator, _settings(tmp_path, cap=20), submit=False)

    assert len(applicator.calls) == 2
    assert all(submit is False for _, submit in applicator.calls)
    assert ApplicationState().list_recent() == []  # nothing persisted


async def test_submitted_rows_roundtrip_via_sibling_state(tmp_path: Path) -> None:
    """After a submit run, a SIBLING ApplicationState sees the persisted rows with correct
    status/board/url — proving the loop's writes are durable across connections."""
    jobs = _jobs(2)
    applicator = _FakeApplicator(ApplicationStatus.SUBMITTED)
    await _run_loop(jobs, applicator, _settings(tmp_path, cap=20), submit=True)

    sibling = ApplicationState()
    for job in jobs:
        assert sibling.has_applied(str(job.url), statuses={ApplicationStatus.SUBMITTED})
    recent = sibling.list_recent()
    assert {str(r.job.url) for r in recent} == {str(j.url) for j in jobs}
    assert all(r.status == ApplicationStatus.SUBMITTED for r in recent)


class _FailSecondRecord(ApplicationState):
    """A real ApplicationState whose second ``record`` raises — simulates a mid-run WAL write
    failure (a read can still succeed, so ``count_today`` freezes and the cap would be bypassed
    if the loop kept going; the loop must instead STOP fail-closed)."""

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(db_path)
        self._record_calls = 0

    def record(self, result: ApplicationResult, cover_letter_path: str | None = None) -> None:
        self._record_calls += 1
        if self._record_calls >= 2:
            raise StateError("injected: WAL write failed")
        super().record(result, cover_letter_path)


async def test_fail_closed_stops_when_record_fails_midloop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a mid-loop record() raises StateError, the loop STOPS (does not process the rest) and
    the row written before the failure survives — the fail-closed daily-cap safety contract."""
    monkeypatch.setattr("job_applicator.workflows.apply.ApplicationState", _FailSecondRecord)

    applicator = _FakeApplicator(ApplicationStatus.SUBMITTED)
    output = await _run_loop(_jobs(3), applicator, _settings(tmp_path, cap=20), submit=True)

    # job1 applied+recorded, job2 applied but record raised → STOP; job3 never attempted.
    assert len(applicator.calls) == 2, applicator.calls
    assert "failed to record" in output.lower()
    # Exactly the pre-failure row is durably persisted (a plain sibling store reads it back).
    assert ApplicationState().count_today(board=_LINKEDIN) == 1
