"""Unit tests for the application state store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from job_applicator.models import ApplicationResult, ApplicationStatus, JobBoard, JobListing
from job_applicator.state import ApplicationState


def _make_job(url: str = "https://linkedin.com/jobs/view/1") -> JobListing:
    return JobListing(
        title="Python Dev",
        company="Acme",
        url=url,
        board=JobBoard.LINKEDIN,
    )


def test_record_and_retrieve(tmp_path: Path) -> None:
    state = ApplicationState(db_path=tmp_path / "apps.db")
    job = _make_job()
    result = ApplicationResult(job=job, status=ApplicationStatus.SUBMITTED)

    state.record(result)

    assert state.has_applied(str(job.url))
    assert not state.has_applied("https://example.com/other")


def test_record_upserts(tmp_path: Path) -> None:
    state = ApplicationState(db_path=tmp_path / "apps.db")
    job = _make_job()
    state.record(ApplicationResult(job=job, status=ApplicationStatus.SUBMITTED))
    state.record(ApplicationResult(job=job, status=ApplicationStatus.FAILED))

    assert state.has_applied(str(job.url), statuses={ApplicationStatus.FAILED})


def test_has_applied_filters_status(tmp_path: Path) -> None:
    state = ApplicationState(db_path=tmp_path / "apps.db")
    job = _make_job()
    state.record(ApplicationResult(job=job, status=ApplicationStatus.SKIPPED))

    assert not state.has_applied(str(job.url))
    assert state.has_applied(str(job.url), statuses={ApplicationStatus.SKIPPED})


def test_has_applied_filters_since(tmp_path: Path) -> None:
    state = ApplicationState(db_path=tmp_path / "apps.db")
    job = _make_job()
    old = datetime.now(UTC) - timedelta(days=10)
    state.record(ApplicationResult(job=job, status=ApplicationStatus.SUBMITTED, timestamp=old))

    assert not state.has_applied(str(job.url), since=datetime.now(UTC) - timedelta(days=5))
    assert state.has_applied(str(job.url), since=datetime.now(UTC) - timedelta(days=15))


def test_count_today(tmp_path: Path) -> None:
    state = ApplicationState(db_path=tmp_path / "apps.db")
    yesterday = datetime.now(UTC) - timedelta(days=1)
    state.record(
        ApplicationResult(
            job=_make_job("https://linkedin.com/jobs/view/1"),
            status=ApplicationStatus.SUBMITTED,
            timestamp=yesterday,
        )
    )
    state.record(
        ApplicationResult(
            job=_make_job("https://linkedin.com/jobs/view/2"),
            status=ApplicationStatus.SUBMITTED,
        )
    )

    assert state.count_today() == 1
    assert state.count_today(board="linkedin") == 1
    assert state.count_today(board="indeed") == 0


def test_count_today_ignores_skipped_and_failed(tmp_path: Path) -> None:
    """Only real submissions count toward the daily cap."""
    state = ApplicationState(db_path=tmp_path / "apps.db")
    state.record(
        ApplicationResult(
            job=_make_job("https://linkedin.com/jobs/view/1"),
            status=ApplicationStatus.SUBMITTED,
        )
    )
    state.record(
        ApplicationResult(
            job=_make_job("https://linkedin.com/jobs/view/2"),
            status=ApplicationStatus.SKIPPED,
        )
    )
    state.record(
        ApplicationResult(
            job=_make_job("https://linkedin.com/jobs/view/3"),
            status=ApplicationStatus.FAILED,
        )
    )

    assert state.count_today() == 1


def test_result_timestamp_is_utc_aware(tmp_path: Path) -> None:
    state = ApplicationState(db_path=tmp_path / "apps.db")
    result = ApplicationResult(job=_make_job(), status=ApplicationStatus.SUBMITTED)

    assert result.timestamp.tzinfo is not None
    state.record(result)

    # Should still be countable today with a UTC-aware bound.
    assert state.count_today() == 1


def test_list_recent(tmp_path: Path) -> None:
    state = ApplicationState(db_path=tmp_path / "apps.db")
    state.record(
        ApplicationResult(
            job=_make_job("https://linkedin.com/jobs/view/1"),
            status=ApplicationStatus.SUBMITTED,
        )
    )

    recent = state.list_recent(limit=10)
    assert len(recent) == 1
    assert recent[0].status == ApplicationStatus.SUBMITTED


def test_has_applied_empty_statuses_returns_false(tmp_path: Path) -> None:
    """C6: an explicit empty statuses set returns False, not an `IN ()` SQL error."""
    state = ApplicationState(db_path=tmp_path / "apps.db")
    state.record(ApplicationResult(job=_make_job(), status=ApplicationStatus.SUBMITTED))
    assert state.has_applied("https://linkedin.com/jobs/view/1", statuses=set()) is False
