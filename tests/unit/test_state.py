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


def test_count_today_normalizes_nonutc_offset(tmp_path: Path) -> None:
    """A SUBMITTED application stamped in a non-UTC offset is counted on its UTC day, not its
    wall-clock day — else a +14:00 'today' that is really yesterday-in-UTC inflates the cap."""
    from datetime import timezone

    state = ApplicationState(db_path=tmp_path / "apps.db")
    # 1h before today's UTC midnight = yesterday-UTC, expressed in +14:00 so its wall-clock DATE
    # reads as today — exactly the case the old TEXT compare miscounted.
    today_midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    ts = (today_midnight - timedelta(hours=1)).astimezone(timezone(timedelta(hours=14)))
    assert ts.date() == today_midnight.date()  # the trap: wall-clock date is today
    state.record(
        ApplicationResult(job=_make_job(), status=ApplicationStatus.SUBMITTED, timestamp=ts)
    )
    assert state.count_today() == 0  # but it is yesterday in UTC → not in today's cap


def test_has_applied_since_normalizes_nonutc(tmp_path: Path) -> None:
    """`since` must be compared on the UTC scale too (stored applied_at is UTC): a `since` AFTER
    the stored instant but in a negative offset must not falsely match via raw TEXT compare."""
    from datetime import timezone

    state = ApplicationState(db_path=tmp_path / "apps.db")
    job = _make_job()
    stored = datetime(2026, 6, 26, 0, 0, tzinfo=UTC)
    state.record(ApplicationResult(job=job, status=ApplicationStatus.SUBMITTED, timestamp=stored))
    # 1h AFTER stored, expressed in -10:00 so its wall-clock string sorts BEFORE the stored one.
    since = datetime(2026, 6, 26, 1, 0, tzinfo=UTC).astimezone(timezone(timedelta(hours=-10)))
    assert state.has_applied(str(job.url), since=since) is False  # stored is before `since`


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
