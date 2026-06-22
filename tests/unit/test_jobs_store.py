"""Unit tests for the JobStore funnel store (the data backbone).

All tests use an isolated ``JobStore(db_path=tmp_path / ...)`` — they never touch the
real ``~/.job-applicator/applications.db``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from job_applicator.embeddings.matching import MatchResult
from job_applicator.jobs_store import JobStore, JobStoreError
from job_applicator.models import FunnelStatus, JobBoard, JobListing


def _job(n: int = 1, **over: object) -> JobListing:
    data: dict[str, object] = {
        "title": f"Engineer {n}",
        "company": f"Co{n}",
        "url": f"https://linkedin.com/jobs/{n}",
        "description": "async pipelines",
        "location": "Remote",
        "requirements": ["python", "asyncio"],
        "board": JobBoard.LINKEDIN,
        "seniority": "senior",
    }
    data.update(over)
    return JobListing(**data)  # type: ignore[arg-type]


def _match(job: JobListing, score: float = 0.9) -> MatchResult:
    return MatchResult(
        job=job,
        score=score,
        semantic_score=score,
        skill_score=score - 0.05,
        matched_skills=["python"],
        missing_skills=["kubernetes"],
        summary="Strong match",
    )


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(db_path=tmp_path / "applications.db")


def test_upsert_job_roundtrip(store: JobStore) -> None:
    store.upsert_job(_job(1), source_query="python remote")
    got = store.get("1")
    assert got is not None
    assert got.id == 1
    assert got.funnel_status is FunnelStatus.FOUND
    assert got.job.title == "Engineer 1"
    assert got.job.company == "Co1"
    assert str(got.job.url) == "https://linkedin.com/jobs/1"
    assert got.job.requirements == ["python", "asyncio"]
    assert got.source_query == "python remote"
    assert got.match_score is None


def test_get_by_url_and_unknown(store: JobStore) -> None:
    store.upsert_job(_job(1))
    assert store.get("https://linkedin.com/jobs/1") is not None
    assert store.get("999") is None
    assert store.get("https://nope.example/x") is None
    assert store.get("") is None
    assert store.get("   ") is None


def test_upsert_match_advances_to_matched_with_scores(store: JobStore) -> None:
    job = _job(1)
    store.upsert_job(job)
    store.upsert_match(_match(job, score=0.81))
    got = store.get("1")
    assert got is not None
    assert got.funnel_status is FunnelStatus.MATCHED
    assert got.match_score == pytest.approx(0.81)
    assert got.matched_skills == ["python"]
    assert got.missing_skills == ["kubernetes"]


def test_upsert_match_on_fresh_job_inserts_as_matched(store: JobStore) -> None:
    """match may persist a job search never recorded — it should insert at 'matched'."""
    store.upsert_match(_match(_job(2)))
    got = store.get("https://linkedin.com/jobs/2")
    assert got is not None
    assert got.funnel_status is FunnelStatus.MATCHED


def test_rediscovery_does_not_downgrade_stage(store: JobStore) -> None:
    job = _job(1)
    store.upsert_match(_match(job))  # matched
    store.mark_tailored(job, tailored_resume_path="/out/t.txt")  # tailored
    store.upsert_job(job)  # re-discovered by a later search
    store.upsert_match(_match(job, score=0.5))  # re-matched
    got = store.get("1")
    assert got is not None
    # Neither re-discovery nor re-match pulls a tailored job back down the funnel.
    assert got.funnel_status is FunnelStatus.TAILORED
    assert got.match_score == pytest.approx(0.5)  # scores still refresh


def test_mark_tailored_sets_status_and_artifacts(store: JobStore) -> None:
    job = _job(1)
    store.mark_tailored(job, tailored_resume_path="/out/t.txt")
    got = store.get("1")
    assert got is not None
    assert got.funnel_status is FunnelStatus.TAILORED
    assert got.tailored_resume_path == "/out/t.txt"
    assert got.cover_letter_path == ""


def test_mark_tailored_with_cover_letter_advances_to_cover_letter(store: JobStore) -> None:
    job = _job(1)
    store.mark_tailored(job, tailored_resume_path="/out/t.txt", cover_letter_path="/out/cl.txt")
    got = store.get("1")
    assert got is not None
    assert got.funnel_status is FunnelStatus.COVER_LETTER
    assert got.cover_letter_path == "/out/cl.txt"


def test_source_query_preserved_on_rediscovery(store: JobStore) -> None:
    store.upsert_job(_job(1), source_query="python remote")
    store.upsert_job(_job(1))  # re-seen with no query
    got = store.get("1")
    assert got is not None
    assert got.source_query == "python remote"


def test_counts_by_stage(store: JobStore) -> None:
    store.upsert_job(_job(1))  # found
    store.upsert_match(_match(_job(2)))  # matched
    store.mark_tailored(_job(3), tailored_resume_path="/out/3.txt")  # tailored
    assert store.counts() == {"found": 1, "matched": 1, "tailored": 1}


def test_list_jobs_filter_and_limit(store: JobStore) -> None:
    store.upsert_job(_job(1))
    store.upsert_match(_match(_job(2)))
    store.upsert_match(_match(_job(3)))
    assert len(store.list_jobs()) == 3
    matched = store.list_jobs(status=FunnelStatus.MATCHED)
    assert {s.job.company for s in matched} == {"Co2", "Co3"}
    assert len(store.list_jobs(limit=1)) == 1


def test_list_jobs_orders_newest_updated_first(store: JobStore) -> None:
    store.upsert_job(_job(1))
    store.upsert_job(_job(2))
    store.upsert_job(_job(1))  # re-touch #1 → its updated_at is now the latest
    ordered = store.list_jobs()
    assert ordered[0].job.company == "Co1"


def test_corrupt_db_raises_jobstoreerror(tmp_path: Path) -> None:
    bad = tmp_path / "applications.db"
    bad.write_text("not a sqlite database")
    with pytest.raises(JobStoreError):
        JobStore(db_path=bad)
