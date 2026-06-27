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


def test_list_jobs_tiebreaks_same_recency_by_score_desc(tmp_path: Path) -> None:
    """Within the same updated_at batch, list_jobs orders best-score-first. Recency-DESC
    is unchanged; match_score-DESC only breaks ties among same-recency rows (so a batch of
    jobs scored together reads best-first instead of in arbitrary insertion order)."""
    import sqlite3

    db = tmp_path / "applications.db"
    store = JobStore(db_path=db)
    low, high = _job(1), _job(2)
    store.upsert_job(low)
    store.upsert_match(_match(low, score=0.40))
    store.upsert_job(high)
    store.upsert_match(_match(high, score=0.90))
    # Force identical recency so the score tiebreak (not insertion order) decides.
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE jobs SET updated_at = '2026-01-01 00:00:00'")
    conn.commit()
    conn.close()

    scores = [j.match_score for j in store.list_jobs()]
    assert scores == [0.90, 0.40], f"expected score-DESC within same recency, got {scores}"


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


def test_salary_not_clobbered_by_salaryless_rescrape(store: JobStore) -> None:
    """A re-write that lacks a salary must NOT wipe a previously-captured one — Indeed shows
    its salary teaser inconsistently across searches, so a later salary-less scrape of the same
    job would otherwise erase good data. Guards upsert_job / upsert_match / mark_tailored."""
    store.upsert_job(_job(1, salary="$120,000 a year"))
    assert store.get("1").job.salary == "$120,000 a year"  # type: ignore[union-attr]

    store.upsert_job(_job(1, salary=None))  # re-discovered, no salary on the card
    assert store.get("1").job.salary == "$120,000 a year"  # type: ignore[union-attr]
    store.upsert_match(_match(_job(1, salary=None)))  # scored, still no salary
    assert store.get("1").job.salary == "$120,000 a year"  # type: ignore[union-attr]
    store.mark_tailored(_job(1, salary=None), tailored_resume_path="/tmp/r.txt")
    assert store.get("1").job.salary == "$120,000 a year"  # type: ignore[union-attr]

    store.upsert_job(_job(1, salary="$140,000 a year"))  # a NEW salary still updates
    assert store.get("1").job.salary == "$140,000 a year"  # type: ignore[union-attr]


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


def test_mark_tailored_does_not_downgrade_cover_letter(store: JobStore) -> None:
    """A re-tailor without a cover letter must not pull a cover_letter job back to tailored."""
    job = _job(1)
    store.mark_tailored(job, tailored_resume_path="/t.txt", cover_letter_path="/cl.txt")
    store.mark_tailored(job, tailored_resume_path="/t2.txt")  # re-tailor, no cover letter
    got = store.get("1")
    assert got is not None
    assert got.funnel_status is FunnelStatus.COVER_LETTER  # stage preserved, not downgraded
    assert got.cover_letter_path == "/cl.txt"  # original cover-letter artifact kept
    assert got.tailored_resume_path == "/t2.txt"  # resume artifact refreshed


def test_rediscovery_preserves_rich_fields(store: JobStore) -> None:
    """A later thin re-search must not clobber a rich description/requirements."""
    rich = _job(
        1, description="Full async pipeline role", requirements=["python", "asyncio", "aws"]
    )
    store.upsert_match(_match(rich))
    store.upsert_job(_job(1, description="", requirements=[]))  # thin list-page re-discovery
    got = store.get("1")
    assert got is not None
    assert got.job.description == "Full async pipeline role"
    assert got.job.requirements == ["python", "asyncio", "aws"]


def test_get_corrupt_row_raises_typed_error(tmp_path: Path) -> None:
    """A row with an out-of-enum funnel_status surfaces a typed JobStoreError, not a raw crash."""
    import sqlite3

    p = tmp_path / "applications.db"
    store = JobStore(db_path=p)
    store.upsert_job(_job(1))
    conn = sqlite3.connect(str(p))
    conn.execute("UPDATE jobs SET funnel_status='archived' WHERE id=1")  # not a FunnelStatus
    conn.commit()
    conn.close()
    with pytest.raises(JobStoreError):
        store.get("1")


def test_list_jobs_filter_and_limit(store: JobStore) -> None:
    store.upsert_job(_job(1))
    store.upsert_match(_match(_job(2)))
    store.upsert_match(_match(_job(3)))
    assert len(store.list_jobs()) == 3
    matched = store.list_jobs(status=FunnelStatus.MATCHED)
    assert {s.job.company for s in matched} == {"Co2", "Co3"}
    assert len(store.list_jobs(limit=1)) == 1


def test_list_jobs_board_filter_before_limit(store: JobStore) -> None:
    """A board filter is applied in SQL before LIMIT, so it sees all matching rows even
    when newer rows of another board would otherwise fill the limit window."""
    store.upsert_job(_job(1))  # linkedin
    store.upsert_job(_job(2))  # linkedin
    store.upsert_job(_job(3, board=JobBoard.INDEED))  # indeed, newest-updated
    linkedin = store.list_jobs(board="linkedin", limit=2)
    assert len(linkedin) == 2  # both linkedin jobs — not hidden behind the newer indeed one
    assert all(s.job.board is JobBoard.LINKEDIN for s in linkedin)


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


def test_seniority_not_clobbered_by_rescrape(store: JobStore) -> None:
    """A re-write lacking seniority must NOT wipe a previously-detected one — the same data-loss
    class as the salary guard. Guards upsert_job / upsert_match / mark_tailored."""
    store.upsert_job(_job(1, seniority="senior"))
    assert store.get("1").job.seniority == "senior"  # type: ignore[union-attr]

    store.upsert_job(_job(1, seniority=None))  # list-card rescrape, no seniority
    assert store.get("1").job.seniority == "senior"  # type: ignore[union-attr]
    store.upsert_match(_match(_job(1, seniority=None)))  # scored, still none
    assert store.get("1").job.seniority == "senior"  # type: ignore[union-attr]
    store.mark_tailored(_job(1, seniority=None), tailored_resume_path="/tmp/r.txt")
    assert store.get("1").job.seniority == "senior"  # type: ignore[union-attr]

    store.upsert_job(_job(1, seniority="staff"))  # a NEW seniority still updates
    assert store.get("1").job.seniority == "staff"  # type: ignore[union-attr]


def test_stores_enable_wal_journal_mode(tmp_path: Path) -> None:
    """All three stores set WAL on init so readers (status / the TUI) don't block on a long
    batch/apply writer sharing the same DB file."""
    import sqlite3

    from job_applicator.batch_state import BatchState
    from job_applicator.state import ApplicationState

    for cls in (JobStore, ApplicationState, BatchState):
        db = tmp_path / f"{cls.__name__}.db"
        cls(db_path=db)  # construction runs _init_schema → sets WAL
        with sqlite3.connect(str(db)) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal", f"{cls.__name__} journal_mode={mode!r}"
