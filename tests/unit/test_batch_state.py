"""Unit tests for the batch progress store."""

from __future__ import annotations

from pathlib import Path

from job_applicator.batch_state import BatchJobStatus, BatchState
from job_applicator.models import BatchRunSpec, JobBoard, JobListing


def _make_job(url: str = "https://linkedin.com/jobs/view/1") -> JobListing:
    return JobListing(
        title="Python Dev",
        company="Acme",
        url=url,
        board=JobBoard.LINKEDIN,
    )


def _spec(
    *,
    site: str = "linkedin",
    query: str | None = "python",
    jobs_file: str | None = None,
    resume_path: str = "/tmp/resume.pdf",
    top_k: int = 5,
    min_score: float = 0.0,
    cover_letter: bool = True,
) -> BatchRunSpec:
    return BatchRunSpec(
        site=site,
        query=query,
        jobs_file=jobs_file,
        resume_path=resume_path,
        top_k=top_k,
        min_score=min_score,
        cover_letter=cover_letter,
    )


def test_start_run_and_record_job(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(_spec(min_score=0.5), run_id="run-1")
    assert run_id == "run-1"

    job = _make_job()
    state.record_job(run_id, job, BatchJobStatus.TAILORED, resume_path="/tmp/tailored.txt")

    assert state.get_job_status(run_id, str(job.url)) == BatchJobStatus.TAILORED


def test_find_existing_run(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    state.start_run(_spec(), run_id="run-2")

    assert state.find_existing_run(_spec()) == "run-2"
    assert state.find_existing_run(_spec(site="indeed")) is None


def test_completed_jobs_filtered(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(_spec(query=None, jobs_file="/tmp/jobs.json"), run_id="run-3")
    done = _make_job("https://linkedin.com/jobs/view/1")
    pending = _make_job("https://linkedin.com/jobs/view/2")
    state.record_job(run_id, done, BatchJobStatus.COMPLETED)
    state.record_job(run_id, pending, BatchJobStatus.PENDING)

    completed = state.list_completed_jobs(run_id)
    assert str(done.url) in completed
    assert str(pending.url) not in completed


def test_complete_run(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    spec = _spec(query=None, cover_letter=False)
    run_id = state.start_run(spec, run_id="run-4")
    state.complete_run(run_id)

    # After completion the run should no longer be found as running.
    assert state.find_existing_run(spec) is None


def test_start_run_with_reset_false_preserves_jobs(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    spec = _spec(query=None, cover_letter=False)
    run_id = state.start_run(spec, run_id="run-5")
    job = _make_job("https://linkedin.com/jobs/view/9")
    state.record_job(run_id, job, BatchJobStatus.TAILORED)

    # Re-starting without reset must keep the recorded job.
    state.start_run(spec, run_id=run_id, reset=False)
    assert state.get_job_status(run_id, str(job.url)) == BatchJobStatus.TAILORED


def test_find_existing_run_requires_matching_params(tmp_path: Path) -> None:
    """run_id alignment: a run with a different top_k is NOT matched, so a resume
    can't bind a run created with different processing params and adopt new ones."""
    state = BatchState(db_path=tmp_path / "batch.db")
    state.start_run(_spec(), run_id="run-6")
    assert state.find_existing_run(_spec(top_k=10)) is None
    assert state.find_existing_run(_spec(top_k=5)) == "run-6"


def test_batch_run_spec_run_id_deterministic_and_param_sensitive() -> None:
    """Item 4: BatchRunSpec.run_id() is the single source for run identity — same
    params → same id; a changed processing param → a different id (so find_existing_run,
    which matches the same fields, can never drift from the id)."""
    assert _spec().run_id() == _spec().run_id()
    assert _spec().run_id() != _spec(top_k=10).run_id()
    assert _spec().run_id() != _spec(query="rust").run_id()
    assert len(_spec().run_id()) == 16
