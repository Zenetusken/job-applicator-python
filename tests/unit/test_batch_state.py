"""Unit tests for the batch progress store."""

from __future__ import annotations

from pathlib import Path

from job_applicator.batch_state import BatchJobStatus, BatchState
from job_applicator.models import JobBoard, JobListing


def _make_job(url: str = "https://linkedin.com/jobs/view/1") -> JobListing:
    return JobListing(
        title="Python Dev",
        company="Acme",
        url=url,
        board=JobBoard.LINKEDIN,
    )


def test_start_run_and_record_job(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(
        run_id="run-1",
        site="linkedin",
        query="python",
        jobs_file=None,
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.5,
        cover_letter=True,
    )
    assert run_id == "run-1"

    job = _make_job()
    state.record_job(run_id, job, BatchJobStatus.TAILORED, resume_path="/tmp/tailored.txt")

    assert state.get_job_status(run_id, str(job.url)) == BatchJobStatus.TAILORED


def test_find_existing_run(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    state.start_run(
        run_id="run-2",
        site="linkedin",
        query="python",
        jobs_file=None,
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.0,
        cover_letter=True,
    )

    found = state.find_existing_run(
        site="linkedin",
        query="python",
        jobs_file=None,
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.0,
        cover_letter=True,
    )
    assert found == "run-2"

    not_found = state.find_existing_run(
        site="indeed",
        query="python",
        jobs_file=None,
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.0,
        cover_letter=True,
    )
    assert not_found is None


def test_completed_jobs_filtered(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(
        run_id="run-3",
        site="linkedin",
        query=None,
        jobs_file="/tmp/jobs.json",
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.0,
        cover_letter=False,
    )
    done = _make_job("https://linkedin.com/jobs/view/1")
    pending = _make_job("https://linkedin.com/jobs/view/2")
    state.record_job(run_id, done, BatchJobStatus.COMPLETED)
    state.record_job(run_id, pending, BatchJobStatus.PENDING)

    completed = state.list_completed_jobs(run_id)
    assert str(done.url) in completed
    assert str(pending.url) not in completed


def test_complete_run(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(
        run_id="run-4",
        site="linkedin",
        query=None,
        jobs_file=None,
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.0,
        cover_letter=False,
    )
    state.complete_run(run_id)

    # After completion the run should no longer be found as running.
    found = state.find_existing_run(
        site="linkedin",
        query=None,
        jobs_file=None,
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.0,
        cover_letter=False,
    )
    assert found is None


def test_start_run_with_reset_false_preserves_jobs(tmp_path: Path) -> None:
    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(
        run_id="run-5",
        site="linkedin",
        query=None,
        jobs_file=None,
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.0,
        cover_letter=False,
    )
    job = _make_job("https://linkedin.com/jobs/view/9")
    state.record_job(run_id, job, BatchJobStatus.TAILORED)

    # Re-starting without reset must keep the recorded job.
    state.start_run(
        run_id=run_id,
        site="linkedin",
        query=None,
        jobs_file=None,
        resume_path="/tmp/resume.pdf",
        top_k=5,
        min_score=0.0,
        cover_letter=False,
        reset=False,
    )
    assert state.get_job_status(run_id, str(job.url)) == BatchJobStatus.TAILORED
