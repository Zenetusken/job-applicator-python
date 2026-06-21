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


def _write_tailored(tmp_path: Path, text: str = "TAILORED RESUME TEXT") -> tuple[Path, Path]:
    from job_applicator.models import TailoredResume

    resume_path = tmp_path / "tailored_x.txt"
    resume_path.write_text(text)
    tailored = TailoredResume(
        original_path="",
        tailored_text=text,
        job_title="Python Dev",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.7,
        skill_score=0.9,
        changes_summary="reordered skills",
        output_path=str(resume_path),
    )
    meta_path = resume_path.with_suffix(".meta.json")
    meta_path.write_text(tailored.model_dump_json())
    return resume_path, meta_path


def test_resume_tailored_resume_reconstructs_from_meta(tmp_path: Path) -> None:
    """Cycle 3b: a persisted TAILORED job with a readable meta.json reconstructs its
    TailoredResume — validates the model_dump_json → model_validate_json round-trip
    (incl. the datetime field) that the mid-job-resume reuse path depends on."""
    from job_applicator.cli import _resume_tailored_resume

    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(_spec(), run_id="run-r")
    job = _make_job()
    resume_path, meta_path = _write_tailored(tmp_path)
    state.record_job(run_id, job, BatchJobStatus.TAILORED, resume_path=str(resume_path))

    reused = _resume_tailored_resume(state, run_id, str(job.url))
    assert reused is not None
    got, got_resume, got_meta = reused
    assert got.tailored_text == "TAILORED RESUME TEXT"
    assert got.match_score == 0.8
    assert got_resume == str(resume_path)
    assert got_meta == str(meta_path)


def test_resume_tailored_resume_none_when_not_reusable(tmp_path: Path) -> None:
    """Not reused unless the job is TAILORED with an artifact: absent → None;
    COMPLETED → None (it's already in list_completed_jobs and won't be re-processed)."""
    from job_applicator.cli import _resume_tailored_resume

    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(_spec(), run_id="run-r2")
    job = _make_job()
    assert _resume_tailored_resume(state, run_id, str(job.url)) is None  # no record

    _write_tailored(tmp_path)
    state.record_job(run_id, job, BatchJobStatus.COMPLETED, resume_path="/tmp/whatever.txt")
    assert _resume_tailored_resume(state, run_id, str(job.url)) is None  # not TAILORED


def test_resume_tailored_resume_none_when_meta_missing_or_corrupt(tmp_path: Path) -> None:
    """TAILORED but meta.json missing or corrupt → None, so the caller re-tailors
    rather than reusing a stale/broken artifact."""
    from job_applicator.cli import _resume_tailored_resume

    state = BatchState(db_path=tmp_path / "batch.db")
    run_id = state.start_run(_spec(), run_id="run-r3")
    job = _make_job()
    resume_path = tmp_path / "tailored_y.txt"
    resume_path.write_text("text")

    state.record_job(run_id, job, BatchJobStatus.TAILORED, resume_path=str(resume_path))
    assert _resume_tailored_resume(state, run_id, str(job.url)) is None  # no meta.json

    resume_path.with_suffix(".meta.json").write_text("{not valid json")
    assert _resume_tailored_resume(state, run_id, str(job.url)) is None  # corrupt meta.json
