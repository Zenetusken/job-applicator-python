"""Integration tests: the batch loop's status writes hit a REAL ``BatchState``.

``tests/unit/test_batch.py`` drives the batch loop with a *mocked* ``BatchState`` (asserting the
loop *calls* ``record_job``/``find_existing_run``), and ``tests/unit/test_batch_state.py`` tests
the store methods in isolation against real SQLite. Neither proves the real loop's writes actually
*persist* — a wrong ``run_id``, a status-enum serialization bug, or an uncommitted transaction
would satisfy the mock call-assertions yet lose the row.

These tests close that seam: drive the real ``batch`` CLI with a REAL ``BatchState`` (a known
``--run-id`` so no spec reconstruction is needed) and read the persisted status back via
``get_job_status``. Everything *except* the state store is faked; ``--no-cover-letter`` keeps the
fake env minimal, and the artifact-writer is patched (it's an I/O seam, not the state seam).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.batch_state import BatchJobStatus, BatchState
from job_applicator.cli import app
from job_applicator.embeddings.matching import MatchResult
from job_applicator.models import (
    BatchRunSpec,
    GroundingReport,
    JobBoard,
    JobListing,
    TailoredResume,
)

# Path-bearing URL on purpose: record_job writes str(job.url) and the test reads _JOB_URL back,
# so the two must be byte-identical. A bare-domain URL ("https://example.com") would normalize to
# a trailing slash through pydantic's HttpUrl and silently mismatch — a path segment avoids that.
_JOB_URL = "https://example.com/job1"
_RUN_ID = "integ-run"


@pytest.fixture
def jobs_file(tmp_path: Path) -> Path:
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "title": "Python Developer",
                    "company": "TechCorp",
                    "url": _JOB_URL,
                    "description": "Python, FastAPI",
                    "board": "linkedin",
                }
            ]
        )
    )
    return path


@pytest.fixture
def resume_file(tmp_path: Path) -> Path:
    path = tmp_path / "resume.txt"
    path.write_text("Jane Dev\njane@example.com\n555-0100\nSkills: Python, FastAPI")
    return path


@pytest.fixture
def real_batchstate_env(resume_file: Path, tmp_path: Path) -> Iterator[dict[str, object]]:
    """Fake everything the batch loop touches EXCEPT ``BatchState`` (real, on the shared tmp DB).

    ``tailor_verified`` is the loop's real call site (not ``tailor``); tests set it to succeed or
    raise. The artifact writer is patched so the state assertions don't depend on real rendering.
    """
    match = MatchResult(
        job=JobListing(
            title="Python Developer",
            company="TechCorp",
            url=_JOB_URL,
            description="Python, FastAPI",
            requirements=["Python", "FastAPI"],
            board=JobBoard.LINKEDIN,
        ),
        score=0.9,
        semantic_score=0.8,
        skill_score=0.7,
        matched_skills=["Python"],
        missing_skills=[],
        summary="good",
    )
    matcher = MagicMock()
    matcher.rank_jobs = AsyncMock(return_value=[match])
    matcher.match_resume_to_job = AsyncMock(return_value=match)

    tailored = MagicMock()
    tailored.tailored_text = "Jane Dev\njane@example.com\n555-0100\nTailored resume text"
    tailored.match_score = 0.9
    tailored.semantic_score = 0.8
    tailored.skill_score = 0.7
    tailored.matched_skills = ["Python"]
    tailored.missing_skills = []
    tailored.changes_summary = "summary"
    tailored.cover_letter_path = ""
    tailor = MagicMock()
    tailor.tailor_verified = AsyncMock(return_value=tailored)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch.object(cli, "_get_settings", return_value=MagicMock()) as mock_settings,
        patch("job_applicator.embeddings.matching.JobMatcher", return_value=matcher),
        patch("job_applicator.documents.resume_tailor.ResumeTailor", return_value=tailor),
        patch("job_applicator.cli._load_user_profile", return_value=MagicMock()),
        patch("job_applicator.cli._run_ats_preflight", return_value=MagicMock(score=1.0)),
        patch("job_applicator.cli._run_ats_post_tailor", return_value=MagicMock(score=1.0)),
        patch(
            "job_applicator.cli._write_tailored_artifacts",
            AsyncMock(return_value=("out.txt", None)),
        ),
    ):
        settings = mock_settings.return_value
        settings.resume_path = str(resume_file)
        settings.style_guide_path = ""
        settings.llm = MagicMock()
        settings.llm.model = "test"
        settings.llm.api_base = "http://test"
        settings.llm.temperature = 0.7
        settings.output_dir = str(output_dir)
        settings.ensure_output_dir.return_value = output_dir
        settings.output = MagicMock()
        settings.output.default_format = "txt"
        settings.output.resume_template = "modern"
        settings.output.cover_letter_template = "modern"
        settings.embedding = MagicMock()
        settings.log_level = "INFO"
        settings.browser = MagicMock()
        settings.browser.headless = True

        yield {"runner": CliRunner(), "tailor": tailor, "resume_file": resume_file}


def _run_batch(env: dict[str, object], jobs_file: Path) -> object:
    runner: CliRunner = env["runner"]  # type: ignore[assignment]
    return runner.invoke(
        app,
        [
            "batch",
            "--resume",
            str(env["resume_file"]),
            "--jobs-file",
            str(jobs_file),
            "--top-k",
            "1",
            "--no-cover-letter",
            "--run-id",
            _RUN_ID,
        ],
    )


def test_batch_run_persists_completed_to_real_store(
    real_batchstate_env: dict[str, object], jobs_file: Path
) -> None:
    """A successful batch job is really recorded COMPLETED in SQLite — proving the loop's write
    path (record_job with the right run_id + status) actually commits, which mock call-assertions
    cannot verify."""
    result = _run_batch(real_batchstate_env, jobs_file)
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]

    status = BatchState().get_job_status(_RUN_ID, _JOB_URL)
    assert status == BatchJobStatus.COMPLETED


def test_batch_failure_persists_failed_job_to_real_store(
    real_batchstate_env: dict[str, object], jobs_file: Path
) -> None:
    """When tailoring raises, the loop records a job-level FAILED in real SQLite (so a later
    --resume-run can retry it) — not a silently-dropped or wrongly-COMPLETED row."""
    tailor: MagicMock = real_batchstate_env["tailor"]  # type: ignore[assignment]
    tailor.tailor_verified = AsyncMock(side_effect=RuntimeError("tailor boom"))

    result = _run_batch(real_batchstate_env, jobs_file)
    assert result.exit_code != 0, result.output  # type: ignore[attr-defined]

    status = BatchState().get_job_status(_RUN_ID, _JOB_URL)
    assert status == BatchJobStatus.FAILED


def test_batch_resume_reuses_existing_cover_letter_from_tailored_meta(
    real_batchstate_env: dict[str, object], jobs_file: Path, tmp_path: Path
) -> None:
    """If a crash happens after the cover-letter file/meta write but before DB completion,
    resume-run should mark the job complete without regenerating the existing letter."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    tailored_path = output_dir / "tailored.txt"
    cover_path = output_dir / "cover.txt"
    tailored_path.write_text("Tailored resume text", encoding="utf-8")
    cover_path.write_text("Existing cover letter", encoding="utf-8")
    tailored = TailoredResume(
        original_path=str(real_batchstate_env["resume_file"]),
        tailored_text="Tailored resume text",
        job_title="Python Developer",
        job_company="TechCorp",
        job_url=_JOB_URL,
        match_score=0.9,
        semantic_score=0.8,
        skill_score=0.7,
        matched_skills=["Python"],
        missing_skills=[],
        changes_summary="summary",
        output_path=str(tailored_path),
        cover_letter_path=str(cover_path),
        grounding_report=GroundingReport(),
    )
    tailored_path.with_suffix(".meta.json").write_text(
        tailored.model_dump_json(indent=2), encoding="utf-8"
    )
    job = JobListing(
        title="Python Developer",
        company="TechCorp",
        url=_JOB_URL,
        description="Python, FastAPI",
        requirements=["Python", "FastAPI"],
        board=JobBoard.LINKEDIN,
    )
    spec = BatchRunSpec(
        site="linkedin",
        jobs_file=str(jobs_file),
        resume_path=str(real_batchstate_env["resume_file"]),
        top_k=1,
        min_score=0.0,
        cover_letter=True,
    )
    state = BatchState()
    state.start_run(spec, run_id=_RUN_ID)
    state.record_job(_RUN_ID, job, BatchJobStatus.TAILORED, resume_path=str(tailored_path))

    cl_generator = MagicMock()
    cl_generator.generate_verified_with_overlay = AsyncMock(
        side_effect=AssertionError("should not regenerate")
    )
    with patch(
        "job_applicator.documents.cover_letter.CoverLetterGenerator",
        return_value=cl_generator,
    ):
        runner: CliRunner = real_batchstate_env["runner"]  # type: ignore[assignment]
        result = runner.invoke(
            app,
            [
                "batch",
                "--resume",
                str(real_batchstate_env["resume_file"]),
                "--jobs-file",
                str(jobs_file),
                "--top-k",
                "1",
                "--run-id",
                _RUN_ID,
            ],
        )

    assert result.exit_code == 0, result.output
    cl_generator.generate_verified_with_overlay.assert_not_called()
    assert BatchState().get_job_status(_RUN_ID, _JOB_URL) == BatchJobStatus.COMPLETED
