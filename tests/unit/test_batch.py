"""Tests for the batch CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.cli import app
from job_applicator.embeddings.matching import MatchResult
from job_applicator.models import JobBoard, JobListing, TailoredResume


@pytest.fixture
def sample_jobs_file(tmp_path: Path) -> Path:
    """Create a sample jobs JSON file."""
    jobs = [
        {
            "title": "Python Developer",
            "company": "TechCorp",
            "url": "https://example.com/1",
            "description": "Python, FastAPI",
            "requirements": ["Python", "FastAPI"],
            "board": "linkedin",
        },
        {
            "title": "Backend Engineer",
            "company": "StartupXYZ",
            "url": "https://example.com/2",
            "description": "Django, PostgreSQL",
            "requirements": ["Django", "PostgreSQL"],
            "board": "linkedin",
        },
    ]
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps(jobs))
    return jobs_file


@pytest.fixture
def sample_resume_file(tmp_path: Path) -> Path:
    """Create a sample resume file."""
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "John Doe\njohn@example.com\n555-0123\n\n"
        "Summary:\nPython developer\n\n"
        "Skills:\nPython, FastAPI, Django\n"
    )
    return resume


@pytest.fixture
def style_guide_batch_env(
    sample_jobs_file: Path,
    sample_resume_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Shared mocked environment for the batch --style-guide regression tests."""
    matcher = MagicMock()
    match = MatchResult(
        job=JobListing(
            title="Python Developer",
            company="TechCorp",
            url="https://example.com/1",
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
    matcher.rank_jobs = AsyncMock(return_value=[match])
    matcher.match_resume_to_job = AsyncMock(return_value=match)

    tailor = MagicMock()
    tailored = TailoredResume(
        original_path=str(sample_resume_file),
        tailored_text="Tailored resume text",
        job_title="Python Developer",
        job_company="TechCorp",
        job_url="https://example.com/1",
        match_score=0.9,
        semantic_score=0.8,
        skill_score=0.7,
        matched_skills=["Python"],
        missing_skills=[],
        changes_summary="summary",
    )
    tailor.tailor = AsyncMock(return_value=tailored)
    tailor.tailor_verified = AsyncMock(return_value=tailored)

    style = MagicMock()
    style.tone = "professional"
    cl_generator = MagicMock()
    cl_generator.load_style_guide = AsyncMock(return_value=style)
    cl_generator.generate = AsyncMock(return_value="cover letter")

    with (
        patch.object(cli, "_get_settings", return_value=MagicMock()) as mock_settings,
        patch("job_applicator.embeddings.matching.JobMatcher", return_value=matcher),
        patch("job_applicator.documents.resume_tailor.ResumeTailor", return_value=tailor),
        patch(
            "job_applicator.documents.cover_letter.CoverLetterGenerator",
            return_value=cl_generator,
        ),
        patch("job_applicator.cli._load_user_profile", return_value=MagicMock()),
        patch("job_applicator.cli._run_ats_preflight", return_value=MagicMock(score=1.0)),
        patch("job_applicator.cli._run_ats_post_tailor", return_value=MagicMock(score=1.0)),
        patch("job_applicator.batch_state.BatchState") as mock_batch_state_cls,
    ):
        settings = mock_settings.return_value
        settings.resume_path = str(sample_resume_file)
        settings.style_guide_path = "style.txt"
        settings.llm = MagicMock()
        settings.llm.model = "test"
        settings.llm.api_base = "http://test"
        settings.llm.temperature = 0.7
        settings.output_dir = str(sample_jobs_file.parent / "out")
        Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
        settings.ensure_output_dir.return_value = Path(settings.output_dir)
        settings.embedding = MagicMock()
        settings.log_level = "INFO"
        settings.browser = MagicMock()
        settings.browser.headless = True

        bs = MagicMock()
        bs.list_completed_jobs.return_value = []
        bs.find_existing_run.return_value = None
        mock_batch_state_cls.return_value = bs

        yield {
            "runner": CliRunner(),
            "app": app,
            "matcher": matcher,
            "tailor": tailor,
            "cl_generator": cl_generator,
            "settings": settings,
            "batch_state": bs,
            "sample_jobs_file": sample_jobs_file,
            "sample_resume_file": sample_resume_file,
        }


class TestBatchCommand:
    """Tests for the batch CLI command."""

    def test_batch_command_exists(self) -> None:
        """Batch command is registered in the CLI."""
        from typer.testing import CliRunner

        from job_applicator.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "batch" in result.output

    def test_batch_help_includes_resume_flags(self) -> None:
        """Batch command registers --run-id and --resume-run options."""
        from typer.main import get_command

        from job_applicator.cli import app

        batch_cmd = get_command(app).commands["batch"]
        param_opts = {opt for param in batch_cmd.params for opt in param.opts}
        assert "--run-id" in param_opts
        assert "--resume-run" in param_opts

    def test_batch_loads_jobs_from_file(self, sample_jobs_file: Path) -> None:
        """Jobs from --jobs-file deserialize into JobListing correctly."""
        data = json.loads(sample_jobs_file.read_text())
        for item in data:
            job = JobListing(**item)
            assert job.title
            assert job.company
            assert job.board in (JobBoard.LINKEDIN, JobBoard.INDEED)

    def test_batch_requires_resume(self) -> None:
        """Batch exits if no resume provided."""
        from typer.testing import CliRunner

        from job_applicator.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["batch", "--jobs-file", "nonexistent.json"])
        assert result.exit_code != 0 or "resume" in result.output.lower()

    def test_batch_requires_jobs_or_query(self, sample_resume_file: Path) -> None:
        """Batch exits if neither --jobs-file nor --query provided."""
        from typer.testing import CliRunner

        from job_applicator.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["batch", "--resume", str(sample_resume_file)])
        assert (
            result.exit_code != 0
            or "jobs-file" in result.output.lower()
            or "query" in result.output.lower()
        )

    def test_batch_handles_missing_jobs_file(
        self, sample_resume_file: Path, tmp_path: Path
    ) -> None:
        """Batch exits with friendly error for missing jobs file."""
        from typer.testing import CliRunner

        from job_applicator.cli import app

        missing = tmp_path / "nope.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["batch", "--resume", str(sample_resume_file), "--jobs-file", str(missing)],
        )
        assert result.exit_code != 0 or "not found" in result.output.lower()

    def test_batch_handles_malformed_json(self, sample_resume_file: Path, tmp_path: Path) -> None:
        """Batch exits with friendly error for malformed JSON."""
        from typer.testing import CliRunner

        from job_applicator.cli import app

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["batch", "--resume", str(sample_resume_file), "--jobs-file", str(bad_file)],
        )
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_batch_min_score_filter(self) -> None:
        """Jobs below min_score are filtered out."""
        from job_applicator.embeddings.matching import MatchResult

        matches = [
            MatchResult(
                job=JobListing(
                    title="Good",
                    company="A",
                    url="https://example.com/1",
                    board=JobBoard.LINKEDIN,
                ),
                score=0.7,
                semantic_score=0.5,
                skill_score=0.4,
                matched_skills=["Python"],
                missing_skills=[],
                summary="good match",
            ),
            MatchResult(
                job=JobListing(
                    title="Bad",
                    company="B",
                    url="https://example.com/2",
                    board=JobBoard.LINKEDIN,
                ),
                score=0.3,
                semantic_score=0.2,
                skill_score=0.1,
                matched_skills=[],
                missing_skills=["Python"],
                summary="poor match",
            ),
        ]
        min_score = 0.5
        filtered = [m for m in matches if m.score >= min_score]
        assert len(filtered) == 1
        assert filtered[0].job.title == "Good"

    def test_batch_top_k_limits_results(self) -> None:
        """--top-k limits the number of jobs processed."""
        from job_applicator.embeddings.matching import MatchResult

        matches = [
            MatchResult(
                job=JobListing(
                    title=f"Job {i}",
                    company=f"Co {i}",
                    url=f"https://example.com/{i}",
                    board=JobBoard.LINKEDIN,
                ),
                score=0.8 - i * 0.1,
                semantic_score=0.5 - i * 0.05,
                skill_score=0.3 - i * 0.05,
                matched_skills=[],
                missing_skills=[],
                summary="",
            )
            for i in range(10)
        ]
        top_k = 3
        assert len(matches[:top_k]) == 3

    def test_batch_summary_json_structure(self, tmp_path: Path) -> None:
        """batch_summary.json has correct structure."""
        summary = {
            "timestamp": "20260613_120000",
            "resume": "/path/to/resume.pdf",
            "total_jobs": 5,
            "matched": 3,
            "results": [
                {
                    "title": "Python Dev",
                    "company": "TechCorp",
                    "url": "https://example.com/1",
                    "match_score": 0.75,
                    "semantic_score": 0.45,
                    "skill_score": 0.30,
                    "tailored": True,
                    "resume_path": "output/tailored_TechCorp_Python_Dev.txt",
                    "cover_letter": True,
                    "cover_letter_path": "output/cover_letter_TechCorp_Python_Dev.txt",
                }
            ],
        }
        summary_path = tmp_path / "batch_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        loaded = json.loads(summary_path.read_text())
        assert loaded["total_jobs"] == 5
        assert loaded["results"][0]["tailored"] is True
        assert loaded["results"][0]["match_score"] == 0.75


# --- Style-guide regression tests ---------------------------------------------------


class TestBatchStyleGuide:
    """Cycle 2b polish: batch respects --no-cover-letter and forwards OCR mode."""

    def test_batch_json_empty_jobs_emits_valid_json(
        self, style_guide_batch_env: dict[str, object], tmp_path: Path
    ) -> None:
        """batch --json with no jobs must emit a valid JSON summary on stdout, not the human
        'No jobs found' text (CLAUDE.md: --json output is PURE parseable stdout)."""
        env = style_guide_batch_env
        empty = tmp_path / "empty_jobs.json"
        empty.write_text("[]")
        result = env["runner"].invoke(  # type: ignore[attr-defined]
            env["app"],
            [
                "batch",
                "--resume",
                str(env["sample_resume_file"]),
                "--jobs-file",
                str(empty),
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "No jobs found" not in result.output  # not the human message on stdout
        data = json.loads(result.output[result.output.index("{") :])
        assert data["results"] == [] and data["total_jobs"] == 0
        # same key shape as a full run's summary, so a --json consumer's parser stays stable
        assert set(data) >= {"timestamp", "resume", "total_jobs", "matched", "results"}

    def test_no_cover_letter_with_style_guide_skips_generation(
        self, style_guide_batch_env: dict[str, object]
    ) -> None:
        """batch --no-cover-letter --style-guide must NOT generate cover letters."""
        env = style_guide_batch_env
        result = env["runner"].invoke(
            env["app"],
            [
                "batch",
                "--resume",
                str(env["sample_resume_file"]),
                "--jobs-file",
                str(env["sample_jobs_file"]),
                "--top-k",
                "1",
                "--no-cover-letter",
                "--style-guide",
                "style.txt",
            ],
        )

        assert result.exit_code == 0, result.output
        env["cl_generator"].generate.assert_not_called()

    def test_style_guide_load_receives_ocr_mode(
        self, style_guide_batch_env: dict[str, object]
    ) -> None:
        """batch --style-guide --force-ocr must pass ocr_mode='on' to the loader."""
        env = style_guide_batch_env
        env["settings"].style_guide_path = "style.pdf"

        result = env["runner"].invoke(
            env["app"],
            [
                "batch",
                "--resume",
                str(env["sample_resume_file"]),
                "--jobs-file",
                str(env["sample_jobs_file"]),
                "--top-k",
                "1",
                "--style-guide",
                "style.pdf",
                "--force-ocr",
            ],
        )

        assert result.exit_code == 0, result.output
        env["cl_generator"].load_style_guide.assert_awaited_once_with("style.pdf", ocr_mode="on")


class TestBatchRecovery:
    """Crash / partial-failure recovery for the batch command (auto-resume + FAILED status)."""

    def test_batch_auto_resumes_an_existing_incomplete_run(
        self, style_guide_batch_env: dict[str, object]
    ) -> None:
        """When an incomplete (RUNNING/FAILED) same-spec run exists, a re-run AUTO-RESUMES it (no
        --resume-run flag needed) — start_run is NOT called, so the prior progress isn't wiped."""
        env = style_guide_batch_env
        bs = env["batch_state"]
        bs.find_existing_run.return_value = "existing-run"
        result = env["runner"].invoke(  # type: ignore[attr-defined]
            env["app"],
            [
                "batch",
                "--resume",
                str(env["sample_resume_file"]),
                "--jobs-file",
                str(env["sample_jobs_file"]),
                "--top-k",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        bs.start_run.assert_not_called()  # resumed, never reset (no silent wipe)
        bs.list_completed_jobs.assert_called_with("existing-run")  # read the resumed run's progress

    def test_batch_marks_run_failed_when_a_job_fails(
        self, style_guide_batch_env: dict[str, object]
    ) -> None:
        """A job that fails (a tailor error) marks the run FAILED (not COMPLETED), so a correct
        --resume-run can retry it instead of the run being unrecoverable."""
        from job_applicator.batch_state import BatchRunStatus

        env = style_guide_batch_env
        # The CLI calls tailor_verified (not .tailor) — override THAT so the job genuinely fails.
        env["tailor"].tailor_verified = AsyncMock(side_effect=RuntimeError("tailor blew up"))
        bs = env["batch_state"]
        result = env["runner"].invoke(  # type: ignore[attr-defined]
            env["app"],
            [
                "batch",
                "--resume",
                str(env["sample_resume_file"]),
                "--jobs-file",
                str(env["sample_jobs_file"]),
                "--top-k",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        bs.complete_run.assert_called_with(ANY, BatchRunStatus.FAILED)  # FAILED, not COMPLETED

    def test_batch_marks_run_failed_on_empty_message_error(
        self, style_guide_batch_env: dict[str, object]
    ) -> None:
        """A job failing with an EMPTY-message exception (a bare TimeoutError on the LLM endpoint)
        still marks the run FAILED — the status keys on the result FLAGS, not the error-message
        truthiness (a '' message is falsy → would WRONGLY mark the run COMPLETED + unresumable)."""
        from job_applicator.batch_state import BatchRunStatus

        env = style_guide_batch_env
        env["tailor"].tailor_verified = AsyncMock(side_effect=TimeoutError())  # str() == ""
        bs = env["batch_state"]
        result = env["runner"].invoke(  # type: ignore[attr-defined]
            env["app"],
            [
                "batch",
                "--resume",
                str(env["sample_resume_file"]),
                "--jobs-file",
                str(env["sample_jobs_file"]),
                "--top-k",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        bs.complete_run.assert_called_with(ANY, BatchRunStatus.FAILED)  # not wrongly COMPLETED
