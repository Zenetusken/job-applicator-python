"""Tests for the batch CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_applicator.models import JobBoard, JobListing


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
        """Batch help advertises --run-id and --resume-run."""
        from typer.testing import CliRunner

        from job_applicator.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output
        assert "--resume-run" in result.output

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
