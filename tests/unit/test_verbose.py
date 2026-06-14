"""Tests for global --verbose and --log-file CLI flags."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from job_applicator.cli import app
from job_applicator.models import ResumeData

runner = CliRunner()


def test_verbose_flag_is_accepted() -> None:
    result = runner.invoke(app, ["--verbose", "ats-check", "--help"])
    assert result.exit_code == 0


def test_log_file_requires_verbose() -> None:
    result = runner.invoke(app, ["--log-file", "out.json", "ats-check", "--help"])
    assert result.exit_code != 0
    assert "verbose" in result.output.lower()


def test_ats_check_verbose_output(tmp_path: Path) -> None:
    resume = tmp_path / "resume.pdf"
    resume.write_text("dummy")
    with patch("job_applicator.documents.resume.ResumeLoader") as mock_loader:
        mock_loader.return_value.load.return_value = ResumeData(
            raw_text="John Doe\njohn@example.com\n555-1234\nSkills: Python",
            name="John Doe",
            email="john@example.com",
            phone="555-1234",
            skills=["Python"],
        )
        result = runner.invoke(app, ["--verbose", "ats-check", "--resume", str(resume)])
    assert result.exit_code == 0
    assert "Verbose Report" in result.output
