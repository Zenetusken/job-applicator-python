"""Tests for global --verbose and --log-file CLI flags."""

from __future__ import annotations

from typer.testing import CliRunner

from job_applicator.cli import app

runner = CliRunner()


def test_verbose_flag_is_accepted() -> None:
    result = runner.invoke(app, ["--verbose", "ats-check", "--help"])
    assert result.exit_code == 0


def test_log_file_requires_verbose() -> None:
    result = runner.invoke(app, ["--log-file", "out.json", "ats-check", "--help"])
    assert result.exit_code != 0
    assert "verbose" in result.output.lower()
