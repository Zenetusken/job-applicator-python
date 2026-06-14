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


def test_search_verbose_flag() -> None:
    result = runner.invoke(app, ["--verbose", "search", "--help"])
    assert result.exit_code == 0


def test_config_init_verbose() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["--verbose", "config-init", "--output", "config.toml"])
        assert result.exit_code == 0
        assert "Verbose Report" in result.output
