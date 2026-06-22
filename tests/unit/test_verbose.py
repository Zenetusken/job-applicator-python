"""Tests for global --verbose and --log-file CLI flags."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_json_verbose_keeps_stdout_pure_json(tmp_path: Path) -> None:
    """`--json --verbose`: the verbose observability report renders to STDERR (err_console),
    so stdout stays parseable JSON. Asserts `result.stdout` (stdout-only in click 8.4;
    `result.output` MERGES both streams and so can't see the split) — the report previously
    polluted stdout and broke `<cmd> --json | jq`.
    """
    from docx import Document

    resume = tmp_path / "r.docx"
    doc = Document()
    doc.add_paragraph("Jane Dev\njane@example.com\n(555) 111-2222\nExperience\nSkills: Python")
    doc.save(str(resume))

    result = runner.invoke(app, ["ats-check", "--resume", str(resume), "--json", "--verbose"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)  # raises if the verbose report leaked onto stdout
    assert "score" in parsed
    assert "Verbose Report" in result.stderr  # the report lives on stderr, where logs belong
