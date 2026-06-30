"""Track D — owner-only permissions on written artifacts (privacy / file hygiene).

The verbose log can hold résumé-derived PII, config.toml can hold credentials, and output/
holds the user's tailored documents — so each is written owner-only (0600 file / 0700 dir)."""

from __future__ import annotations

import stat
from pathlib import Path

from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.config import AppSettings
from job_applicator.utils.path import set_owner_only
from job_applicator.utils.verbose import VerboseReporter


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_set_owner_only_sets_mode(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text("x")
    set_owner_only(f, 0o600)
    assert _mode(f) == 0o600


def test_set_owner_only_is_best_effort_on_missing_path(tmp_path: Path) -> None:
    """A chmod failure (here: a missing path) must NOT raise — it's hygiene, not load-bearing."""
    set_owner_only(tmp_path / "missing", 0o600)  # no raise


def test_ensure_output_dir_is_owner_only(app_settings: AppSettings, tmp_path: Path) -> None:
    app_settings.output_dir = str(tmp_path / "out")
    out = app_settings.ensure_output_dir()
    assert _mode(out) == 0o700  # tailored résumés / cover letters are the user's data


def test_verbose_log_file_is_owner_only(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    VerboseReporter(command="x", args={}, config={}).render(console=None, log_file=str(log))
    assert log.exists()
    assert _mode(log) == 0o600  # the report can hold résumé-derived PII


def test_config_init_writes_owner_only_config(tmp_path: Path) -> None:
    out = tmp_path / "config.toml"
    result = CliRunner().invoke(cli.app, ["config-init", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert _mode(out) == 0o600  # may later hold credentials / api keys
