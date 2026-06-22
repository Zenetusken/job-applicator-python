"""CLI behaviour tests for `import-cookies` per-site spec gating.

These never hit a browser or network: the indeed path has no feed-verify, and
the error paths fail before saving. COOKIE_PATH on both scrapers is redirected
to a tmp dir so a test can never clobber the user's real session cookies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from job_applicator.cli import app
from job_applicator.scrapers.indeed import IndeedScraper
from job_applicator.scrapers.linkedin import LinkedInScraper

runner = CliRunner()


@pytest.fixture
def isolated_cookie_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(LinkedInScraper, "COOKIE_PATH", tmp_path / "linkedin.json")
    monkeypatch.setattr(IndeedScraper, "COOKIE_PATH", tmp_path / "indeed.json")
    return tmp_path


def _write_cookie_file(path: Path, cookies: list[dict[str, object]]) -> str:
    path.write_text(json.dumps({"cookies": cookies}))
    return str(path)


def test_indeed_import_warns_without_cf_clearance(isolated_cookie_paths: Path) -> None:
    src = _write_cookie_file(
        isolated_cookie_paths / "src.json",
        [{"name": "CTK", "value": "x", "domain": ".indeed.com", "path": "/"}],
    )
    result = runner.invoke(app, ["import-cookies", "--site", "indeed", "--file", src])
    assert result.exit_code == 0, result.output
    assert "cf_clearance" in result.output  # soft warning, not a hard failure
    assert (isolated_cookie_paths / "indeed.json").exists()  # still saved


def test_indeed_import_succeeds_with_cf_clearance(isolated_cookie_paths: Path) -> None:
    src = _write_cookie_file(
        isolated_cookie_paths / "src.json",
        [{"name": "cf_clearance", "value": "abc", "domain": ".indeed.com", "path": "/"}],
    )
    result = runner.invoke(app, ["import-cookies", "--site", "indeed", "--file", src])
    assert result.exit_code == 0, result.output
    assert "cf_clearance" not in result.output  # no missing-cookie warning
    assert (isolated_cookie_paths / "indeed.json").exists()


def test_indeed_rejects_linkedin_only_flags(isolated_cookie_paths: Path) -> None:
    result = runner.invoke(app, ["import-cookies", "--site", "indeed", "--li-at", "x"])
    assert result.exit_code == 1
    assert "LinkedIn-only" in result.output


def test_linkedin_requires_li_at(isolated_cookie_paths: Path) -> None:
    src = _write_cookie_file(
        isolated_cookie_paths / "src.json",
        [{"name": "bcookie", "value": "x", "domain": ".linkedin.com", "path": "/"}],
    )
    result = runner.invoke(app, ["import-cookies", "--site", "linkedin", "--file", src])
    assert result.exit_code == 1
    assert "li_at" in result.output
    assert not (isolated_cookie_paths / "linkedin.json").exists()  # not saved on failure


def test_unsupported_site_rejected(isolated_cookie_paths: Path) -> None:
    result = runner.invoke(app, ["import-cookies", "--site", "glassdoor", "--li-at", "x"])
    assert result.exit_code == 1
    assert "Unsupported site" in result.output


def test_help_shows_browser_extra_name_not_eaten_by_markup() -> None:
    """--help must show the 'browser' extra name; Rich previously ate `[browser]` → 'the  extra'."""
    result = runner.invoke(app, ["import-cookies", "--help"])
    assert result.exit_code == 0
    assert "the  extra" not in result.output  # the markup-eaten double-space gap is gone
    assert "'browser'" in result.output  # the quoted extra name renders (absent pre-fix)
