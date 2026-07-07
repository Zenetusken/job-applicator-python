"""Unit tests for the live selector-health diagnostic layer."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

import job_applicator.cli as cli
import job_applicator.selector_health as selector_health
from job_applicator.config import AppSettings
from job_applicator.jobs_store import JobStore
from job_applicator.models import (
    BoardSelectorHealth,
    JobBoard,
    JobListing,
    SelectorHealthReport,
    SelectorProbe,
    SelectorProbeResult,
    SelectorProbeStatus,
)
from job_applicator.selector_registry import APPLY_SURFACE, SEARCH_SURFACE, selector_probes


def _job(n: int = 1) -> JobListing:
    return JobListing(
        title=f"Engineer {n}",
        company=f"Co{n}",
        url=f"https://linkedin.com/jobs/{n}",
        description="async pipelines",
        location="Remote",
        board=JobBoard.LINKEDIN,
    )


def _browser_cm() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _report(status: SelectorProbeStatus) -> SelectorHealthReport:
    result = SelectorProbeResult(
        board=JobBoard.LINKEDIN,
        surface=SEARCH_SURFACE,
        name="job card containers",
        selector=".job-card-container",
        required=True,
        matched_count=1 if status == SelectorProbeStatus.PASSED else 0,
        status=status,
        details=".job-card-container=1",
        url="https://www.linkedin.com/jobs/search",
    )
    return SelectorHealthReport(
        status=status,
        generated_at=datetime.now(UTC),
        boards=[
            BoardSelectorHealth(
                board=JobBoard.LINKEDIN,
                surface=SEARCH_SURFACE,
                status=status,
                url=result.url,
                results=[result],
            )
        ],
    )


class _FakeTarget:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    async def query_selector_all(self, selector: str) -> list[object]:
        return [object()] * self._counts.get(selector, 0)


def test_registry_contains_required_selector_groups() -> None:
    linkedin_search = selector_probes(JobBoard.LINKEDIN, SEARCH_SURFACE)
    linkedin_apply = selector_probes(JobBoard.LINKEDIN, APPLY_SURFACE)
    indeed_search = selector_probes(JobBoard.INDEED, SEARCH_SURFACE)

    assert any(p.required and p.name == "job card containers" for p in linkedin_search)
    assert any(p.required and p.name == "title link" for p in linkedin_search)
    assert any(p.required and p.name == "Easy Apply button" for p in linkedin_apply)
    assert any((not p.required) and p.name == "submit buttons" for p in linkedin_apply)
    assert any(p.required and p.name == "job card containers" for p in indeed_search)
    assert any(p.required and p.name == "title link" for p in indeed_search)


async def test_probe_result_required_miss_fails_optional_miss_warns() -> None:
    required = SelectorProbe(
        board=JobBoard.LINKEDIN,
        surface=SEARCH_SURFACE,
        name="title link",
        selector=".missing-title",
        selectors=[".missing-title"],
        required=True,
    )
    optional = SelectorProbe(
        board=JobBoard.LINKEDIN,
        surface=SEARCH_SURFACE,
        name="salary",
        selector=".missing-salary",
        selectors=[".missing-salary"],
        required=False,
    )
    target = _FakeTarget({})

    required_result = await selector_health._evaluate_probe(
        required, target, url="https://example.test"
    )
    optional_result = await selector_health._evaluate_probe(
        optional, target, url="https://example.test"
    )

    assert required_result.status == SelectorProbeStatus.FAIL
    assert optional_result.status == SelectorProbeStatus.WARN
    assert selector_health.aggregate_status([required_result.status, optional_result.status]) == (
        SelectorProbeStatus.FAIL
    )


async def test_linkedin_form_controls_accept_advance_without_submit() -> None:
    target = _FakeTarget({'button:has-text("Next")': 1})

    result = await selector_health._linkedin_form_controls_result(
        target,  # type: ignore[arg-type]
        url="https://www.linkedin.com/jobs/view/1",
    )

    assert result.status == SelectorProbeStatus.PASSED
    assert result.required is True


async def test_linkedin_external_apply_is_skipped_not_failed() -> None:
    target = _FakeTarget({'button[aria-label^="Apply to" i]': 1})

    report = await selector_health._linkedin_external_apply_skip_report(
        target,  # type: ignore[arg-type]
        url="https://www.linkedin.com/jobs/view/1",
    )

    assert report is not None
    assert report.status == SelectorProbeStatus.SKIPPED
    assert report.ok is True
    assert {result.status for result in report.boards[0].results} == {SelectorProbeStatus.SKIPPED}


async def test_failure_diagnostics_are_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_write(
        report: SelectorHealthReport,
        page: object,
    ) -> list[str]:
        assert report.status == SelectorProbeStatus.FAIL
        assert page is None
        return ["/tmp/selector-health.txt"]

    monkeypatch.setattr(selector_health, "write_failure_diagnostics", _fake_write)
    service = selector_health.SelectorHealthService(MagicMock(), AppSettings())
    attached = await service._attach_failure_artifacts(_report(SelectorProbeStatus.FAIL), None)

    assert attached.artifacts == ["/tmp/selector-health.txt"]
    assert attached.boards[0].artifacts == ["/tmp/selector-health.txt"]
    assert attached.boards[0].results[0].artifacts == ["/tmp/selector-health.txt"]


def test_failure_diagnostic_summary_file_is_written(tmp_path: Path) -> None:
    artifacts = asyncio.run(
        selector_health.write_failure_diagnostics(
            _report(SelectorProbeStatus.FAIL), page=None, debug_dir=tmp_path
        )
    )

    assert len(artifacts) == 1
    summary = Path(artifacts[0])
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "status: fail" in text
    assert "job card containers" in text


def test_selector_health_json_stdout_is_pure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_report(**_kwargs: object) -> SelectorHealthReport:
        return _report(SelectorProbeStatus.PASSED)

    monkeypatch.setattr(cli, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(cli, "_run_selector_health_report", _fake_report)

    result = CliRunner().invoke(
        cli.app,
        ["selector-health", "--site", "linkedin", "--surface", "search", "-q", "python", "--json"],
    )

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed["status"] == "pass"
    assert "Selector health" not in result.stdout


def test_search_selector_health_failure_aborts_before_scrape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_report(**_kwargs: object) -> SelectorHealthReport:
        return _report(SelectorProbeStatus.FAIL)

    scraper = MagicMock(scrape=AsyncMock(return_value=[_job()]))
    monkeypatch.setattr(cli, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(cli, "_make_scraper", lambda *a, **k: scraper)
    monkeypatch.setattr(cli, "_run_selector_health_report", _fake_report)

    result = CliRunner().invoke(cli.app, ["search", "-q", "python", "--selector-health", "--json"])

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    scraper.scrape.assert_not_awaited()
    assert "Selector health preflight failed" in result.stderr


def test_apply_selector_health_failure_aborts_before_form_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_report(**_kwargs: object) -> SelectorHealthReport:
        return _report(SelectorProbeStatus.FAIL)

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job())
    make_applicator = MagicMock()
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: store)
    monkeypatch.setattr(cli, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(cli, "_make_applicator", make_applicator)
    monkeypatch.setattr(cli, "_run_selector_health_report", _fake_report)

    result = CliRunner().invoke(
        cli.app,
        ["apply", "--from", "1", "--no-cover-letter", "--selector-health"],
    )

    assert result.exit_code == 1, result.output
    make_applicator.assert_not_called()
    assert "Selector health preflight failed" in result.stderr
