"""TUI shell tests — Pilot-driven, headless, account-safe.

The Textual app is driven via ``App.run_test()`` / ``Pilot`` (no real terminal). Launch
reads only the local SQLite store — never the account, a browser, or the LLM. Async
tests run under the project's ``asyncio_mode = auto``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from textual.widgets import DataTable
from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.config import AppSettings
from job_applicator.jobs_store import JobStore, JobStoreError
from job_applicator.models import JobBoard, JobListing
from job_applicator.tui.app import JobApplicatorApp


def _job(n: int, **over: object) -> JobListing:
    data: dict[str, object] = {
        "title": f"Engineer {n}",
        "company": f"Co{n}",
        "url": f"https://linkedin.com/jobs/{n}",
        "description": "async pipelines",
        "location": "Remote",
        "requirements": ["python"],
        "board": JobBoard.LINKEDIN,
    }
    data.update(over)
    return JobListing(**data)  # type: ignore[arg-type]


def _app(tmp_path: Path, *, seed: int = 2) -> JobApplicatorApp:
    store = JobStore(db_path=tmp_path / "applications.db")
    for i in range(1, seed + 1):
        store.upsert_job(_job(i), source_query="python")
    app_state = MagicMock()
    app_state.list_recent.return_value = []
    return JobApplicatorApp(
        settings=AppSettings(resume_path="/cv/r.pdf"), store=store, app_state=app_state
    )


# ----------------------------------------------------------------- Pilot tests
async def test_tui_mounts_and_lists_jobs(tmp_path: Path) -> None:
    app = _app(tmp_path, seed=3)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", DataTable)
        assert table.row_count == 3
        assert table.has_focus  # the job list owns focus, not the (hidden) filter Input
        assert app._current is not None  # detail follows the first row


async def test_tui_navigation_updates_detail(tmp_path: Path) -> None:
    app = _app(tmp_path, seed=2)
    async with app.run_test() as pilot:
        await pilot.pause()
        first = app._current
        await pilot.press("j")
        await pilot.pause()
        assert app._current is not None and app._current is not first


async def test_tui_filter_narrows_then_clears(tmp_path: Path) -> None:
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1, company="Acme"))
    store.upsert_job(_job(2, company="Globex"))
    app_state = MagicMock()
    app_state.list_recent.return_value = []
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", DataTable)
        assert table.row_count == 2
        await pilot.press("slash")
        await pilot.pause()
        await pilot.press("g", "l", "o", "b", "e", "x")
        await pilot.pause()
        from textual.widgets import Input

        assert app.query_one("#filter", Input).value == "globex"  # the "/" trigger didn't leak
        await pilot.press("enter")
        await pilot.pause()
        assert table.row_count == 1  # only Globex
        await pilot.press("escape")
        await pilot.pause()
        assert table.row_count == 2  # filter cleared


async def test_tui_empty_store(tmp_path: Path) -> None:
    store = JobStore(db_path=tmp_path / "applications.db")
    app_state = MagicMock()
    app_state.list_recent.return_value = []
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#joblist", DataTable).row_count == 0
        assert app._current is None


async def test_tui_refresh_picks_up_new_jobs(tmp_path: Path) -> None:
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app_state = MagicMock()
    app_state.list_recent.return_value = []
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", DataTable)
        assert table.row_count == 1
        store.upsert_job(_job(2))  # a new job lands in the store
        await pilot.press("r")
        await pilot.pause()
        assert table.row_count == 2


async def test_tui_launch_reads_only_local_state(tmp_path: Path) -> None:
    """Mounting touches ONLY the injected store/app_state — no browser/LLM/account."""
    store = MagicMock()
    store.list_jobs.return_value = []
    app_state = MagicMock()
    app_state.list_recent.return_value = []
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
    store.list_jobs.assert_called()
    app_state.list_recent.assert_called()


async def test_tui_store_error_is_shown_not_raised(tmp_path: Path) -> None:
    """A store read failure on load is surfaced in the UI, never crashes it."""
    store = MagicMock()
    store.list_jobs.side_effect = JobStoreError("database is locked")
    app_state = MagicMock()
    app_state.list_recent.return_value = []
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "database is locked" in app._load_error


# -------------------------------------- launch wiring (CliRunner stdout ≠ TTY)
def test_bare_invocation_non_tty_shows_help() -> None:
    """Bare `job-applicator` in a non-TTY (pipe/CI) prints help + exits 0 — no hung UI."""
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output
    assert "tui" in result.output


def test_tui_command_non_tty_clean_error() -> None:
    """`job-applicator tui` in a non-TTY gives a clean message, not a Textual crash."""
    result = CliRunner().invoke(cli.app, ["tui"])
    assert result.exit_code == 1
    assert "interactive terminal" in result.stderr
