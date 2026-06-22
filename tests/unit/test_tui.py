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


async def test_tui_launch_reads_only_local_state(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Mounting reads the injected store/app_state and constructs NO browser / scraper /
    applicator / LLM — the account-safety invariant, asserted on the negative too."""
    import litellm

    import job_applicator.factories as factories

    browser = MagicMock()
    scraper = MagicMock()
    applicator = MagicMock()
    acompletion = MagicMock()
    monkeypatch.setattr(factories, "_make_browser", browser)
    monkeypatch.setattr(factories, "_make_scraper", scraper)
    monkeypatch.setattr(factories, "_make_applicator", applicator)
    monkeypatch.setattr(litellm, "acompletion", acompletion)

    store = MagicMock()
    store.list_jobs.return_value = []
    app_state = MagicMock()
    app_state.list_recent.return_value = []
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
    store.list_jobs.assert_called()  # reads local state…
    app_state.list_recent.assert_called()
    browser.assert_not_called()  # …and nothing else
    scraper.assert_not_called()
    applicator.assert_not_called()
    acompletion.assert_not_called()


async def test_tui_applied_count_counts_only_submitted(tmp_path: Path) -> None:
    """The status line's 'applied' count reflects SUBMITTED applications only."""
    from datetime import UTC, datetime

    from job_applicator.models import ApplicationResult, ApplicationStatus

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app_state = MagicMock()
    app_state.list_recent.return_value = [
        ApplicationResult(
            job=_job(2), status=ApplicationStatus.SUBMITTED, timestamp=datetime.now(UTC)
        ),
        ApplicationResult(
            job=_job(3), status=ApplicationStatus.FAILED, timestamp=datetime.now(UTC)
        ),
    ]
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._applied_count == 1  # the FAILED one is not counted


def test_tui_detail_markup_renders_and_escapes() -> None:
    """The detail pane renders a job's fields and escapes markup metacharacters."""
    from datetime import UTC, datetime

    from rich.markup import escape

    from job_applicator.models import FunnelStatus, StoredJob

    job = JobListing(
        title="Sr Eng",
        company="Ac[me]",  # a "[" would be Rich markup if unescaped
        url="https://x/9",
        board=JobBoard.LINKEDIN,
        location="Remote",
        salary="$160k",
        requirements=["python"],
        description="Async role.",
    )
    stored = StoredJob(
        id=9,
        job=job,
        funnel_status=FunnelStatus.COVER_LETTER,
        match_score=0.81,
        semantic_score=0.8,
        skill_score=0.83,
        matched_skills=["python"],
        missing_skills=["k8s"],
        tailored_resume_path="/out/t.txt",
        cover_letter_path="/out/cl.txt",
        first_seen_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    md = app._detail_markup(stored)
    for token in (
        "Sr Eng",
        "Remote",
        "$160k",
        "81%",
        "python",
        "k8s",
        "/out/t.txt",
        "/out/cl.txt",
        "Async role.",
    ):
        assert token in md, token
    assert escape("Ac[me]") in md  # company is escaped, not interpreted as markup


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


def test_tui_tty_guard_requires_both_streams(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The TUI launches only when BOTH stdout and stdin are a TTY — a TTY stdout with a
    piped stdin (`producer | job-applicator`) must NOT launch (Textual would hang)."""
    import sys

    monkeypatch.setattr(sys, "stdout", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdin", MagicMock(isatty=lambda: False))
    assert cli._tui_tty_ok() is False
    monkeypatch.setattr(sys, "stdin", MagicMock(isatty=lambda: True))
    assert cli._tui_tty_ok() is True


def test_bare_invocation_tty_launches_tui(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """In a real terminal, bare `job-applicator` launches the TUI."""
    import job_applicator.tui as tui_pkg

    launched = MagicMock()
    monkeypatch.setattr(cli, "_tui_tty_ok", lambda: True)
    monkeypatch.setattr(cli, "_get_settings", lambda *a, **k: MagicMock())
    monkeypatch.setattr(tui_pkg, "run_tui", launched)
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    launched.assert_called_once()


def test_tui_command_tty_launches(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`job-applicator tui` in a real terminal launches the TUI."""
    import job_applicator.tui as tui_pkg

    launched = MagicMock()
    monkeypatch.setattr(cli, "_tui_tty_ok", lambda: True)
    monkeypatch.setattr(cli, "_get_settings", lambda *a, **k: MagicMock())
    monkeypatch.setattr(tui_pkg, "run_tui", launched)
    result = CliRunner().invoke(cli.app, ["tui"])
    assert result.exit_code == 0, result.output
    launched.assert_called_once()


def test_tui_launch_store_error_is_clean(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A store-construction failure at launch is a clean typed error, not a traceback."""
    import job_applicator.tui as tui_pkg

    monkeypatch.setattr(cli, "_tui_tty_ok", lambda: True)
    monkeypatch.setattr(cli, "_get_settings", lambda *a, **k: MagicMock())

    def _boom(_settings: object) -> None:
        raise JobStoreError("cannot open db")

    monkeypatch.setattr(tui_pkg, "run_tui", _boom)
    result = CliRunner().invoke(cli.app, ["tui"])
    assert result.exit_code == 1
    assert "cannot open db" in result.stderr
    assert "Traceback (most recent call last)" not in (result.stdout + result.stderr)


async def test_tui_tailor_action_marks_tailored(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pressing `t` runs the (mocked) tailor in a worker and advances the job to tailored."""
    from job_applicator.models import FunnelStatus, TailoredResume
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    fake = TailoredResume(
        original_path="/r.pdf",
        tailored_text="T",
        job_title="Engineer 1",
        job_company="Co1",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="c",
        output_path="/out/tailored.txt",
    )

    async def _fake_tailor(_settings: object, _job: object) -> TailoredResume:
        return fake

    monkeypatch.setattr(actions, "tailor_job", _fake_tailor)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("t")
        await app.workers.wait_for_complete()
        await pilot.pause()
    got = store.get("https://linkedin.com/jobs/1")
    assert got is not None and got.funnel_status is FunnelStatus.TAILORED
    assert got.tailored_resume_path == "/out/tailored.txt"


async def test_tui_tailor_needs_resume(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pressing `t` with no résumé configured warns and never runs the tailor."""
    from job_applicator.models import FunnelStatus
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    never = MagicMock()
    monkeypatch.setattr(actions, "tailor_job", never)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path=""),  # unset
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
    never.assert_not_called()
    assert store.get("https://linkedin.com/jobs/1").funnel_status is FunnelStatus.FOUND


def test_tui_statusline_unset_resume_keeps_dim_markup() -> None:
    """When resume_path is unset (first-run default), the sentinel keeps its dim styling
    instead of being escaped into literal '[dim]…' text."""
    app = JobApplicatorApp(
        settings=AppSettings(resume_path=""), store=MagicMock(), app_state=MagicMock()
    )
    line = app._statusline()
    assert "[dim]not set" in line  # markup preserved
    assert "\\[dim]" not in line  # not escaped to literal brackets
