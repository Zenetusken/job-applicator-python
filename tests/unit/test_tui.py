"""TUI shell tests — Pilot-driven, headless, account-safe.

The Textual app is driven via ``App.run_test()`` / ``Pilot`` (no real terminal). Launch
reads only the local SQLite store — never the account, a browser, or the LLM. Async
tests run under the project's ``asyncio_mode = auto``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def _browser_cm() -> MagicMock:
    """A stand-in for `_make_browser(...)` — an async context manager yielding a browser."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _mr(job: JobListing) -> object:
    from job_applicator.embeddings.matching import MatchResult

    return MatchResult(
        job=job,
        score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        matched_skills=["python"],
        missing_skills=[],
        summary="ok",
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
        # only the SUBMITTED job's URL is tracked as applied (the FAILED one is not)
        assert app._applied_urls == {"https://linkedin.com/jobs/2"}


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
        "t.txt",  # artifact paths shown as basename (no ugly mid-path wrap)
        "cl.txt",
        "Async role.",
    ):
        assert token in md, token
    assert "/out/" not in md  # the directory is dropped — basename only
    assert escape("Ac[me]") in md  # company is escaped, not interpreted as markup


def test_tui_detail_hides_placeholder_url() -> None:
    """A manual-tailor job (placeholder URL) shows no meaningless URL line; a real URL does."""
    from datetime import UTC, datetime

    from job_applicator.models import StoredJob

    def _stored(url: str) -> StoredJob:
        return StoredJob(
            id=1,
            job=JobListing(title="Dev", company="Acme", url=url, board=JobBoard.INDEED),
            first_seen_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    assert "placeholder" not in app._detail_markup(_stored("https://example.com/placeholder"))
    assert "linkedin.com/jobs/1" in app._detail_markup(_stored("https://linkedin.com/jobs/1"))
    # the URL is a clickable link (mouse → open), since the TUI captures terminal selection
    assert "@click=app.open_url" in app._detail_markup(_stored("https://linkedin.com/jobs/1"))


async def test_tui_open_url_opens_browser(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`o` opens the selected job's posting in the default browser."""
    import webbrowser

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    opened: dict[str, str] = {}
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.setdefault("url", u) or True)
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
    assert opened["url"] == "https://linkedin.com/jobs/1"


async def test_tui_copy_url_to_clipboard(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`y` copies the selected job's URL to the clipboard."""
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    copied: dict[str, str] = {}
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "copy_to_clipboard", lambda text: copied.setdefault("url", text))
        await pilot.press("y")
        await pilot.pause()
    assert copied["url"] == "https://linkedin.com/jobs/1"


async def test_tui_open_url_noop_on_placeholder(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A placeholder-URL job (manual tailor) doesn't open a browser on `o`."""
    import webbrowser

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(
        JobListing(
            title="Dev",
            company="Acme",
            url="https://example.com/placeholder",
            board=JobBoard.INDEED,
        )
    )
    never = MagicMock()
    monkeypatch.setattr(webbrowser, "open", never)
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
    never.assert_not_called()


async def test_tui_set_resume_sets_session_and_persists(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`e` → résumé modal → submit sets the session résumé AND writes it to config.toml."""
    from textual.widgets import Input

    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app = JobApplicatorApp(
        settings=AppSettings(resume_path=""),  # unset → the "press e" state
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#path", Input).value = "/cv/me.pdf"
        await pilot.press("enter")
        await pilot.pause()
    assert app._settings.resume_path == "/cv/me.pdf"  # session
    assert cfg.exists() and 'resume_path = "/cv/me.pdf"' in cfg.read_text()  # persisted


def test_persist_resume_path_updates_existing_config(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """_persist_resume_path replaces the existing resume_path line, preserving the rest."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('profile_name = "me"\nresume_path = "/old.pdf"\nlog_level = "INFO"\n')
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    saved = app._persist_resume_path("/new.pdf")
    assert saved == cfg
    text = cfg.read_text()
    assert 'resume_path = "/new.pdf"' in text and 'resume_path = "/old.pdf"' not in text
    assert 'profile_name = "me"' in text and 'log_level = "INFO"' in text  # rest preserved


def test_persist_resume_path_inserts_when_absent(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A config without a resume_path gets a top-level key prepended; rest preserved."""
    import tomllib

    cfg = tmp_path / "config.toml"
    cfg.write_text('profile_name = "me"\nlog_level = "INFO"\n')
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    assert app._persist_resume_path("/new.pdf") == cfg
    data = tomllib.loads(cfg.read_text())
    assert data["resume_path"] == "/new.pdf"
    assert data["profile_name"] == "me" and data["log_level"] == "INFO"


def test_persist_resume_path_no_duplicate_with_comment(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A commented example + an active key → the ACTIVE one is replaced, no duplicate key
    (the result still parses)."""
    import tomllib

    cfg = tmp_path / "config.toml"
    cfg.write_text('# resume_path = "/example.pdf"\nresume_path = "/old.pdf"\n')
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    assert app._persist_resume_path("/new.pdf") == cfg
    assert tomllib.loads(cfg.read_text())["resume_path"] == "/new.pdf"  # parses → no dup key
    assert "/old.pdf" not in cfg.read_text()


def test_persist_resume_path_rejects_mis_target_without_corrupting(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """A resume_path that would only land inside a [table] is rejected (None), file untouched."""
    cfg = tmp_path / "config.toml"
    original = '[llm]\nresume_path = "/wrong.pdf"\n'
    cfg.write_text(original)
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    # _persist reads the env directly (not settings), so a mock settings avoids loading
    # this intentionally-invalid config through pydantic.
    app = JobApplicatorApp(settings=MagicMock(), store=MagicMock(), app_state=MagicMock())
    assert app._persist_resume_path("/new.pdf") is None  # validation rejects
    assert cfg.read_text() == original  # not corrupted


async def test_tui_applied_job_counted_once_not_double(tmp_path: Path) -> None:
    """A job that is in the funnel AND applied shows as 'applied' once — no double-count."""
    from datetime import UTC, datetime

    from job_applicator.models import ApplicationResult, ApplicationStatus

    store = JobStore(db_path=tmp_path / "applications.db")
    job = _job(1)
    store.mark_tailored(job, tailored_resume_path="/t.txt")
    store.set_cover_letter("https://linkedin.com/jobs/1", "/cl.txt")  # head stage = cover_letter
    app_state = MagicMock()
    app_state.list_recent.return_value = [
        ApplicationResult(job=job, status=ApplicationStatus.SUBMITTED, timestamp=datetime.now(UTC))
    ]
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        line = app._statusline()
        assert "1 applied" in line
        assert "0 cover letter" in line  # NOT also counted at its head stage
        assert app._effective_stage(app._all[0]) == "applied"  # sidebar/detail agree


async def test_search_jobs_fallback_to_found_when_scoring_fails(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """If scoring raises, the scraped jobs are still persisted (as found) — never lost."""
    import job_applicator.factories as factories
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    jobs = [_job(1), _job(2)]
    monkeypatch.setattr(factories, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(
        factories, "_make_scraper", lambda *a, **k: MagicMock(scrape=AsyncMock(return_value=jobs))
    )

    def _boom(settings: object, j: object) -> object:
        raise RuntimeError("embeddings down")

    monkeypatch.setattr(actions, "_score_jobs", _boom)
    store = MagicMock()
    n = await actions.search_jobs(
        AppSettings(resume_path="/cv.pdf"), store, SearchParams(query="x", board=JobBoard.LINKEDIN)
    )
    assert n == 2
    assert store.upsert_job.call_count == 2  # fell back to found
    store.upsert_match.assert_not_called()


async def test_account_busy_blocks_a_second_action(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """While an account worker runs, _account_busy() is True and a 2nd action is refused."""
    import asyncio

    from job_applicator.models import JobBoard
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions
    from job_applicator.tui.screens import SearchScreen

    gate = asyncio.Event()

    async def _slow_search(settings: object, store: object, params: object) -> int:
        await gate.wait()  # hold the worker in RUNNING
        return 0

    monkeypatch.setattr(actions, "search_jobs", _slow_search)
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._search_worker(SearchParams(query="x", board=JobBoard.LINKEDIN))
        await pilot.pause()
        assert app._account_busy() is True  # the real predicate, not a mock
        app.action_search()  # a second account action…
        await pilot.pause()
        assert not isinstance(app.screen, SearchScreen)  # …is refused while busy
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._account_busy() is False


def test_conftest_isolates_real_store(tmp_path: Path) -> None:
    """The autouse isolation fixture points a no-arg JobStore away from real user state."""
    import pathlib

    real = pathlib.Path.home() / ".job-applicator" / "applications.db"
    assert JobStore()._path != real  # isolated by tests/conftest.py::_isolate_local_state


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


async def test_tui_cover_letter_action_advances_funnel(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pressing `c` runs the (mocked) cover-letter action and advances to cover_letter."""
    from job_applicator.models import CoverLetterResult, FunnelStatus
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    fake = CoverLetterResult(
        job_title="Engineer 1",
        job_company="Co1",
        job_url="https://linkedin.com/jobs/1",
        cover_letter_text="Dear hiring manager,",
        attempt=1,
        prompt_version="1.0",
        output_path="/out/cover.txt",
    )

    async def _fake_cl(_settings: object, _job: object) -> CoverLetterResult:
        return fake

    monkeypatch.setattr(actions, "cover_letter_job", _fake_cl)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("c")
        await app.workers.wait_for_complete()
        await pilot.pause()
    got = store.get("https://linkedin.com/jobs/1")
    assert got is not None and got.funnel_status is FunnelStatus.COVER_LETTER
    assert got.cover_letter_path == "/out/cover.txt"


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


async def test_tui_search_opens_modal_without_touching_account(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pressing `s` opens the search modal but constructs NO browser; cancel keeps it so."""
    import job_applicator.factories as factories
    from job_applicator.tui.screens import SearchScreen

    make_browser = MagicMock()
    monkeypatch.setattr(factories, "_make_browser", make_browser)
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        assert isinstance(app.screen, SearchScreen)  # modal is up…
        make_browser.assert_not_called()  # …but nothing has touched the account
        await pilot.press("escape")  # cancel
        await pilot.pause()
        make_browser.assert_not_called()


async def test_tui_search_submit_runs_and_persists(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Submitting the search modal runs the (mocked) scrape and the results land in the store."""
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")  # empty
    captured: dict[str, str] = {}

    async def _fake_search(_settings: object, st: JobStore, params: object) -> int:
        captured["query"] = params.query  # type: ignore[attr-defined]
        st.upsert_job(_job(7), source_query=params.query)  # type: ignore[attr-defined]
        return 1

    monkeypatch.setattr(actions, "search_jobs", _fake_search)
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("p", "y", "t", "h", "o", "n")  # query into the focused field
        await pilot.press("enter")  # submit (the deliberate, account-authorizing act)
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert captured["query"] == "python"
    assert store.get("https://linkedin.com/jobs/7") is not None


async def test_search_jobs_scores_when_resume_set(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """search_jobs scores scraped jobs against the résumé (→ upsert_match) when one is set."""
    import job_applicator.factories as factories
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    jobs = [_job(1), _job(2)]
    monkeypatch.setattr(factories, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(
        factories, "_make_scraper", lambda *a, **k: MagicMock(scrape=AsyncMock(return_value=jobs))
    )
    monkeypatch.setattr(actions, "_score_jobs", lambda settings, j: [_mr(jobs[0]), _mr(jobs[1])])
    store = MagicMock()
    n = await actions.search_jobs(
        AppSettings(resume_path="/cv.pdf"), store, SearchParams(query="x", board=JobBoard.LINKEDIN)
    )
    assert n == 2
    assert store.upsert_match.call_count == 2
    store.upsert_job.assert_not_called()


async def test_search_jobs_found_when_no_resume(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Without a résumé, search_jobs persists scraped jobs as found (no scoring)."""
    import job_applicator.factories as factories
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    jobs = [_job(1), _job(2)]
    monkeypatch.setattr(factories, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(
        factories, "_make_scraper", lambda *a, **k: MagicMock(scrape=AsyncMock(return_value=jobs))
    )
    score = MagicMock()
    monkeypatch.setattr(actions, "_score_jobs", score)
    store = MagicMock()
    n = await actions.search_jobs(
        AppSettings(resume_path=""), store, SearchParams(query="x", board=JobBoard.LINKEDIN)
    )
    assert n == 2
    assert store.upsert_job.call_count == 2
    store.upsert_match.assert_not_called()
    score.assert_not_called()  # no scoring without a résumé


async def test_tui_apply_modal_no_account_until_confirm(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pressing `a` opens the apply modal but constructs NO browser; cancel keeps it so."""
    import job_applicator.factories as factories
    from job_applicator.tui.screens import ApplyScreen

    make_browser = MagicMock()
    monkeypatch.setattr(factories, "_make_browser", make_browser)
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ApplyScreen)
        make_browser.assert_not_called()
        await pilot.press("escape")
        await pilot.pause()
        make_browser.assert_not_called()


async def test_tui_apply_dry_run_is_default(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Apply with the danger checkbox UNchecked runs a dry run (submit=False)."""
    from datetime import UTC, datetime

    from job_applicator.models import ApplicationResult, ApplicationStatus
    from job_applicator.tui import actions

    captured: dict[str, bool] = {}

    async def _fake_apply(_settings: object, job: object, *, submit: bool) -> ApplicationResult:
        captured["submit"] = submit
        return ApplicationResult(
            job=job, status=ApplicationStatus.PENDING, timestamp=datetime.now(UTC)
        )  # type: ignore[arg-type]

    monkeypatch.setattr(actions, "apply_job", _fake_apply)
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        await pilot.click("#go")  # checkbox left unchecked
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert captured["submit"] is False


async def test_tui_apply_real_submit_requires_checkbox(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Ticking the danger checkbox before Apply sends a real submit (submit=True)."""
    from datetime import UTC, datetime

    from job_applicator.models import ApplicationResult, ApplicationStatus
    from job_applicator.tui import actions

    captured: dict[str, bool] = {}

    async def _fake_apply(_settings: object, job: object, *, submit: bool) -> ApplicationResult:
        captured["submit"] = submit
        return ApplicationResult(
            job=job, status=ApplicationStatus.SUBMITTED, timestamp=datetime.now(UTC)
        )  # type: ignore[arg-type]

    monkeypatch.setattr(actions, "apply_job", _fake_apply)
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        await pilot.click("#real")  # tick the danger checkbox …
        await pilot.click("#go")  # … then Apply
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert captured["submit"] is True


async def test_apply_job_skips_over_cap_without_browser(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """apply_job(submit=True) over the daily cap returns SKIPPED and never opens a browser."""
    import job_applicator.factories as factories
    from job_applicator.models import ApplicationStatus
    from job_applicator.tui import actions

    make_browser = MagicMock()
    monkeypatch.setattr(factories, "_make_browser", make_browser)
    monkeypatch.setattr(
        "job_applicator.state.ApplicationState",
        lambda *a, **k: MagicMock(
            has_applied=lambda url: False, count_today=lambda board=None: 999
        ),
    )
    result = await actions.apply_job(AppSettings(), _job(1), submit=True)
    assert result.status is ApplicationStatus.SKIPPED
    make_browser.assert_not_called()


async def test_tui_search_empty_query_keeps_modal_open(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Submitting the search modal with a blank query keeps it open and touches nothing."""
    import job_applicator.factories as factories
    from job_applicator.tui.screens import SearchScreen

    make_browser = MagicMock()
    monkeypatch.setattr(factories, "_make_browser", make_browser)
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("enter")  # submit with an empty query
        await pilot.pause()
        assert isinstance(app.screen, SearchScreen)  # still open
        make_browser.assert_not_called()


async def test_tui_account_action_blocked_when_busy(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """While an account action is running, a second one is refused (not started/cancelled)."""
    from job_applicator.tui.screens import SearchScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_account_busy", lambda: True)
        await pilot.press("s")
        await pilot.pause()
        assert not isinstance(app.screen, SearchScreen)  # modal never opened while busy


async def test_tui_worker_error_does_not_crash_app(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A non-typed bug inside an action worker surfaces as a toast — the app stays alive."""
    from job_applicator.models import FunnelStatus
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))

    async def _boom(_settings: object, _job: object) -> object:
        raise RuntimeError("kaboom")  # NOT a JobApplicatorError

    monkeypatch.setattr(actions, "tailor_job", _boom)
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
        assert app.is_running  # survived the worker bug
    # the post-action store write never ran (the error was caught, not silently succeeded)
    assert store.get("https://linkedin.com/jobs/1").funnel_status is FunnelStatus.FOUND


async def test_apply_job_skips_already_applied_without_browser(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """apply_job(submit=True) on an already-applied job returns ALREADY_APPLIED and never
    opens a browser — the dedup gate fires before any account touch."""
    import job_applicator.factories as factories
    from job_applicator.models import ApplicationStatus
    from job_applicator.tui import actions

    make_browser = MagicMock()
    monkeypatch.setattr(factories, "_make_browser", make_browser)
    monkeypatch.setattr(
        "job_applicator.state.ApplicationState",
        lambda *a, **k: MagicMock(has_applied=lambda url: True, count_today=lambda board=None: 0),
    )
    result = await actions.apply_job(AppSettings(), _job(1), submit=True)
    assert result.status is ApplicationStatus.ALREADY_APPLIED
    make_browser.assert_not_called()


def test_tui_statusline_unset_resume_keeps_dim_markup() -> None:
    """When resume_path is unset (first-run default), the sentinel keeps its dim styling
    instead of being escaped into literal '[dim]…' text."""
    app = JobApplicatorApp(
        settings=AppSettings(resume_path=""), store=MagicMock(), app_state=MagicMock()
    )
    line = app._statusline()
    assert "[dim]not set" in line  # markup preserved
    assert "\\[dim]" not in line  # not escaped to literal brackets
