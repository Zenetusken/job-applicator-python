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


def _mr(job: JobListing, score: float = 0.8) -> object:
    from job_applicator.embeddings.matching import MatchResult

    return MatchResult(
        job=job,
        score=score,
        semantic_score=score,
        skill_score=score,
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


async def test_tui_help_key_does_not_leak_into_filter(tmp_path: Path) -> None:
    """`?` must NOT open help while the filter Input is focused — like the documented
    `/`-leak, a printable key typed mid-filter belongs in the Input, not the app binding."""
    from textual.widgets import Input

    from job_applicator.tui.screens import HelpScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("slash")  # open + focus the filter Input
        await pilot.pause()
        await pilot.press("question_mark")  # ? while filtering → a literal char, not help
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)  # help did NOT open
        assert "?" in app.query_one("#filter", Input).value  # the char went to the Input


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
    """`o` opens the selected job's posting in the default browser (off-thread worker)."""
    import webbrowser

    monkeypatch.setenv("DISPLAY", ":0")  # pass the headless guard
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
        await app.workers.wait_for_complete()  # the browser open runs in a thread worker
        await pilot.pause()
    assert opened["url"] == "https://linkedin.com/jobs/1"


async def test_tui_open_url_headless_does_not_launch(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """On headless Linux (no DISPLAY) `o` does NOT launch a browser (would hijack the TUI)."""
    import sys
    import webbrowser

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    never = MagicMock()
    monkeypatch.setattr(webbrowser, "open", never)
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("o")
        await app.workers.wait_for_complete()
        await pilot.pause()
    never.assert_not_called()


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


async def test_tui_open_url_toast_escapes_markup_in_url(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A scraped URL with a tag-like bracket (?ref=[red]…) is escaped in the toast — notify
    renders Rich markup, so an unescaped span would be silently dropped from the message."""
    import sys

    from rich.markup import escape

    monkeypatch.setattr(sys, "platform", "linux")  # headless → synchronous warning toast
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    raw = "https://linkedin.com/jobs/1?ref=[red]abc"
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1, url=raw))
    toasts: list[str] = []
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "notify", lambda msg, **k: toasts.append(msg))
        app.action_open_url()
        await pilot.pause()
    assert toasts and escape(raw) in toasts[0]  # bracket span escaped, not stripped


def _tailored_app(tmp_path: Path, **paths: str) -> JobApplicatorApp:
    """An app whose single selected job carries the given artifact path(s)."""
    store = JobStore(db_path=tmp_path / "applications.db")
    job = _job(1)
    store.upsert_job(job)
    store.mark_tailored(job, **paths)  # tailored_resume_path=… [+ cover_letter_path=…]
    return JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )


def test_tui_artifact_lines_are_clickable(tmp_path: Path) -> None:
    """The tailored-résumé and cover-letter lines are click-to-open links; the path is NOT
    baked into the markup (the action resolves it from the selected job at click time)."""
    from datetime import UTC, datetime

    from job_applicator.models import StoredJob

    stored = StoredJob(
        id=1,
        job=_job(1),
        tailored_resume_path="/out/t.txt",
        cover_letter_path="/out/cl.txt",
        first_seen_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    md = app._detail_markup(stored)
    assert "@click=app.open_tailored" in md and "@click=app.open_cover" in md
    assert "/out/" not in md  # no path embedded — resolved from _current on click


async def test_tui_open_tailored_artifact(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Opening the tailored artifact launches the default viewer with its file:// URI."""
    import webbrowser

    monkeypatch.setenv("DISPLAY", ":0")  # pass the headless guard
    art = tmp_path / "tailored_Acme_Dev.txt"
    art.write_text("resume text", encoding="utf-8")
    opened: dict[str, str] = {}
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.setdefault("uri", u) or True)
    app = _tailored_app(tmp_path, tailored_resume_path=str(art))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_open_tailored()
        await app.workers.wait_for_complete()  # off-thread open worker
        await pilot.pause()
    assert opened["uri"].startswith("file://") and opened["uri"].endswith("tailored_Acme_Dev.txt")


async def test_tui_open_cover_artifact(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The cover-letter line opens its own artifact (open_cover wired to cover_letter_path)."""
    import webbrowser

    monkeypatch.setenv("DISPLAY", ":0")
    resume = tmp_path / "tailored.txt"
    resume.write_text("r", encoding="utf-8")
    cover = tmp_path / "cover_letter_Acme.txt"
    cover.write_text("dear hiring manager", encoding="utf-8")
    opened: dict[str, str] = {}
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.setdefault("uri", u) or True)
    app = _tailored_app(tmp_path, tailored_resume_path=str(resume), cover_letter_path=str(cover))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_open_cover()
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert opened["uri"].endswith("cover_letter_Acme.txt")  # the cover, not the résumé


async def test_tui_open_tailored_headless_does_not_launch(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """On headless Linux (no DISPLAY) opening an artifact does NOT launch a viewer."""
    import sys
    import webbrowser

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    art = tmp_path / "t.txt"
    art.write_text("x", encoding="utf-8")
    never = MagicMock()
    monkeypatch.setattr(webbrowser, "open", never)
    app = _tailored_app(tmp_path, tailored_resume_path=str(art))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_open_tailored()
        await app.workers.wait_for_complete()
        await pilot.pause()
    never.assert_not_called()


async def test_tui_open_tailored_noop_when_absent(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A job with no tailored artifact: opening it warns and launches nothing."""
    import webbrowser

    monkeypatch.setenv("DISPLAY", ":0")
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))  # no artifact generated
    never = MagicMock()
    monkeypatch.setattr(webbrowser, "open", never)
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_open_tailored()
        await app.workers.wait_for_complete()
        await pilot.pause()
    never.assert_not_called()


async def test_tui_open_tailored_missing_file_does_not_launch(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A recorded artifact path whose file is gone warns and launches nothing (not a crash)."""
    import webbrowser

    monkeypatch.setenv("DISPLAY", ":0")
    never = MagicMock()
    monkeypatch.setattr(webbrowser, "open", never)
    app = _tailored_app(tmp_path, tailored_resume_path=str(tmp_path / "gone.txt"))  # never created
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_open_tailored()
        await app.workers.wait_for_complete()
        await pilot.pause()
    never.assert_not_called()


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

    async def _slow_search(
        settings: object, store: object, params: object, on_progress=None, on_job=None
    ) -> int:  # type: ignore[no-untyped-def]
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

    async def _fake_cl(
        _settings: object, _job: object, tailored_resume_path: str = ""
    ) -> CoverLetterResult:
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


async def test_cover_letter_job_uses_tailored_resume_text(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """cover_letter_job reads a given tailored-résumé artifact and passes its text to the
    generator; with no path it falls back to empty (the original résumé)."""
    from job_applicator.tui import actions

    tailored = tmp_path / "tailored.txt"
    tailored.write_text("TAILORED RESUME BODY", encoding="utf-8")
    captured: dict[str, str] = {}

    async def _generate(
        job, user, resume, style_guide=None, tone_section="", tailored_resume_text=""
    ):  # type: ignore[no-untyped-def]
        captured["tailored"] = tailored_resume_text
        return "Dear hiring manager,"

    monkeypatch.setattr(
        "job_applicator.documents.cover_letter.CoverLetterGenerator",
        lambda *a, **k: MagicMock(generate=_generate),
    )
    monkeypatch.setattr(
        "job_applicator.documents.resume.ResumeLoader",
        lambda: MagicMock(load=lambda p: MagicMock()),
    )
    monkeypatch.setattr("job_applicator.factories._make_runtime", lambda s: MagicMock())
    monkeypatch.setattr(
        "job_applicator.documents.tone_detector.ToneDetector",
        lambda: MagicMock(format_for_prompt=lambda t: ""),
    )
    monkeypatch.setattr("job_applicator.utils.profile._detect_tone", lambda job: MagicMock())
    monkeypatch.setattr("job_applicator.utils.profile._load_user_profile", lambda s: MagicMock())
    settings = AppSettings(resume_path="/r.pdf", output_dir=str(tmp_path / "out"))

    await actions.cover_letter_job(settings, _job(1), tailored_resume_path=str(tailored))
    assert captured["tailored"] == "TAILORED RESUME BODY"  # tailored text read + passed
    await actions.cover_letter_job(settings, _job(1))  # no tailored path
    assert captured["tailored"] == ""  # falls back to the original résumé


async def test_ats_check_runs_checker_on_resume(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """ats_check loads the résumé and returns the ATSChecker result (offline)."""
    from job_applicator.models import ATSCompatibilityResult
    from job_applicator.tui import actions

    result = ATSCompatibilityResult(score=0.9, warnings=[], suggestions=[])
    checker = MagicMock()
    checker.return_value.check.return_value = result
    monkeypatch.setattr("job_applicator.documents.ats_checker.ATSChecker", checker)
    monkeypatch.setattr(
        "job_applicator.documents.resume.ResumeLoader",
        lambda: MagicMock(load=lambda p: MagicMock(), parse_text=lambda t: MagicMock()),
    )
    got = await actions.ats_check(AppSettings(resume_path="/r.pdf"))
    assert got is result
    checker.return_value.check.assert_called_once()


async def test_ats_check_falls_back_on_empty_tailored(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A whitespace-only / unreadable tailored artifact falls back to the configured résumé."""
    from job_applicator.models import ATSCompatibilityResult
    from job_applicator.tui import actions

    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")  # whitespace-only
    result = ATSCompatibilityResult(score=0.9)
    loader = MagicMock(load=MagicMock(return_value=MagicMock()), parse_text=MagicMock())
    checker = MagicMock()
    checker.return_value.check.return_value = result
    monkeypatch.setattr("job_applicator.documents.ats_checker.ATSChecker", checker)
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    got = await actions.ats_check(
        AppSettings(resume_path="/r.pdf"), tailored_resume_path=str(empty)
    )
    assert got is result
    loader.load.assert_called_once()  # fell back to the configured résumé
    loader.parse_text.assert_not_called()  # empty tailored text → not parsed


async def test_tui_ats_check_shows_modal(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`A` runs the (mocked) ATS check and shows the result modal."""
    from job_applicator.models import ATSCompatibilityResult
    from job_applicator.tui import actions
    from job_applicator.tui.screens import AtsScreen

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    result = ATSCompatibilityResult(score=0.7, warnings=["No phone number."], suggestions=[])

    async def _fake_ats(_settings: object, tailored: str = "") -> ATSCompatibilityResult:
        return result

    monkeypatch.setattr(actions, "ats_check", _fake_ats)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("A")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, AtsScreen)


async def test_tui_ats_check_needs_resume(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`A` with no résumé configured warns and never runs the ATS check."""
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    never = MagicMock()
    monkeypatch.setattr(actions, "ats_check", never)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path=""),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("A")
        await pilot.pause()
    never.assert_not_called()


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

    async def _fake_search(
        _settings: object, st: JobStore, params: object, on_progress=None, on_job=None
    ) -> int:  # type: ignore[no-untyped-def]
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


async def test_search_jobs_reports_phase_progress(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """search_jobs calls on_progress at each phase (open browser → search → score)."""
    import job_applicator.factories as factories
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    jobs = [_job(1), _job(2)]
    monkeypatch.setattr(factories, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(
        factories, "_make_scraper", lambda *a, **k: MagicMock(scrape=AsyncMock(return_value=jobs))
    )
    monkeypatch.setattr(actions, "_score_jobs", lambda settings, j: [_mr(jobs[0]), _mr(jobs[1])])
    msgs: list[str] = []
    await actions.search_jobs(
        AppSettings(resume_path="/cv.pdf"),
        MagicMock(),
        SearchParams(query="python", board=JobBoard.LINKEDIN),
        on_progress=msgs.append,
    )
    joined = " | ".join(msgs)
    assert "Opening a browser" in joined and "Searching" in joined and "Scoring" in joined


async def test_search_jobs_forwards_per_item_progress(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """search_jobs forwards on_progress INTO scraper.scrape, so the scraper's per-card ticks
    reach the UI sink alongside the phase messages (the scraper formats; actions just wires)."""
    import job_applicator.factories as factories
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    jobs = [_job(1), _job(2)]

    async def _fake_scrape(params: object, on_progress=None, on_job=None):  # type: ignore[no-untyped-def]
        if on_progress is not None:  # the scraper emits per-card progress through its sink
            on_progress("Scraping job 1/2 on LinkedIn…")
            on_progress("Scraping job 2/2 on LinkedIn…")
        return jobs

    monkeypatch.setattr(factories, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(factories, "_make_scraper", lambda *a, **k: MagicMock(scrape=_fake_scrape))
    monkeypatch.setattr(actions, "_score_jobs", lambda settings, j: [_mr(jobs[0]), _mr(jobs[1])])
    msgs: list[str] = []
    await actions.search_jobs(
        AppSettings(resume_path="/cv.pdf"),
        MagicMock(),
        SearchParams(query="python", board=JobBoard.LINKEDIN),
        on_progress=msgs.append,
    )
    joined = " | ".join(msgs)
    assert "Scraping job 1/2" in joined and "Scraping job 2/2" in joined  # per-item forwarded
    assert "Scoring" in joined  # phase messages still flow too


async def test_search_jobs_streams_found_then_matched(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A streaming scraper: each job is upserted as found AS it arrives (and the on_job UI
    hook fires), then scoring re-upserts each as matched — incremental persistence + upgrade."""
    import job_applicator.factories as factories
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    jobs = [_job(1), _job(2)]

    async def _fake_scrape(params: object, on_progress=None, on_job=None):  # type: ignore[no-untyped-def]
        for j in jobs:  # the scraper streams each listing as it lands
            if on_job is not None:
                on_job(j)
        return jobs

    monkeypatch.setattr(factories, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(factories, "_make_scraper", lambda *a, **k: MagicMock(scrape=_fake_scrape))
    monkeypatch.setattr(actions, "_score_jobs", lambda s, j: [_mr(jobs[0]), _mr(jobs[1])])
    store = MagicMock()
    seen: list[object] = []
    await actions.search_jobs(
        AppSettings(resume_path="/cv.pdf"),
        store,
        SearchParams(query="x", board=JobBoard.LINKEDIN),
        on_job=seen.append,
    )
    assert [j.title for j in seen] == ["Engineer 1", "Engineer 2"]  # UI hook saw each stream
    assert store.upsert_job.call_count == 2  # persisted as found during the stream (emit)
    assert store.upsert_match.call_count == 2  # then upgraded to matched after scoring


async def test_search_jobs_stream_does_not_downgrade_tailored(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Re-scraping a job already tailored must NOT downgrade it: emit's found-upsert during
    the stream preserves the stage + artifact paths (the store no-downgrades on rediscovery)."""
    import job_applicator.factories as factories
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")
    job = _job(1)
    store.upsert_job(job)
    store.mark_tailored(job, tailored_resume_path="/out/t.txt")  # advance to tailored

    async def _fake_scrape(params: object, on_progress=None, on_job=None):  # type: ignore[no-untyped-def]
        if on_job is not None:
            on_job(job)  # re-scrape the SAME url mid-stream
        return [job]

    monkeypatch.setattr(factories, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(factories, "_make_scraper", lambda *a, **k: MagicMock(scrape=_fake_scrape))
    # No résumé → no scoring → only emit's found-upsert runs (the downgrade risk path).
    await actions.search_jobs(
        AppSettings(), store, SearchParams(query="x", board=JobBoard.LINKEDIN)
    )
    stored = store.list_jobs(limit=10)[0]
    assert stored.funnel_status.value == "tailored"  # NOT downgraded to found
    assert stored.tailored_resume_path == "/out/t.txt"  # artifact path preserved


async def test_tui_search_streams_rows_incrementally(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Rows appear in the table AS the search streams them (not all at once at the end), and
    the loading spinner clears on the first result."""
    import asyncio

    from textual.widgets import DataTable

    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    gate = asyncio.Event()

    async def _fake(_s: object, st: JobStore, _p: object, on_progress=None, on_job=None) -> int:  # type: ignore[no-untyped-def]
        st.upsert_job(_job(101), source_query="x")  # mimic emit: persist THEN notify
        if on_job is not None:
            on_job(_job(101))
        await gate.wait()  # hold after the first streamed row
        st.upsert_job(_job(102), source_query="x")
        if on_job is not None:
            on_job(_job(102))
        return 2

    monkeypatch.setattr(actions, "search_jobs", _fake)
    store = JobStore(db_path=tmp_path / "applications.db")  # empty
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", DataTable)
        assert table.row_count == 0
        app._search_worker(SearchParams(query="x", board=JobBoard.LINKEDIN))
        await pilot.pause()
        assert table.row_count == 1  # first row streamed in (not waiting for the whole scrape)
        assert table.loading is False  # spinner cleared on the first result
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert table.row_count == 2  # second row streamed in too


async def test_tui_reload_sorts_best_match_first(tmp_path: Path) -> None:
    """The list VIEW shows scored jobs best-first then unscored — independent of the store's
    updated_at order (we sort the view, not the shared list_jobs query the CLI also uses)."""
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_match(_mr(_job(1), 0.30))  # low score, upserted first (oldest updated_at)
    store.upsert_match(_mr(_job(2), 0.90))  # high score
    store.upsert_job(_job(3))  # unscored, newest updated_at
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        order = [s.job.title for s in app._all]
        # scored desc (E2@90% then E1@30%), then the unscored E3 last — NOT updated_at order
        assert order == ["Engineer 2", "Engineer 1", "Engineer 3"]


async def test_tui_search_shows_per_item_progress(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A per-item scrape tick reaches the live status line (⏳ Scraping N/M) DURING the
    search — verified mid-flight via a gate, then cleared when the worker finishes."""
    import asyncio

    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    gate = asyncio.Event()

    async def _fake(_s: object, _st: object, _p: object, on_progress=None, on_job=None) -> int:  # type: ignore[no-untyped-def]
        if on_progress is not None:
            on_progress("Scraping job 2/3 on LinkedIn…")
        await gate.wait()  # hold the worker so we can observe the live line
        return 0

    monkeypatch.setattr(actions, "search_jobs", _fake)
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._search_worker(SearchParams(query="x", board=JobBoard.LINKEDIN))
        await pilot.pause()
        line = app._statusline()
        assert "⏳" in line and "Scraping job 2/3" in line  # the per-item count is live
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._busy == ""  # cleared after the search


async def test_tui_busy_indicator_in_statusline(tmp_path: Path) -> None:
    """_set_busy shows a live '⏳ …' line in the status bar and clears back to counts."""
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_busy("Tailoring X…")
        assert "⏳" in app._statusline() and "Tailoring X" in app._statusline()
        app._set_busy("")
        assert "⏳" not in app._statusline()  # restored to the funnel counts


async def test_tui_search_clears_busy_and_loading(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """After a search, the list spinner (loading) and the busy line are both cleared."""
    from textual.widgets import DataTable

    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")

    async def _fake(_s: object, _st: object, _p: object, on_progress=None, on_job=None) -> int:  # type: ignore[no-untyped-def]
        if on_progress is not None:
            on_progress("Searching…")
        return 0

    monkeypatch.setattr(actions, "search_jobs", _fake)
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("p", "y", "t", "h", "o", "n")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._busy == ""
        assert app.query_one("#joblist", DataTable).loading is False


async def test_tui_help_modal_opens_and_lists_keys(tmp_path: Path) -> None:
    """`?` opens a read-only key reference that renders the actions + the account-safety
    note; Esc dismisses it. Touches no account (pure presentation)."""
    from textual.widgets import Static

    from job_applicator.tui.screens import HelpScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        body = str(app.screen.query_one("#helpbody").query_one(Static).render())
        assert "tailor" in body.lower() and "search" in body.lower()  # actions listed
        assert "dry-run" in body  # account-safety reinforced in-context
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)  # dismissed


async def test_tui_modal_fades_in_to_full_opacity(tmp_path: Path) -> None:
    """Modals fade in on mount (a 'layer appeared' transition) and SETTLE at full opacity.
    The visual feel is for live testing; this guards against a modal stuck transparent."""
    from job_applicator.tui.screens import HelpScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause(0.4)  # let the ~0.18s fade complete
        assert isinstance(app.screen, HelpScreen)
        assert app.screen.styles.opacity == 1.0  # fully visible after the fade, not stuck at 0


def test_tui_empty_state_points_at_in_app_keys(tmp_path: Path) -> None:
    """With an empty store the detail pane guides via IN-APP keys (s / e / ?), not a CLI
    command — the TUI shouldn't tell a user sitting inside it to go run the CLI."""
    store = JobStore(db_path=tmp_path / "applications.db")  # empty
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    markup = app._detail_markup(None)
    assert "job-applicator search" not in markup  # no longer points OUT to the CLI
    assert "[cyan]s[/cyan]" in markup and "[cyan]?[/cyan]" in markup  # in-app keys


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
