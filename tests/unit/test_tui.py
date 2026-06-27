"""TUI shell tests — Pilot-driven, headless, account-safe.

The Textual app is driven via ``App.run_test()`` / ``Pilot`` (no real terminal). Launch
reads only the local SQLite store — never the account, a browser, or the LLM. Async
tests run under the project's ``asyncio_mode = auto``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from textual.widgets import OptionList
from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.config import AppSettings
from job_applicator.jobs_store import JobStore, JobStoreError
from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile
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
        table = app.query_one("#joblist", OptionList)
        assert table.option_count == 3
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
        table = app.query_one("#joblist", OptionList)
        assert table.option_count == 2
        await pilot.press("slash")
        await pilot.pause()
        await pilot.press("g", "l", "o", "b", "e", "x")
        await pilot.pause()
        from textual.widgets import Input

        assert app.query_one("#filter", Input).value == "globex"  # the "/" trigger didn't leak
        await pilot.press("enter")
        await pilot.pause()
        assert table.option_count == 1  # only Globex
        await pilot.press("escape")
        await pilot.pause()
        assert table.option_count == 2  # filter cleared


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
        assert app.query_one("#joblist", OptionList).option_count == 0
        assert app._current is None


async def test_tui_refresh_picks_up_new_jobs(tmp_path: Path) -> None:
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app_state = MagicMock()
    app_state.list_recent.return_value = []
    app = JobApplicatorApp(settings=AppSettings(), store=store, app_state=app_state)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", OptionList)
        assert table.option_count == 1
        store.upsert_job(_job(2))  # a new job lands in the store
        await pilot.press("r")
        await pilot.pause()
        assert table.option_count == 2


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


def _stored(**job_over: object) -> object:
    from datetime import UTC, datetime

    from job_applicator.models import StoredJob

    return StoredJob(
        id=1,
        job=_job(1, **job_over),
        first_seen_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def test_jobboard_display_name() -> None:
    """The user-facing board name is properly cased; the enum value stays the wire form."""
    assert JobBoard.LINKEDIN.display_name == "LinkedIn"
    assert JobBoard.INDEED.display_name == "Indeed"
    assert JobBoard.LINKEDIN.value == "linkedin"  # wire form unchanged


def test_tui_detail_shows_full_description() -> None:
    """The full posting renders (the detail pane scrolls) — not capped at 600 chars, which
    hid the rest of the posting even though the pane can scroll."""
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    desc = "HEAD " + "x" * 700 + " UNIQUE_TAIL_MARKER"  # > 600 chars
    md = app._detail_markup(_stored(description=desc))
    assert "UNIQUE_TAIL_MARKER" in md  # the tail beyond char 600 is present


def test_tui_detail_elides_long_url() -> None:
    """A long tracking URL is elided in the DISPLAY but stays clickable — o/click/y act on the
    full stored URL, never this display text."""
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    url = "https://www.linkedin.com/jobs/view/4267843029/?refId=" + "Z" * 200
    md = app._detail_markup(_stored(url=url))
    assert "@click=app.open_url" in md  # still clickable (the action reads j.url, not display)
    assert "…" in md  # the display form is elided
    assert "Z" * 200 not in md  # the tracking tail is not dumped into the pane
    assert "linkedin.com/jobs/view/4267843029" in md  # the meaningful head is kept


def test_tui_detail_shows_board_proper_case() -> None:
    """The detail line shows the board properly cased ('LinkedIn'), not the enum value."""
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    md = app._detail_markup(_stored(company="Acme"))
    assert "LinkedIn" in md  # proper-cased board name


def test_tui_detail_elides_long_artifact_filename() -> None:
    """A long artifact filename is middle-elided (keeps prefix + timestamp), still clickable."""
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    from datetime import UTC, datetime

    from job_applicator.models import StoredJob

    longname = "/out/tailored_" + "C" * 60 + "_194403.txt"
    stored = StoredJob(
        id=1,
        job=_job(1),
        tailored_resume_path=longname,
        first_seen_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    md = app._detail_markup(stored)
    assert "…" in md  # middle-elided
    assert "C" * 60 not in md  # the long run is collapsed
    assert "194403.txt" in md  # the identifying tail is kept
    assert "@click=app.open_tailored" in md  # still clickable


async def test_tui_joblist_company_on_metadata_line(tmp_path: Path) -> None:
    """Two-line cards: the company sits on each job's 2nd (metadata) line, so even a very long
    title can never push it off-screen — it's always present in the card, and the card WRAPS to
    the pane (no horizontal scroll)."""
    from textual.widgets import OptionList

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1, title="T" * 120, company="Acme Corporation Worldwide Holdings"))
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        t = app.query_one("#joblist", OptionList)
        card = str(t.get_option_at_index(0).prompt)  # the 2-line card text
        assert "Acme Corporation Worldwide Holdings" in card  # company always present (line 2)
        assert t.virtual_size.width <= t.size.width  # wraps to the pane, no h-scroll


async def test_tui_detail_scroll_keys(tmp_path: Path) -> None:
    """`]` pages the detail pane down — so a keyboard user can read a long posting that the
    list-focused arrow keys can't reach."""
    from textual.containers import VerticalScroll

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1, description="paragraph\n" * 400))  # overflows the pane
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        pane = app.query_one("#detailscroll", VerticalScroll)
        assert pane.scroll_target_y == 0
        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert pane.scroll_target_y > 0  # the posting scrolled down


def test_tui_joblist_loading_widget_override() -> None:
    """The loading widget is a self-rendering LoadingIndicator subclass (a container with
    composed children collapses when used as a cover — the regression this guards)."""
    from textual.widgets import LoadingIndicator

    from job_applicator.tui.app import JobList, _JobListLoading

    widget = JobList().get_loading_widget()
    assert isinstance(widget, _JobListLoading)
    assert isinstance(widget, LoadingIndicator)  # leaf, renders itself — does not collapse


async def test_tui_joblist_loading_is_themed_on_screen(tmp_path: Path) -> None:
    """Real frame: loading mounts a self-rendering cover that FILLS the area (non-collapsed)
    with a SOLID background — guards both the grey bleed AND the 0x0-collapse regression."""
    from textual.widgets import OptionList

    from job_applicator.tui.app import _JobListLoading

    store = JobStore(db_path=tmp_path / "applications.db")
    for i in range(8):  # enough rows that the cover area is clearly non-trivial
        store.upsert_job(_job(i))
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", OptionList)
        table.loading = True  # the reactive the real search worker uses
        await pilot.pause()
        cover = table._cover_widget  # the loading cover lives here (not the query tree)
        assert isinstance(cover, _JobListLoading)  # our themed widget, not the framework default
        assert cover.size.width > 0 and cover.size.height > 0  # renders (didn't collapse to 0x0)
        assert cover.styles.background.a == 1.0  # solid → no grey bleed-through


async def test_tui_panes_fill_body_equally(tmp_path: Path) -> None:
    """The list and detail panes are the same height (both fill the body) — a DataTable
    defaults to height:auto, which left the left side short of the bottom with few rows."""
    from textual.containers import VerticalScroll
    from textual.widgets import OptionList

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))  # a SINGLE row — the case where auto-height was short
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", OptionList)
        detail = app.query_one("#detailscroll", VerticalScroll)
        body_height = app.query_one("#body").size.height
        # Both panes fill the body (symmetric). A collapsed auto-height table (~2 rows) fails
        # the body-equality, so this catches the regression even without the symmetry assert.
        assert table.size.height == detail.size.height == body_height


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


async def test_tui_open_pdf_artifact(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The `p` key opens the tailored-résumé PDF when one exists."""
    import webbrowser

    monkeypatch.setenv("DISPLAY", ":0")
    txt = tmp_path / "tailored_Acme_Dev.txt"
    txt.write_text("resume text", encoding="utf-8")
    pdf = tmp_path / "tailored_Acme_Dev_20260625_120000_000000_modern.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    opened: dict[str, str] = {}
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.setdefault("uri", u) or True)
    app = _tailored_app(tmp_path, tailored_resume_path=str(txt), pdf_path=str(pdf))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_open_pdf()
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert opened["uri"].endswith("tailored_Acme_Dev_20260625_120000_000000_modern.pdf")


async def test_tui_open_pdf_falls_back_to_cover_letter_pdf(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`action_open_pdf` falls back to the cover-letter PDF when no résumé PDF exists."""
    import webbrowser

    monkeypatch.setenv("DISPLAY", ":0")
    txt = tmp_path / "tailored_Acme_Dev.txt"
    txt.write_text("resume text", encoding="utf-8")
    cl_pdf = tmp_path / "cover_letter_Acme_Dev_20260625_120000_000000_modern.pdf"
    cl_pdf.write_bytes(b"%PDF-1.4 fake")
    opened: dict[str, str] = {}
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.setdefault("uri", u) or True)
    app = _tailored_app(tmp_path, tailored_resume_path=str(txt), cover_letter_pdf_path=str(cl_pdf))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_open_pdf()
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert opened["uri"].endswith("cover_letter_Acme_Dev_20260625_120000_000000_modern.pdf")


async def test_tui_open_pdf_noop_when_absent(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A job with no generated PDF: `p` warns and launches nothing."""
    import webbrowser

    monkeypatch.setenv("DISPLAY", ":0")
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    never = MagicMock()
    monkeypatch.setattr(webbrowser, "open", never)
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_open_pdf()
        await app.workers.wait_for_complete()
        await pilot.pause()
    never.assert_not_called()


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
        from textual.widgets import Tab

        # Counts live on the tabs now: the job shows as applied(1), NOT also at its head stage.
        assert "1" in str(app.query_one("#stage-applied", Tab).label)
        assert "0" in str(app.query_one("#stage-cover_letter", Tab).label)
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

    async def _boom(settings: object, j: object) -> object:
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
        app._search_worker([SearchParams(query="x", board=JobBoard.LINKEDIN)])
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

    async def _fake_tailor(
        _settings: object, _job: object, *, style_guide_path: str = "", **kw: object
    ) -> TailoredResume:
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


async def test_tui_tailor_survives_store_write_failure(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A failing funnel-store write AFTER a successful tailor must NOT crash the worker and lose
    the result — the artifact is already written, so it surfaces a warning toast instead."""
    from job_applicator.jobs_store import JobStoreError
    from job_applicator.models import TailoredResume
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

    async def _fake_tailor(_s: object, _j: object, *, style_guide_path: str = "", **kw: object):
        return fake

    def _boom(*_a: object, **_k: object) -> None:
        raise JobStoreError("database is locked")

    monkeypatch.setattr(actions, "tailor_job", _fake_tailor)
    monkeypatch.setattr(store, "mark_tailored", _boom)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    notes: list[dict[str, object]] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "notify", lambda *_a, **k: notes.append(k))
        await pilot.press("t")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert any(n.get("severity") == "warning" for n in notes)  # warned, didn't crash


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
        _settings: object,
        _job: object,
        tailored_resume_path: str = "",
        *,
        style_guide_path: str = "",
        **kw: object,
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
    monkeypatch.setattr(
        "job_applicator.utils.profile._load_user_profile",
        lambda s, *, resume_name="": MagicMock(),
    )
    settings = AppSettings(resume_path="/r.pdf", output_dir=str(tmp_path / "out"))

    await actions.cover_letter_job(settings, _job(1), tailored_resume_path=str(tailored))
    assert captured["tailored"] == "TAILORED RESUME BODY"  # tailored text read + passed
    await actions.cover_letter_job(settings, _job(1))  # no tailored path
    assert captured["tailored"] == ""  # falls back to the original résumé


async def test_cover_letter_job_uses_resume_name_for_sign_off(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """cover_letter_job resolves the applicant name from the parsed résumé so the
    generated cover letter is signed correctly."""
    from job_applicator.tui import actions

    captured: dict[str, str] = {}

    async def _generate(
        job, user, resume, style_guide=None, tone_section="", tailored_resume_text=""
    ):  # type: ignore[no-untyped-def]
        return f"Dear hiring manager,\n\nLetter.\n\nSincerely,\n{user.first_name} {user.last_name}"

    monkeypatch.setattr(
        "job_applicator.documents.cover_letter.CoverLetterGenerator",
        lambda *a, **k: MagicMock(generate=_generate),
    )
    monkeypatch.setattr(
        "job_applicator.documents.resume.ResumeLoader",
        lambda: MagicMock(load=lambda p: ResumeData(raw_text="resume text", name="Sam Sample")),
    )
    monkeypatch.setattr("job_applicator.factories._make_runtime", lambda s: MagicMock())
    monkeypatch.setattr(
        "job_applicator.documents.tone_detector.ToneDetector",
        lambda: MagicMock(format_for_prompt=lambda t: ""),
    )
    monkeypatch.setattr("job_applicator.utils.profile._detect_tone", lambda job: MagicMock())

    def _capture_load(s, *, resume_name="") -> UserProfile:  # type: ignore[no-untyped-def]
        captured["resume_name"] = resume_name
        return UserProfile(first_name="Sam", last_name="Sample", email="s@e.com", phone="")

    monkeypatch.setattr(
        "job_applicator.utils.profile._load_user_profile",
        _capture_load,
    )
    settings = AppSettings(resume_path="/r.pdf", output_dir=str(tmp_path / "out"))

    result = await actions.cover_letter_job(settings, _job(1))
    assert captured["resume_name"] == "Sam Sample"
    assert "Sam Sample" in result.cover_letter_text


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


async def test_tui_search_modal_collects_per_board_counts(tmp_path: Path) -> None:
    """The search modal collects a per-board result count into one SearchParams per selected
    board, each clamped to 1-50 (blank/unparseable -> default 25). With the default selection
    (LinkedIn on, Indeed off) it yields exactly one LinkedIn plan."""
    from textual.widgets import Input

    from job_applicator.tui.screens import SearchScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()

        async def linkedin_count(n: str) -> int:
            out: dict[str, object] = {}
            app.push_screen(SearchScreen(), lambda r: out.__setitem__("r", r))
            await pilot.pause()
            app.screen.query_one("#q", Input).value = "python"
            app.screen.query_one("#n_linkedin", Input).value = n
            app.screen._submit()  # type: ignore[attr-defined]
            await pilot.pause()
            plans = out["r"]
            assert isinstance(plans, list) and len(plans) == 1  # LinkedIn on, Indeed off
            assert plans[0].board is JobBoard.LINKEDIN
            return plans[0].max_results

        assert await linkedin_count("40") == 40  # honoured as entered
        assert await linkedin_count("100") == 50  # clamped to the cap
        assert await linkedin_count("0") == 1  # clamped to the floor
        assert await linkedin_count("") == 25  # blank -> default
        # The integer Input permits a bare "-"/"+" (restrict regex), which int() rejects;
        # the except-ValueError fallback must keep that from crashing the dismiss.
        assert await linkedin_count("-") == 25  # unparseable -> default (not a crash)


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
    monkeypatch.setattr(
        actions, "_score_jobs", AsyncMock(return_value=[_mr(jobs[0]), _mr(jobs[1])])
    )
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
    score = AsyncMock()
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
    monkeypatch.setattr(
        actions, "_score_jobs", AsyncMock(return_value=[_mr(jobs[0]), _mr(jobs[1])])
    )
    msgs: list[str] = []
    await actions.search_jobs(
        AppSettings(resume_path="/cv.pdf"),
        MagicMock(),
        SearchParams(query="python", board=JobBoard.LINKEDIN),
        on_progress=msgs.append,
    )
    joined = " | ".join(msgs)
    assert "Opening a browser" in joined and "Searching" in joined and "Scoring" in joined
    # Proper-cased board name in the phase messages (consistent with the scraper's "on
    # LinkedIn…" per-item line), not the lowercase enum value.
    assert "LinkedIn" in joined and "linkedin for" not in joined


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
    monkeypatch.setattr(
        actions, "_score_jobs", AsyncMock(return_value=[_mr(jobs[0]), _mr(jobs[1])])
    )
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
    monkeypatch.setattr(
        actions, "_score_jobs", AsyncMock(return_value=[_mr(jobs[0]), _mr(jobs[1])])
    )
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

    from textual.widgets import OptionList

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
        table = app.query_one("#joblist", OptionList)
        assert table.option_count == 0
        app._search_worker([SearchParams(query="x", board=JobBoard.LINKEDIN)])
        await pilot.pause()
        assert table.option_count == 1  # first row streamed in (not waiting for the whole scrape)
        assert table.loading is False  # spinner cleared on the first result
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert table.option_count == 2  # second row streamed in too


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
        app._search_worker([SearchParams(query="x", board=JobBoard.LINKEDIN)])
        await pilot.pause()
        line = app._statusline()
        assert "⏳" in line and "Scraping job 2/3" in line  # the per-item count is live
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._busy == ""  # cleared after the search


async def test_tui_busy_indicator_in_statusline(tmp_path: Path) -> None:
    """_set_busy shows a live '⏳ …' line in the status bar and clears back to the sort line."""
    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_busy("Tailoring X…")
        assert "⏳" in app._statusline() and "Tailoring X" in app._statusline()
        app._set_busy("")
        assert "⏳" not in app._statusline()  # restored to the sort/filter line


async def test_tui_search_clears_busy_and_loading(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """After a search, the list spinner (loading) and the busy line are both cleared."""
    from textual.widgets import OptionList

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
        assert app.query_one("#joblist", OptionList).loading is False


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


async def test_tui_modal_visible_with_animations_disabled(tmp_path: Path) -> None:
    """Reduced-motion (animations off): the fade must degrade to an INSTANT show — the modal
    is fully visible, never stuck transparent. Guards the disabled-animation path against a
    future Textual change (today it instant-sets to the final value)."""
    from job_applicator.tui.screens import HelpScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        app.animation_level = "none"  # as if TEXTUAL_ANIMATIONS=none / reduced motion
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        assert app.screen.styles.opacity == 1.0  # instant-shown, not stuck at 0


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

    async def _fake_apply(
        _settings: object, job: object, *, submit: bool, cover_letter: str | None = None
    ) -> ApplicationResult:
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


async def test_tui_apply_cover_letter_bound_to_job_not_current(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """C5 + divergence guard: TUI apply attaches the cover letter generated for THE JOB BEING
    APPLIED (matching the CLI), captured WITH the job at action time — so even if the selection
    (_current) drifts while the modal is open, the job's own letter is used, never another's."""
    from datetime import UTC, datetime

    from job_applicator.models import ApplicationResult, ApplicationStatus
    from job_applicator.tui import actions

    letter_a = tmp_path / "a.txt"
    letter_a.write_text("LETTER-A", encoding="utf-8")
    letter_b = tmp_path / "b.txt"
    letter_b.write_text("LETTER-B", encoding="utf-8")
    store = JobStore(db_path=tmp_path / "applications.db")
    store.mark_tailored(_job(1), tailored_resume_path="/t.txt", cover_letter_path=str(letter_a))
    store.mark_tailored(_job(2), tailored_resume_path="/t.txt", cover_letter_path=str(letter_b))

    captured: dict[str, object] = {}

    async def _fake_apply(_s: object, job: object, *, submit: bool, cover_letter=None):  # type: ignore[no-untyped-def]
        captured["cover_letter"] = cover_letter
        return ApplicationResult(
            job=job, status=ApplicationStatus.PENDING, timestamp=datetime.now(UTC)
        )  # type: ignore[arg-type]

    monkeypatch.setattr(actions, "apply_job", _fake_apply)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app._current = store.get(str(_job(1).url))  # select job A
        await pilot.press("a")  # modal captures A + A's cover-letter path
        await pilot.pause()
        app._current = store.get(str(_job(2).url))  # selection drifts to B while the modal is open
        await pilot.click("#go")  # dry run (checkbox unchecked)
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert captured["cover_letter"] == "LETTER-A"  # A's own letter, not B's


async def test_tui_apply_real_submit_requires_checkbox(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Ticking the danger checkbox before Apply sends a real submit (submit=True)."""
    from datetime import UTC, datetime

    from job_applicator.models import ApplicationResult, ApplicationStatus
    from job_applicator.tui import actions

    captured: dict[str, bool] = {}

    async def _fake_apply(
        _settings: object, job: object, *, submit: bool, cover_letter: str | None = None
    ) -> ApplicationResult:
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
            has_applied=lambda url, **kw: False, count_today=lambda board=None: 999
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

    async def _boom(
        _settings: object, _job: object, *, style_guide_path: str = "", **kw: object
    ) -> object:
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
        lambda *a, **k: MagicMock(
            has_applied=lambda url, **kw: True, count_today=lambda board=None: 0
        ),
    )
    result = await actions.apply_job(AppSettings(), _job(1), submit=True)
    assert result.status is ApplicationStatus.ALREADY_APPLIED
    make_browser.assert_not_called()


async def test_apply_job_dedups_already_applied_status_like_cli(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """C6: the TUI dedups on ALREADY_APPLIED too (not just SUBMITTED), matching the CLI — exercised
    against the REAL has_applied status set, with a recorded ALREADY_APPLIED outcome."""
    import job_applicator.factories as factories
    from job_applicator.models import ApplicationResult, ApplicationStatus
    from job_applicator.state import ApplicationState
    from job_applicator.tui import actions

    # A prior ALREADY_APPLIED outcome (e.g. the applicator detected an existing application).
    ApplicationState().record(
        ApplicationResult(job=_job(1), status=ApplicationStatus.ALREADY_APPLIED)
    )
    make_browser = MagicMock()
    monkeypatch.setattr(factories, "_make_browser", make_browser)
    result = await actions.apply_job(AppSettings(), _job(1), submit=True)
    assert result.status is ApplicationStatus.ALREADY_APPLIED  # skipped, not re-attempted
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


# --------------------------------------------- stage filter + sort (Cycle D)
def _staged_app(tmp_path: Path, *, applied: tuple[int, ...] = ()) -> JobApplicatorApp:
    """An app seeded with one job at each head stage — found(1), matched(2), tailored(3),
    cover_letter(4) — plus optional SUBMITTED 'applied' overlay for the given job numbers."""
    from datetime import UTC, datetime

    from job_applicator.models import ApplicationResult, ApplicationStatus

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))  # found
    store.upsert_match(_mr(_job(2)))  # matched
    store.mark_tailored(_job(3), tailored_resume_path="/out/t.txt")  # tailored
    store.mark_tailored(  # cover_letter
        _job(4), tailored_resume_path="/out/t.txt", cover_letter_path="/out/c.txt"
    )
    app_state = MagicMock()
    app_state.list_recent.return_value = [
        ApplicationResult(
            job=_job(n), status=ApplicationStatus.SUBMITTED, timestamp=datetime.now(UTC)
        )
        for n in applied
    ]
    return JobApplicatorApp(
        settings=AppSettings(resume_path="/cv/r.pdf"), store=store, app_state=app_state
    )


async def test_tui_stage_filter_cycles_through_stages(tmp_path: Path) -> None:
    """`f` cycles the funnel-stage filter (all → found → matched → … → applied → all),
    narrowing the list to one stage at a time and wrapping back to the full list."""
    app = _staged_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", OptionList)
        assert table.option_count == 4  # all stages shown
        app.action_cycle_stage_filter()  # → found
        await pilot.pause()
        assert app._stage_filter == "found" and table.option_count == 1
        assert app._current is not None and app._current.job.company == "Co1"
        app.action_cycle_stage_filter()  # → matched
        await pilot.pause()
        assert app._stage_filter == "matched" and table.option_count == 1
        assert app._current is not None and app._current.job.company == "Co2"
        for _ in range(4):  # matched → tailored → cover_letter → applied → all
            app.action_cycle_stage_filter()
        await pilot.pause()
        assert app._stage_filter is None and table.option_count == 4  # full circle


async def test_tui_stage_filter_composes_with_text_filter(tmp_path: Path) -> None:
    """The stage filter and the text filter narrow together (logical AND)."""
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1, company="Acme"))  # found
    store.upsert_job(_job(2, company="Globex"))  # found
    store.upsert_match(_mr(_job(3, company="Acme")))  # matched
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", OptionList)
        app._filter = "acme"  # text filter → jobs 1 (found) + 3 (matched)
        app._repaint()
        assert table.option_count == 2
        app.action_cycle_stage_filter()  # add stage filter → found → only job 1
        await pilot.pause()
        assert app._stage_filter == "found" and table.option_count == 1
        assert app._current is not None and app._current.job.company == "Acme"


async def test_tui_stage_filter_respects_applied_overlay(tmp_path: Path) -> None:
    """A SUBMITTED job shows under the 'applied' filter — not its store head stage —
    matching how the status-line counts compose the ApplicationState overlay."""
    app = _staged_app(tmp_path, applied=(4,))  # job 4 (cover_letter in store) is SUBMITTED
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", OptionList)
        assert app._applied_urls == {"https://linkedin.com/jobs/4"}
        app._stage_filter = "cover_letter"  # job 4 reads as applied now → not here
        app._repaint()
        assert table.option_count == 0
        app._stage_filter = "applied"  # … it shows here instead
        app._repaint()
        assert table.option_count == 1
        assert app._current is not None and app._current.job.company == "Co4"


async def test_tui_escape_clears_stage_and_text_filters(tmp_path: Path) -> None:
    """Esc resets BOTH the stage filter and the text filter — back to the full list."""
    app = _staged_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", OptionList)
        app.action_cycle_stage_filter()  # stage → found
        await pilot.pause()
        app._filter = "co1"  # and a text filter
        app._repaint()
        assert app._stage_filter == "found" and app._filter == "co1"
        await pilot.press("escape")
        await pilot.pause()
        assert app._stage_filter is None and app._filter == ""
        assert table.option_count == 4  # both cleared → all jobs


async def test_tui_sort_cycles_and_reorders(tmp_path: Path) -> None:
    """`S` cycles best-match → recent → funnel-stage → salary → best-match, reordering."""
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_match(_mr(_job(1), score=0.2))  # matched, scored, seeded FIRST (older)
    store.upsert_job(_job(2))  # found, unscored, seeded LAST (newer)
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._sort_mode == "match"  # scored job first
        assert app._visible()[0].job.company == "Co1"
        await pilot.press("S")  # → recent (job 2 updated last)
        await pilot.pause()
        assert app._sort_mode == "recent" and app._visible()[0].job.company == "Co2"
        await pilot.press("S")  # → funnel stage (found sorts before matched)
        await pilot.pause()
        assert app._sort_mode == "stage" and app._visible()[0].job.company == "Co2"
        await pilot.press("S")  # → salary (neither job lists pay; mode still advances)
        await pilot.pause()
        assert app._sort_mode == "salary"
        await pilot.press("S")  # → back to match
        await pilot.pause()
        assert app._sort_mode == "match" and app._visible()[0].job.company == "Co1"


async def test_tui_empty_stage_shows_guidance(tmp_path: Path) -> None:
    """Filtering to a stage with no jobs shows guidance (not 'No jobs yet') pointing at the
    keys to change or clear the filter."""
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))  # only a 'found' job
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app._stage_filter = "applied"  # no applied jobs
        app._repaint()
        assert app.query_one("#joblist", OptionList).option_count == 0
        detail = app._detail_markup(None)
        assert "No jobs match the current filter" in detail
        assert "[cyan]f[/cyan]" in detail  # points at the stage-filter key, not 'No jobs yet'


def test_tui_statusline_reflects_sort_and_filters() -> None:
    """The status line shows the active sort + the board/salary/text filters (+ shown count).
    It does NOT show the funnel counts or the stage — those are the tabs now."""
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    app._all = []
    line = app._statusline()  # defaults
    assert "sort: best match" in line
    assert "stage:" not in line  # stage is the active tab, not the status line
    app._sort_mode = "recent"
    app._board_filter = "indeed"
    line = app._statusline()
    assert "sort: recent" in line and "board: Indeed" in line and "shown" in line


async def test_tui_statusline_renders_sort_stage_in_running_frame(tmp_path: Path) -> None:
    """Real-frame guard (a string method can pass while the on-screen frame is wrong): after
    real keypresses the RUNNING status-line Static renders the sort, then the stage filter."""
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/cv/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )

    def _rendered() -> str:
        from textual.widgets import Static

        r = app.query_one("#statusline", Static).render()
        return r.plain if hasattr(r, "plain") else str(r)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert "sort: best match" in _rendered()  # sort shown by default
        await pilot.press("S")  # → recent
        await pilot.pause()
        assert "sort: recent" in _rendered()
        app.action_cycle_stage_filter()  # stage → found: moves the active TAB, not the status line
        await pilot.pause()
        from textual.widgets import Tabs

        assert app.query_one("#stagetabs", Tabs).active == "stage-found"
        assert "stage:" not in _rendered()  # stage lives on the tab now, not the status


async def test_tui_salary_sort_filter_and_toggle(tmp_path: Path) -> None:
    """Salary sort ranks high→low with unlisted last; the min-salary filter KEEPS unlisted
    jobs, and only the separate 'hide unlisted' toggle removes them."""
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1, salary="$150,000 a year"))  # high
    store.upsert_job(_job(2, salary="$50,000 a year"))  # low
    store.upsert_job(_job(3, salary=None))  # unlisted
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app._sort_mode = "salary"
        app._reload()
        await pilot.pause()
        order = [s.job.title for s in app._visible()]
        assert order[0] == "Engineer 1"  # highest salary first
        assert order[-1] == "Engineer 3"  # unlisted sorts last
        # min $100k drops the $50k job but KEEPS the unlisted one
        app._min_salary = 100_000
        app._repaint()
        await pilot.pause()
        assert {s.job.title for s in app._visible()} == {"Engineer 1", "Engineer 3"}
        # the hide-unlisted toggle now removes Engineer 3
        app._hide_unlisted = True
        app._repaint()
        await pilot.pause()
        assert {s.job.title for s in app._visible()} == {"Engineer 1"}


async def test_tui_salary_keys_status_and_esc(tmp_path: Path) -> None:
    """Real-frame: `m` cycles the floor (shown in the status line), `u` hides unlisted-pay
    jobs, and Esc resets both back to the full list."""
    from textual.widgets import Static

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1, salary="$150,000 a year"))
    store.upsert_job(_job(2, salary=None))
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/cv/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )

    def _rendered() -> str:
        r = app.query_one("#statusline", Static).render()
        return r.plain if hasattr(r, "plain") else str(r)

    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#joblist", OptionList)
        await pilot.press("m")  # min salary → $40k
        await pilot.pause()
        assert app._min_salary == 40_000
        assert "min $40k" in _rendered()
        assert table.option_count == 2  # $150k clears it; the unlisted job is kept
        await pilot.press("u")  # hide unlisted-pay jobs
        await pilot.pause()
        assert app._hide_unlisted is True
        assert "listed pay only" in _rendered()
        assert table.option_count == 1  # only the job with listed pay remains
        await pilot.press("escape")  # resets ALL filters
        await pilot.pause()
        assert app._min_salary == 0 and app._hide_unlisted is False
        assert table.option_count == 2


# ----------------------------------------- Indeed + multi-board search (Cycle E)
async def test_tui_search_modal_multi_board_yields_one_plan_per_board(tmp_path: Path) -> None:
    """Selecting both boards yields one SearchParams per board — each with that board's own
    count — and the shared query/location/remote copied to both."""
    from textual.widgets import Checkbox, Input

    from job_applicator.tui.screens import SearchScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        out: dict[str, object] = {}
        app.push_screen(SearchScreen(), lambda r: out.__setitem__("r", r))
        await pilot.pause()
        scr = app.screen
        scr.query_one("#q", Input).value = "python"
        scr.query_one("#loc", Input).value = "Berlin"
        scr.query_one("#remote", Checkbox).value = True
        scr.query_one("#bd_indeed", Checkbox).value = True  # LinkedIn already on by default
        scr.query_one("#n_linkedin", Input).value = "10"
        scr.query_one("#n_indeed", Input).value = "5"
        scr._submit()  # type: ignore[attr-defined]
        await pilot.pause()
        plans = out["r"]
        assert isinstance(plans, list) and len(plans) == 2
        by_board = {p.board: p for p in plans}
        assert by_board[JobBoard.LINKEDIN].max_results == 10
        assert by_board[JobBoard.INDEED].max_results == 5
        for p in plans:  # shared params copied to every board
            assert p.query == "python" and p.location == "Berlin" and p.remote_only is True


async def test_tui_search_modal_requires_a_board(tmp_path: Path) -> None:
    """Unchecking both boards refuses to submit — nothing is dismissed and the modal stays
    open (so a search can never run with no board selected)."""
    from textual.widgets import Checkbox, Input

    from job_applicator.tui.screens import SearchScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        out: dict[str, object] = {"r": "untouched"}
        app.push_screen(SearchScreen(), lambda r: out.__setitem__("r", r))
        await pilot.pause()
        scr = app.screen
        scr.query_one("#q", Input).value = "python"
        scr.query_one("#bd_linkedin", Checkbox).value = False  # both boards off now
        scr._submit()  # type: ignore[attr-defined]
        await pilot.pause()
        assert isinstance(app.screen, SearchScreen)  # still open, not dismissed
        assert out["r"] == "untouched"  # nothing handed back
        assert "at least one board" in scr._warning_text().lower()  # type: ignore[attr-defined]


async def test_tui_search_modal_warning_is_board_aware(tmp_path: Path) -> None:
    """The warning reflects exactly the selected boards: the real-account warning for
    LinkedIn, the public note for Indeed, a prompt when neither."""
    from textual.widgets import Checkbox

    from job_applicator.tui.screens import SearchScreen

    app = _app(tmp_path, seed=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(SearchScreen(), lambda r: None)
        await pilot.pause()
        scr = app.screen
        w = scr._warning_text()  # type: ignore[attr-defined]   # default: LinkedIn on only
        assert "real account" in w and "public" not in w
        scr.query_one("#bd_indeed", Checkbox).value = True  # both on
        w = scr._warning_text()  # type: ignore[attr-defined]
        assert "real account" in w and "public" in w
        scr.query_one("#bd_linkedin", Checkbox).value = False  # none on
        scr.query_one("#bd_indeed", Checkbox).value = False
        assert "at least one board" in scr._warning_text().lower()  # type: ignore[attr-defined]


async def test_tui_search_worker_runs_each_selected_board(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The worker scrapes every selected board in ONE account worker, calling search_jobs once
    per board (in order) with that board's params; the final toast names the boards + total."""
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    calls: list[tuple[str, int]] = []

    async def _fake(_s, st, params, on_progress=None, on_job=None) -> int:  # type: ignore[no-untyped-def]
        calls.append((params.board.value, params.max_results))
        return 3

    monkeypatch.setattr(actions, "search_jobs", _fake)
    app = _app(tmp_path, seed=0)  # empty store
    toasts: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "notify", lambda msg, **k: toasts.append(msg))
        app._search_worker(
            [
                SearchParams(query="x", board=JobBoard.LINKEDIN, max_results=10),
                SearchParams(query="x", board=JobBoard.INDEED, max_results=5),
            ]
        )
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert calls == [("linkedin", 10), ("indeed", 5)]  # both boards, in order, with counts
    # The summary breaks the total down PER BOARD (not a lumped "across LinkedIn, Indeed").
    assert toasts and "6 job(s)" in toasts[-1]
    assert "LinkedIn: 3" in toasts[-1] and "Indeed: 3" in toasts[-1]


async def test_tui_search_worker_one_board_failing_does_not_abort_the_other(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """If a board errors (e.g. Indeed can't clear Cloudflare), it's toasted and skipped — the
    other board's results still land, and the app does not crash."""
    from job_applicator.exceptions import JobApplicatorError
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    async def _fake(_s, st, params, on_progress=None, on_job=None) -> int:  # type: ignore[no-untyped-def]
        if params.board is JobBoard.INDEED:
            raise JobApplicatorError("Indeed: could not clear the challenge")
        return 4

    monkeypatch.setattr(actions, "search_jobs", _fake)
    app = _app(tmp_path, seed=0)
    toasts: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "notify", lambda msg, **k: toasts.append(msg))
        app._search_worker(
            [
                SearchParams(query="x", board=JobBoard.LINKEDIN, max_results=10),
                SearchParams(query="x", board=JobBoard.INDEED, max_results=10),
            ]
        )
        await app.workers.wait_for_complete()
        await pilot.pause()
    # one toast for the Indeed failure, one summary naming the failed board; LinkedIn's 4 land
    assert any("Indeed" in t and "could not clear" in t for t in toasts)
    assert any("4 job(s)" in t and "Indeed: failed" in t for t in toasts)


async def test_tui_search_worker_zero_result_board_is_visible_not_credited(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """The reported bug: Indeed returns 0 (blocked UPSTREAM of extraction → 0 cards, which is
    NOT an error) while LinkedIn returns 10. The summary must show 'Indeed: 0' so the empty
    board is visible — never fold it into a lumped total that credits Indeed as succeeded."""
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    async def _fake(_s, st, params, on_progress=None, on_job=None) -> int:  # type: ignore[no-untyped-def]
        return 10 if params.board is JobBoard.LINKEDIN else 0

    monkeypatch.setattr(actions, "search_jobs", _fake)
    app = _app(tmp_path, seed=0)
    toasts: list[tuple[str, str]] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(
            app, "notify", lambda msg, **k: toasts.append((msg, str(k.get("severity", ""))))
        )
        app._search_worker(
            [
                SearchParams(query="x", board=JobBoard.LINKEDIN, max_results=10),
                SearchParams(query="x", board=JobBoard.INDEED, max_results=10),
            ]
        )
        await app.workers.wait_for_complete()
        await pilot.pause()
    summary, severity = toasts[-1]
    assert "10 job(s)" in summary  # the running total still reflects only what actually landed
    assert "LinkedIn: 10" in summary
    assert "Indeed: 0" in summary  # the empty board is SURFACED, not silently dropped
    assert severity == "warning"  # a 0-count board flags the summary so the user notices


async def test_tui_search_modal_layout_fits_and_aligns(tmp_path: Path) -> None:
    """Real-frame layout guard (a green widget-exists test can pass while the frame is broken):
    the board rows align (checkbox + count input), the counts are visible (non-zero width),
    and the modal box never exceeds the screen — it SCROLLS rather than clipping the buttons
    on a short terminal."""
    from textual.containers import VerticalScroll
    from textual.widgets import Checkbox, Input

    from job_applicator.tui.screens import SearchScreen

    # Short terminal: the box must fit the screen and scroll (buttons reachable, not clipped).
    app = _app(tmp_path, seed=0)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.push_screen(SearchScreen(), lambda r: None)
        await pilot.pause()
        box = app.screen.query_one("#searchbox", VerticalScroll)
        assert box.region.bottom <= 24 and box.region.y >= 0  # box fits the screen
        assert box.virtual_size.height > box.size.height  # too tall → scrolls, not clipped

    # Normal terminal: rows align, counts are visible, buttons sit in the viewport.
    app2 = _app(tmp_path, seed=0)
    async with app2.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app2.push_screen(SearchScreen(), lambda r: None)
        await pilot.pause()
        scr = app2.screen
        cbL = scr.query_one("#bd_linkedin", Checkbox).region
        cbI = scr.query_one("#bd_indeed", Checkbox).region
        nL = scr.query_one("#n_linkedin", Input).region
        nI = scr.query_one("#n_indeed", Input).region
        assert cbL.x == cbI.x  # board checkboxes aligned
        assert nL.x == nI.x and nL.width > 0 and nI.width > 0  # counts aligned + visible
        assert nL.x >= cbL.right  # the count sits to the right of its checkbox
        assert scr.query_one("#buttons").region.bottom <= 40  # buttons visible without scroll


async def test_tui_search_then_refuses_a_second_concurrent_account_worker(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """Defense-in-depth (account safety): if a second search is dispatched while one is already
    running (e.g. two modals stacked), _search_then refuses it — never two account workers /
    two browsers at once."""
    import asyncio

    from job_applicator.scrapers.base import SearchParams
    from job_applicator.tui import actions

    gate = asyncio.Event()
    calls: list[str] = []

    async def _fake(_s, st, params, on_progress=None, on_job=None) -> int:  # type: ignore[no-untyped-def]
        calls.append(params.board.value)
        await gate.wait()  # hold the first worker running
        return 0

    monkeypatch.setattr(actions, "search_jobs", _fake)
    app = _app(tmp_path, seed=0)
    toasts: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "notify", lambda msg, **k: toasts.append(msg))
        app._search_then([SearchParams(query="x", board=JobBoard.LINKEDIN)])  # worker A starts
        await pilot.pause()
        assert app._account_busy()
        app._search_then([SearchParams(query="y", board=JobBoard.INDEED)])  # must be refused
        await pilot.pause()
        assert calls == ["linkedin"]  # the second search never started
        assert any("already running" in t for t in toasts)
        gate.set()
        await app.workers.wait_for_complete()


# ------------------------------------------------ board column + filter (Cycle F)
async def test_tui_board_badge_in_each_card(tmp_path: Path) -> None:
    """Each job card carries a colour-coded, FULL-NAME board badge (LinkedIn / Indeed) on its
    metadata line — distinct colours for an at-a-glance scan."""
    from textual.widgets import OptionList

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))  # LinkedIn (the _job default)
    store.upsert_job(_job(2, url="https://indeed.com/jobs/2", board=JobBoard.INDEED))
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        t = app.query_one("#joblist", OptionList)
        cards = [str(t.get_option_at_index(r).prompt) for r in range(t.option_count)]
        assert any("LinkedIn" in c for c in cards) and any("Indeed" in c for c in cards)


async def test_tui_board_filter_cycles_and_narrows(tmp_path: Path) -> None:
    """`b` cycles the board filter (all → linkedin → indeed → all), narrowing the list."""
    from textual.widgets import OptionList

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))  # LinkedIn
    store.upsert_job(_job(2, url="https://indeed.com/jobs/2", board=JobBoard.INDEED))
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        t = app.query_one("#joblist", OptionList)
        assert t.option_count == 2  # all boards
        await pilot.press("b")  # → linkedin
        await pilot.pause()
        assert app._board_filter == "linkedin" and t.option_count == 1
        assert app._current is not None and app._current.job.board is JobBoard.LINKEDIN
        await pilot.press("b")  # → indeed
        await pilot.pause()
        assert app._board_filter == "indeed" and t.option_count == 1
        assert app._current is not None and app._current.job.board is JobBoard.INDEED
        await pilot.press("b")  # → all
        await pilot.pause()
        assert app._board_filter is None and t.option_count == 2


async def test_tui_board_filter_composes_with_stage_filter(tmp_path: Path) -> None:
    """Board + stage filters narrow together (logical AND)."""
    from textual.widgets import OptionList

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))  # LinkedIn, found
    store.upsert_match(
        _mr(_job(2, url="https://indeed.com/2", board=JobBoard.INDEED))
    )  # Indeed, matched
    store.upsert_match(_mr(_job(3)))  # LinkedIn, matched
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        t = app.query_one("#joblist", OptionList)
        app._board_filter = "linkedin"
        app._stage_filter = "matched"
        app._repaint()
        assert t.option_count == 1  # LinkedIn AND matched → only job 3
        assert app._current is not None and app._current.job.company == "Co3"


async def test_tui_escape_clears_board_filter_too(tmp_path: Path) -> None:
    """Esc resets the board filter along with the stage + text filters."""
    from textual.widgets import OptionList

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    store.upsert_job(_job(2, url="https://indeed.com/jobs/2", board=JobBoard.INDEED))
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        t = app.query_one("#joblist", OptionList)
        await pilot.press("b")  # board → linkedin
        await pilot.pause()
        assert app._board_filter == "linkedin" and t.option_count == 1
        await pilot.press("escape")
        await pilot.pause()
        assert app._board_filter is None and t.option_count == 2


def test_tui_statusline_shows_board_filter() -> None:
    """The status line shows the active board filter (+ the narrowed shown-count)."""
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    app._all = []
    app._board_filter = "indeed"
    line = app._statusline()
    assert "board: Indeed" in line and "shown" in line


async def test_tui_joblist_cards_wrap_no_h_overflow(tmp_path: Path) -> None:
    """Real-frame guard: the cards WRAP to the list pane, so even a very long title/company
    never causes horizontal scroll (the OptionList fills width and folds long lines)."""
    from textual.widgets import OptionList

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1, title="Senior Platform Engineer", company="Globex International"))
    store.upsert_job(
        _job(2, url="https://indeed.com/2", board=JobBoard.INDEED, title="X" * 140)  # very long
    )
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        t = app.query_one("#joblist", OptionList)
        assert t.option_count == 2
        assert t.virtual_size.width <= t.size.width  # wraps; no horizontal scroll even at 140ch


# ----------------------------------------------------------------- style-guide UI flow
async def test_tui_set_style_guide_persists_config(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`g` → style-guide modal → submit sets the session style guide AND writes config.toml."""
    from textual.widgets import Input

    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app = JobApplicatorApp(
        settings=AppSettings(),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("g")
        await pilot.pause()
        app.screen.query_one("#path", Input).value = "/style/example.pdf"
        await pilot.press("enter")
        await pilot.pause()
    assert app._settings.style_guide_path == "/style/example.pdf"
    assert cfg.exists() and 'style_guide_path = "/style/example.pdf"' in cfg.read_text()


async def test_tui_style_guide_status_line(tmp_path: Path) -> None:
    """The status line shows the configured style guide (basename) or the unset hint."""
    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/docs/cv.pdf", style_guide_path="/docs/style.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        line = app._statusline()
        assert "Style:" in line
        assert "style.pdf" in line and "/docs/" not in line  # basename only, not the full path


def test_tui_statusline_style_guide_unset_hint() -> None:
    """When no style guide is set, the status line hints at the `g` key."""
    app = JobApplicatorApp(settings=AppSettings(), store=MagicMock(), app_state=MagicMock())
    line = app._statusline()
    assert "Style:" in line
    assert "press 'g'" in line


async def test_tui_tailor_action_passes_style_guide(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pressing `t` passes the configured style_guide_path into the TUI tailor action."""
    from job_applicator.models import FunnelStatus, TailoredResume
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    captured: dict[str, str] = {}
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

    async def _fake_tailor(  # type: ignore[no-untyped-def]
        settings: AppSettings,
        job: JobListing,
        *,
        style_guide_path: str = "",
        **kw: object,
    ) -> TailoredResume:
        captured["style_guide_path"] = style_guide_path
        return fake

    monkeypatch.setattr(actions, "tailor_job", _fake_tailor)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf", style_guide_path="/style.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("t")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert captured.get("style_guide_path") == "/style.pdf"
    got = store.get("https://linkedin.com/jobs/1")
    assert got is not None and got.funnel_status is FunnelStatus.TAILORED


async def test_tui_cover_letter_action_passes_style_guide(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pressing `c` passes the configured style_guide_path into the cover-letter action."""
    from job_applicator.models import CoverLetterResult, FunnelStatus
    from job_applicator.tui import actions

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    captured: dict[str, str] = {}
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
        settings: AppSettings,
        job: JobListing,
        tailored_resume_path: str = "",
        *,
        style_guide_path: str = "",
        **kw: object,
    ) -> CoverLetterResult:
        captured["style_guide_path"] = style_guide_path
        return fake

    monkeypatch.setattr(actions, "cover_letter_job", _fake_cl)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf", style_guide_path="/style.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("c")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert captured.get("style_guide_path") == "/style.pdf"
    got = store.get("https://linkedin.com/jobs/1")
    assert got is not None and got.funnel_status is FunnelStatus.COVER_LETTER


async def test_tui_style_guide_cancel_keeps_previous(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Cancelling the style-guide modal leaves the previous path unchanged."""
    from textual.widgets import Input

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    app = JobApplicatorApp(
        settings=AppSettings(style_guide_path="/existing.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("g")
        await pilot.pause()
        assert app.screen.query_one("#path", Input).value == "/existing.pdf"
        await pilot.press("escape")
        await pilot.pause()
    assert app._settings.style_guide_path == "/existing.pdf"


def test_tui_help_includes_style_guide_key() -> None:
    """The in-app help references the new `g` style-guide key."""
    from job_applicator.tui.screens import HelpScreen

    screen = HelpScreen()
    assert "set style-guide" in screen._HELP.lower()


async def test_tui_tailor_pdf_action_records_pdf_path(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Pressing `T` renders a PDF résumé and records its path in the store."""
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
        output_path="/out/tailored.pdf",
        pdf_path="/out/tailored.pdf",
    )

    async def _fake_tailor_pdf(
        _settings: object,
        _job: object,
        *,
        style_guide_path: str = "",
        output_format: object = None,
        **kw: object,
    ) -> TailoredResume:
        assert output_format is not None and output_format.value == "pdf"
        return fake

    monkeypatch.setattr(actions, "tailor_job", _fake_tailor_pdf)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("T")
        await app.workers.wait_for_complete()
        await pilot.pause()
    got = store.get("https://linkedin.com/jobs/1")
    assert got is not None and got.funnel_status is FunnelStatus.TAILORED
    assert got.pdf_path == "/out/tailored.pdf"


async def test_tui_cover_letter_pdf_action_records_pdf_path(  # type: ignore[no-untyped-def]
    tmp_path: Path, monkeypatch
) -> None:
    """Pressing `C` renders a PDF cover letter and records its path in the store."""
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
        output_path="/out/cover.pdf",
        pdf_path="/out/cover.pdf",
    )

    async def _fake_cl_pdf(
        _settings: object,
        _job: object,
        tailored_resume_path: str = "",
        *,
        style_guide_path: str = "",
        output_format: object = None,
        **kw: object,
    ) -> CoverLetterResult:
        assert output_format is not None and output_format.value == "pdf"
        return fake

    monkeypatch.setattr(actions, "cover_letter_job", _fake_cl_pdf)
    app = JobApplicatorApp(
        settings=AppSettings(resume_path="/r.pdf"),
        store=store,
        app_state=MagicMock(list_recent=lambda **k: []),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("C")
        await app.workers.wait_for_complete()
        await pilot.pause()
    got = store.get("https://linkedin.com/jobs/1")
    assert got is not None and got.funnel_status is FunnelStatus.COVER_LETTER
    assert got.cover_letter_pdf_path == "/out/cover.pdf"


def test_tui_help_includes_pdf_keys() -> None:
    """The in-app help references the new PDF action keys."""
    from job_applicator.tui.screens import HelpScreen

    screen = HelpScreen()
    assert "tailor résumé pdf" in screen._HELP.lower()
    assert "cover letter pdf" in screen._HELP.lower()
    assert "open generated pdf" in screen._HELP.lower()


async def test_tui_cursor_preserved_across_refresh_and_sort(tmp_path: Path) -> None:
    """Refresh / sort / filter rebuild the table; the SELECTED job must stay selected (by row
    key) instead of snapping back to the first listing — the reported 'r/S just jump to the
    first job' bug."""
    app = _app(tmp_path, seed=3)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("j")  # move off the first row
        await pilot.pause()
        selected = app._current.id if app._current else None
        assert selected is not None
        await pilot.press("r")  # refresh
        await pilot.pause()
        assert app._current is not None and app._current.id == selected  # kept selection
        await pilot.press("S")  # cycle sort — same job stays selected, not reset to row 0
        await pilot.pause()
        assert app._current is not None and app._current.id == selected


async def test_tui_stage_tabs_filter_sync_and_counts(tmp_path: Path) -> None:
    """The stage tabs show per-stage counts, filter on click, mirror the `f` cycle, and reset
    on Esc — the 'tabs + visual hierarchy' redesign (increment 1)."""
    from textual.widgets import Tab, Tabs

    from job_applicator.tui.app import _stage_to_tab

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))
    store.upsert_job(_job(2))
    store.upsert_match(_mr(_job(3)))  # one matched job
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        tabs = app.query_one("#stagetabs", Tabs)
        table = app.query_one("#joblist", OptionList)
        assert tabs.active == "stage-all"
        assert "3" in str(app.query_one("#stage-all", Tab).label)  # All shows the total
        assert "1" in str(app.query_one("#stage-matched", Tab).label)  # Matched count
        tabs.active = "stage-matched"  # click a tab → filters
        await pilot.pause()
        assert app._stage_filter == "matched" and table.option_count == 1
        app.action_cycle_stage_filter()  # the f cycle moves the active tab too (no loop)
        await pilot.pause()
        assert tabs.active == _stage_to_tab(app._stage_filter)
        await pilot.press("escape")  # Esc resets the tab to All
        await pilot.pause()
        assert app._stage_filter is None and tabs.active == "stage-all" and table.option_count == 3


async def test_tui_filter_modal_applies_view_controls(tmp_path: Path) -> None:
    """`f` opens the grouped Filter & sort panel; applying it sets board/sort (etc.) at once and
    re-queries — the footer no longer needs the individual b/m/S cycle keys."""
    from textual.widgets import Select

    from job_applicator.tui.app import JobList
    from job_applicator.tui.screens import FilterScreen

    store = JobStore(db_path=tmp_path / "applications.db")
    store.upsert_job(_job(1))  # LinkedIn
    store.upsert_job(_job(2, url="https://indeed.com/2", board=JobBoard.INDEED))
    app = JobApplicatorApp(
        settings=AppSettings(), store=store, app_state=MagicMock(list_recent=lambda **k: [])
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f")
        await pilot.pause()
        screen = app.screen_stack[-1]
        assert isinstance(screen, FilterScreen)
        screen.query_one("#f_board", Select).value = "indeed"
        screen.query_one("#f_sort", Select).value = "recent"
        await pilot.pause()
        screen._submit()
        await pilot.pause()
        assert app._board_filter == "indeed" and app._sort_mode == "recent"
        assert app.query_one("#joblist", JobList).option_count == 1  # only the Indeed job
        # the footer is lean: the per-filter cycle keys are no longer shown (they live in `f`)
        shown = {b.key for b in app.BINDINGS if getattr(b, "show", True)}
        assert "f" in shown and not ({"b", "m", "S"} & shown)


async def test_tui_filter_modal_cancel_keeps_state(tmp_path: Path) -> None:
    """Cancelling the Filter panel (Esc/None) changes nothing."""
    from job_applicator.tui.screens import FilterScreen

    app = _app(tmp_path, seed=3)
    async with app.run_test() as pilot:
        await pilot.pause()
        before = (app._board_filter, app._sort_mode, app._min_salary)
        await pilot.press("f")
        await pilot.pause()
        assert isinstance(app.screen_stack[-1], FilterScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert (app._board_filter, app._sort_mode, app._min_salary) == before


def test_score_style_sentiment_palette() -> None:
    """The match score gets a sentiment colour: green (strong) / yellow (moderate) / red (weak),
    and dim when unscored."""
    from job_applicator.tui.app import _score_style

    assert _score_style(0.82) == "bold green"
    assert _score_style(0.70) == "bold green"
    assert _score_style(0.58) == "bold yellow"
    assert _score_style(0.49) == "bold red"
    assert _score_style(None) == "dim"


def test_job_card_renders_row_divider_not_blank_gap() -> None:
    """Compact table look: each card ends with a $panel rule divider (a table-row separator),
    NOT a blank gap line — and the stage spine rail continues through the divider."""
    import io

    from rich.console import Console
    from rich.text import Text

    from job_applicator.tui.app import _JobCard

    c = Console(width=40, file=io.StringIO(), force_terminal=False)
    c.print(_JobCard([Text("Title"), Text("Co"), Text("61%  ·  LinkedIn")], "cyan", "#242f38"))
    lines = c.file.getvalue().rstrip("\n").split("\n")
    assert "─" in lines[-1]  # last line is the row divider
    assert all(line.strip() for line in lines)  # no blank gap line anywhere
    assert lines[-1].lstrip().startswith("▌")  # stage spine rail continues through the divider
