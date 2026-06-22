"""The Textual application: a navigable home over the job-funnel store.

Layout: a status line (résumé + funnel summary), a job-list sidebar (the funnel head from
``JobStore``), a detail pane for the highlighted job, and a footer keybar. Actions run on
the selected job in background workers — tailor / cover-letter (LLM, account-safe),
search / apply (account-touching, behind explicit confirms), and résumé setup. Launching,
navigating, and filtering touch only local state.
"""

from __future__ import annotations

from collections.abc import Awaitable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import DataTable, Footer, Input, Static

from job_applicator.exceptions import JobApplicatorError
from job_applicator.models import ApplicationStatus, FunnelStatus

if TYPE_CHECKING:
    from job_applicator.config import AppSettings
    from job_applicator.jobs_store import JobStore
    from job_applicator.models import JobListing, StoredJob
    from job_applicator.scrapers.base import SearchParams
    from job_applicator.state import ApplicationState

T = TypeVar("T")

# Stage → display glyph/colour for the sidebar + detail.
_STAGE_STYLE: dict[str, str] = {
    "found": "white",
    "matched": "cyan",
    "tailored": "yellow",
    "cover_letter": "magenta",
    "applied": "green",
}


class JobApplicatorApp(App[None]):
    """Navigable home screen over the funnel store — browse and act on jobs in-app."""

    TITLE = "job-applicator"

    CSS = """
    #statusline { height: 4; border: round $primary; padding: 0 1; }
    #body { height: 1fr; }
    #joblist { width: 45%; border-right: solid $panel; }
    #detail { width: 1fr; padding: 0 1; }
    #filter { dock: bottom; display: none; border: round $accent; }
    #filter.visible { display: block; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("t", "tailor", "Tailor"),
        Binding("c", "cover_letter", "Cover letter"),
        Binding("s", "search", "Search"),
        Binding("a", "apply", "Apply"),
        Binding("e", "set_resume", "Résumé"),
        Binding("o", "open_url", "Open"),
        Binding("y", "copy_url", "Copy URL", show=False),
        Binding("slash", "filter", "Filter"),
        Binding("escape", "clear_filter", "Clear filter", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(
        self,
        *,
        settings: AppSettings,
        store: JobStore,
        app_state: ApplicationState,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._store = store
        self._app_state = app_state
        self._all: list[StoredJob] = []
        self._by_key: dict[str, StoredJob] = {}
        self._current: StoredJob | None = None  # job shown in the detail pane
        self._applied_urls: set[str] = set()  # URLs with a SUBMITTED record (applied)
        self._busy = ""  # transient "⏳ working…" status while a worker runs
        self._filter = ""
        self._load_error = ""

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Static(id="statusline")
        with Horizontal(id="body"):
            yield DataTable(id="joblist", cursor_type="row", zebra_stripes=True)
            yield VerticalScroll(Static(id="detail"))
        yield Input(id="filter", placeholder="filter title/company — Enter to apply, Esc to clear")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#joblist", DataTable)
        table.add_columns("Stage", "Score", "Title", "Company")
        self._reload()
        table.focus()  # the job list owns focus, not the (hidden) filter Input

    # -------------------------------------------------------------------- data
    def _reload(self) -> None:
        """(Re)load jobs from the store and repaint. Account-safe; errors are shown,
        not raised, so a DB hiccup never crashes the UI."""
        self._load_error = ""
        try:
            self._all = self._store.list_jobs(limit=500)
            self._applied_urls = {
                str(a.job.url)
                for a in self._app_state.list_recent(limit=10_000)
                if a.status == ApplicationStatus.SUBMITTED
            }
        except JobApplicatorError as exc:
            self._all, self._applied_urls = [], set()
            self._load_error = str(exc)
        self._repaint()

    def _visible(self) -> list[StoredJob]:
        if not self._filter:
            return self._all
        needle = self._filter.lower()
        return [
            s for s in self._all if needle in s.job.title.lower() or needle in s.job.company.lower()
        ]

    def _effective_stage(self, s: StoredJob) -> str:
        """The job's stage, overridden to 'applied' when ApplicationState has a SUBMITTED
        record for it — so the sidebar, the counts, and the CLI `status` agree (a job
        applied via the funnel stays in the store at its head stage)."""
        return "applied" if str(s.job.url) in self._applied_urls else s.funnel_status.value

    def _repaint(self) -> None:
        self.query_one("#statusline", Static).update(self._statusline())
        table = self.query_one("#joblist", DataTable)
        table.clear()
        self._by_key = {}
        visible = self._visible()
        for s in visible:
            key = str(s.id)
            self._by_key[key] = s
            stage = self._effective_stage(s)
            style = _STAGE_STYLE.get(stage, "white")
            score = f"{s.match_score:.0%}" if s.match_score is not None else "—"
            table.add_row(
                f"[{style}]{stage.replace('_', ' ')}[/{style}]",
                score,
                escape(s.job.title),
                escape(s.job.company),
                key=key,
            )
        self._update_detail(visible[0] if visible else None)

    # ------------------------------------------------------------------ render
    def _statusline(self) -> str:
        if self._load_error:
            return f"[red]⚠ {escape(self._load_error)}[/red]"
        # Pre-styled sentinel when unset (the default first-run state); escape() only the
        # real path, never the sentinel's own markup.
        path = self._settings.resume_path
        resume = f"[cyan]{escape(path)}[/cyan]" if path else "[dim]not set — press 'e' to set[/dim]"
        if self._busy:  # a worker is running — show live progress instead of static counts
            return f"Résumé {resume}\n[yellow]⏳ {escape(self._busy)}[/yellow]"
        # Compose by URL (furthest-stage-wins) so an applied job — which stays in the store
        # at its head stage — is counted once as applied, matching the CLI `status`.
        counts: dict[str, int] = {}
        for s in self._all:
            stage = self._effective_stage(s)
            counts[stage] = counts.get(stage, 0) + 1
        head_urls = {str(s.job.url) for s in self._all}
        counts["applied"] = counts.get("applied", 0) + len(self._applied_urls - head_urls)
        parts = [f"{counts.get(st.value, 0)} {st.value.replace('_', ' ')}" for st in FunnelStatus]
        shown = len(self._visible())
        filt = (
            f"   [dim](filter: {escape(self._filter)} — {shown} shown)[/dim]"
            if self._filter
            else ""
        )
        return f"Résumé {resume}\n{' · '.join(parts)}{filt}"

    def _set_busy(self, msg: str) -> None:
        """Show a live '⏳ <msg>' progress line while a worker runs (empty string clears
        it, restoring the funnel counts) — so latency never looks like a freeze."""
        self._busy = msg
        self.query_one("#statusline", Static).update(self._statusline())

    def _update_detail(self, job: StoredJob | None) -> None:
        self._current = job
        self.query_one("#detail", Static).update(self._detail_markup(job))

    def _detail_markup(self, s: StoredJob | None) -> str:
        if s is None:
            if self._all:
                return "[dim]No jobs match the filter.[/dim]"
            return (
                "[dim]No jobs yet.\n\nRun [/dim][cyan]job-applicator search -q '…'[/cyan]"
                "[dim] to discover jobs, then [/dim][cyan]match[/cyan][dim] to score them.[/dim]"
            )
        j = s.job
        stage = self._effective_stage(s)
        style = _STAGE_STYLE.get(stage, "white")
        lines = [
            f"[bold]{escape(j.title)}[/bold]",
            f"[green]{escape(j.company)}[/green]   [dim]{j.board.value}[/dim]",
            "",
            f"Stage     [{style}]{stage.replace('_', ' ')}[/{style}]",
            f"Location  {escape(j.location) or '—'}",
            f"Salary    {escape(j.salary) if j.salary else '—'}",
        ]
        if s.match_score is not None:
            sem, skill = s.semantic_score or 0, s.skill_score or 0
            lines.append(
                f"Match     {s.match_score:.0%}  "
                f"[dim](semantic {sem:.0%} · skill {skill:.0%})[/dim]"
            )
        if s.matched_skills:
            lines.append(f"Skills ✓  {escape(', '.join(s.matched_skills[:8]))}")
        if s.missing_skills:
            lines.append(f"Skills ✗  [red]{escape(', '.join(s.missing_skills[:8]))}[/red]")
        if s.tailored_resume_path:
            lines.append(f"Résumé    [dim]{escape(Path(s.tailored_resume_path).name)}[/dim]")
        if s.cover_letter_path:
            lines.append(f"Cover     [dim]{escape(Path(s.cover_letter_path).name)}[/dim]")
        url = str(j.url)
        if url and "example.com/placeholder" not in url:  # hide the manual-tailor placeholder
            # Click the link (mouse) to open it; `o` opens, `y` copies — the TUI captures
            # the mouse, so plain terminal select/copy doesn't work.
            lines += [
                "",
                f"[@click=app.open_url][blue underline]{escape(url)}[/blue underline][/]"
                "  [dim](o open · y copy)[/dim]",
            ]
        if j.description:
            desc = j.description[:600] + ("…" if len(j.description) > 600 else "")
            lines += ["", "[bold]Description[/bold]", escape(desc)]
        return "\n".join(lines)

    # ----------------------------------------------------------------- actions
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key.value
        if key is not None:
            self._update_detail(self._by_key.get(key))

    def action_refresh(self) -> None:
        self._reload()

    def _current_url(self) -> str | None:
        """The selected job's posting URL, or None (with a toast) when there isn't one."""
        if self._current is None:
            self.notify("No job selected.", severity="warning")
            return None
        url = str(self._current.job.url)
        if not url or "example.com/placeholder" in url:
            self.notify("This job has no URL.", severity="warning")
            return None
        return url

    def action_open_url(self) -> None:
        """Open the selected job's posting in the default browser, OFF the UI thread. Uses
        YOUR browser (not the tool's automation session) — a normal human click, no
        anti-bot risk."""
        import os
        import sys

        url = self._current_url()
        if url is None:
            return
        # On headless Linux the default "browser" may be a TERMINAL browser (w3m/lynx) that
        # would block the loop and draw over the TUI — don't risk it; point at copy instead.
        if sys.platform == "linux" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self.notify(
                f"No graphical browser here — press 'y' to copy:  {url}",
                severity="warning",
                timeout=8,
            )
            return
        self._open_url_worker(url)

    @work(thread=True)
    def _open_url_worker(self, url: str) -> None:
        import webbrowser

        try:
            ok = webbrowser.open(url)
        except Exception:  # any browser-launch failure → fall back to the copy hint
            ok = False
        if ok:
            self.call_from_thread(self.notify, f"Opened in browser:  {url}", timeout=4)
        else:
            self.call_from_thread(
                self.notify,
                f"No browser available — press 'y' to copy:  {url}",
                severity="warning",
                timeout=8,
            )

    def action_copy_url(self) -> None:
        """Copy the selected job's URL to the clipboard (OSC 52; the TUI captures the mouse,
        so plain terminal selection can't)."""
        url = self._current_url()
        if url is None:
            return
        self.copy_to_clipboard(url)
        self.notify(f"URL copied (OSC 52):  {url}", timeout=5)

    def action_set_resume(self) -> None:
        """Set the résumé path in-app (no TOML editing) — opens a modal, saves to config."""
        from job_applicator.tui.screens import SetupScreen

        self.push_screen(SetupScreen(self._settings.resume_path), self._set_resume_then)

    def _set_resume_then(self, path: str | None) -> None:
        if not path:  # cancelled
            return
        self._settings.resume_path = path  # take effect immediately (this session)
        saved = self._persist_resume_path(path)
        self._reload()  # repaint the status line
        if saved is not None:
            self.notify(f"Résumé set ✓ — saved to {saved}", timeout=6)
        else:
            self.notify(
                "Résumé set for this session (couldn't write config — set resume_path "
                "manually to persist).",
                severity="warning",
                timeout=8,
            )

    def _persist_resume_path(self, path: str) -> Path | None:
        """Best-effort: write a top-level ``resume_path`` into the config file — create it
        if missing, else replace an ACTIVE top-level line, else prepend.

        Safe over a credentialed config: the result is re-parsed and must yield exactly
        this top-level ``resume_path`` (so a line accidentally rewritten inside a ``[table]``
        or a duplicate key that won't parse is rejected), and the swap is ATOMIC
        (temp file + ``os.replace``) so a torn write can never destroy the config. Returns
        the file written, or None on any failure (caller falls back to a session-only set).
        """
        import json
        import os
        import re
        import tomllib

        from job_applicator.config import CONFIG_FILE_ENV_VAR, DEFAULT_CONFIG_FILE

        cfg = Path(os.environ.get(CONFIG_FILE_ENV_VAR, DEFAULT_CONFIG_FILE))
        line = f"resume_path = {json.dumps(path)}"
        try:
            if cfg.exists():
                text = cfg.read_text(encoding="utf-8")
                # Match an ACTIVE line only (no leading '#', so we never un-comment an
                # example into a duplicate key); else prepend at the very top (top-level).
                pattern = re.compile(r"(?m)^[ \t]*resume_path[ \t]*=.*$")
                new_text = (
                    pattern.sub(line, text, count=1) if pattern.search(text) else f"{line}\n{text}"
                )
            else:
                new_text = f"# job-applicator config\n{line}\n"
            # Validate before committing: must parse AND set resume_path at top level.
            if tomllib.loads(new_text).get("resume_path") != path:
                return None
            tmp = cfg.with_name(f"{cfg.name}.tmp")
            tmp.write_text(new_text, encoding="utf-8")
            os.replace(tmp, cfg)  # atomic swap
        except (OSError, ValueError):  # ValueError covers tomllib.TOMLDecodeError
            return None
        return cfg

    def _selected_job_with_resume(self) -> JobListing | None:
        """The selected job's listing, or None (with a toast) when there's no selection
        or no résumé configured — the shared guard for the LLM actions."""
        if self._current is None:
            self.notify("No job selected.", severity="warning")
            return None
        if not self._settings.resume_path:
            self.notify("Set a résumé first — press 'e'.", severity="warning")
            return None
        return self._current.job

    def action_tailor(self) -> None:
        """Tailor the selected job's résumé in a background worker. Account-safe."""
        job = self._selected_job_with_resume()
        if job is not None:
            self._tailor_worker(job)

    @work(exclusive=True, group="action")
    async def _tailor_worker(self, job: JobListing) -> None:
        from job_applicator.tui import actions

        self._set_busy(f"Tailoring {job.title}…")
        try:
            tailored = await self._run_action("Tailor", actions.tailor_job(self._settings, job))
        finally:
            self._set_busy("")
        if tailored is None:
            return
        self._store.mark_tailored(job, tailored_resume_path=tailored.output_path)
        self._reload()
        self.notify(f"Tailored ✓  →  {tailored.output_path}", timeout=6)

    def action_cover_letter(self) -> None:
        """Write a cover letter for the selected job in a background worker. Account-safe."""
        job = self._selected_job_with_resume()
        if job is not None:
            self._cover_letter_worker(job)

    @work(exclusive=True, group="action")
    async def _cover_letter_worker(self, job: JobListing) -> None:
        from job_applicator.tui import actions

        self._set_busy(f"Writing a cover letter for {job.title}…")
        try:
            result = await self._run_action(
                "Cover letter", actions.cover_letter_job(self._settings, job)
            )
        finally:
            self._set_busy("")
        if result is None:
            return
        self._store.set_cover_letter(str(job.url), result.output_path)
        self._reload()
        self.notify(f"Cover letter ✓  →  {result.output_path}", timeout=6)

    def action_search(self) -> None:
        """Open the search modal. Touches the account only on submit — the modal collects
        the query and shows the 'opens a browser on your real account' warning."""
        if self._account_busy():
            self.notify(
                "An account action is already running — wait for it to finish.",
                severity="warning",
            )
            return
        from job_applicator.tui.screens import SearchScreen

        self.push_screen(SearchScreen(), self._search_then)

    def _search_then(self, params: SearchParams | None) -> None:
        if params is not None:  # None = cancelled; nothing touched the account
            self._search_worker(params)

    @work(group="account")
    async def _search_worker(self, params: SearchParams) -> None:
        from job_applicator.tui import actions

        table = self.query_one("#joblist", DataTable)
        table.loading = True  # built-in spinner on the list while the scrape/score runs
        self._set_busy("Searching…")
        try:
            found = await self._run_action(
                "Search",
                actions.search_jobs(
                    self._settings, self._store, params, on_progress=self._set_busy
                ),
            )
        finally:
            table.loading = False
            self._set_busy("")
        if found is None:
            return
        self._reload()
        self.notify(f"Found {found} job(s) — added to your funnel.", timeout=6)

    def action_apply(self) -> None:
        """Open the apply modal for the selected job. Dry-run by default; a real submit is
        gated behind the modal's danger checkbox. Account-touching only on confirm."""
        if self._current is None:
            self.notify("No job selected.", severity="warning")
            return
        from job_applicator.models import JobBoard

        if self._current.job.board is not JobBoard.LINKEDIN:
            self.notify(
                f"Automated apply is LinkedIn-only — apply to "
                f"{self._current.job.board.value} jobs manually.",
                severity="warning",
                timeout=8,
            )
            return
        if self._account_busy():
            self.notify(
                "An account action is already running — wait for it to finish.",
                severity="warning",
            )
            return
        from job_applicator.tui.screens import ApplyScreen

        job = self._current.job
        self.push_screen(ApplyScreen(job), partial(self._apply_dispatch, job))

    def _apply_dispatch(self, job: JobListing, submit: bool | None) -> None:
        if submit is not None:  # None = cancelled; nothing touched the account
            self._apply_worker(job, submit=submit)

    @work(group="account")
    async def _apply_worker(self, job: JobListing, *, submit: bool) -> None:
        from job_applicator.tui import actions

        mode = "Submitting a real application to" if submit else "Dry-run for"
        self._set_busy(f"{mode} {job.title} — a browser will open…")
        try:
            result = await self._run_action(
                "Apply", actions.apply_job(self._settings, job, submit=submit)
            )
        finally:
            self._set_busy("")
        if result is None:
            return
        self._reload()
        suffix = "" if submit else "  (dry run — nothing submitted)"
        self.notify(f"{job.title}: {result.status.value}{suffix}", timeout=8)

    def _account_busy(self) -> bool:
        """True while an account-touching worker (search/apply) is running. Account actions
        run one-at-a-time and uninterrupted — so a real submission always completes and is
        recorded, never cancelled mid-flight by a second action (which would risk a
        duplicate)."""
        from textual.worker import WorkerState

        active = (WorkerState.PENDING, WorkerState.RUNNING)
        return any(w.group == "account" and w.state in active for w in self.workers)

    async def _run_action(self, label: str, coro: Awaitable[T]) -> T | None:
        """Await an action coroutine, turning ANY failure into a toast — a worker bug must
        never tear down the whole app. Returns the result, or None on failure."""
        try:
            return await coro
        except JobApplicatorError as exc:
            self.notify(f"{label} failed: {exc}", severity="error", timeout=8)
        except Exception as exc:  # surface any worker bug as a toast, keep the app alive
            self.log.error(f"{label} worker error: {exc!r}")
            self.notify(f"{label} error: {exc}", severity="error", timeout=10)
        return None

    def action_cursor_down(self) -> None:
        self.query_one("#joblist", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#joblist", DataTable).action_cursor_up()

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.value = ""
        box.add_class("visible")
        # Focus on the NEXT frame, not synchronously: if we focused now, the "/" that
        # triggered this action would land in the freshly-focused Input. Deferring means
        # the Input isn't focused while "/" is in flight, so it starts genuinely empty.
        self.call_after_refresh(box.focus)

    def action_clear_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.value = ""
        box.remove_class("visible")
        self._filter = ""
        self._repaint()
        self.query_one("#joblist", DataTable).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._filter = event.value.strip()
        event.input.remove_class("visible")
        self._repaint()
        self.query_one("#joblist", DataTable).focus()


def run_tui(settings: AppSettings) -> None:
    """Build the local store(s) and run the TUI. Offline + account-safe."""
    from job_applicator.jobs_store import JobStore
    from job_applicator.state import ApplicationState

    JobApplicatorApp(settings=settings, store=JobStore(), app_state=ApplicationState()).run()
