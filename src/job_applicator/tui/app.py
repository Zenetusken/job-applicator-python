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
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Input, LoadingIndicator, Static

from job_applicator.exceptions import JobApplicatorError
from job_applicator.models import ApplicationStatus, FunnelStatus, JobBoard

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

# Stage-filter cycle: None (all) then each funnel stage in funnel order. `_effective_stage`
# returns exactly these values (incl. the ApplicationState "applied" overlay), so the filter
# compares cleanly against it. `_STAGE_ORDER` ranks a stage for the "stage" sort.
_STAGE_CYCLE: list[str | None] = [None, *(st.value for st in FunnelStatus)]
_STAGE_ORDER: dict[str, int] = {st.value: i for i, st in enumerate(FunnelStatus)}

# Board-filter cycle: None (all) then each board; compared against `s.job.board.value`.
# `_BOARD_STYLE` colours the list's Board column for a quick LinkedIn-vs-Indeed scan.
_BOARD_CYCLE: list[str | None] = [None, *(b.value for b in JobBoard)]
_BOARD_STYLE: dict[str, str] = {"linkedin": "blue", "indeed": "yellow"}

# Sort orders cycled by `S`; "match" (best opportunity first) is the default. The labels are
# what the status line shows.
_SORT_CYCLE: list[str] = ["match", "recent", "stage"]
_SORT_LABEL: dict[str, str] = {
    "match": "best match",
    "recent": "recent",
    "stage": "funnel stage",
}


def _elide(text: str, limit: int) -> str:
    """Truncate to ``limit`` columns with a trailing ellipsis (display only — elide the raw
    value BEFORE escaping, since escape() changes length)."""
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _elide_mid(text: str, limit: int) -> str:
    """Elide the MIDDLE, keeping head + tail — for artifact filenames whose identity lives in
    both the prefix (kind/company) and the suffix (timestamp)."""
    if len(text) <= limit:
        return text
    keep = max(2, limit - 1)
    head = (keep + 1) // 2
    return text[:head] + "…" + text[-(keep - head) :]


class _JobListLoading(LoadingIndicator):
    """Loading state shown over the job list while a search runs. A ``LoadingIndicator``
    subclass (a self-rendering leaf — what Textual's loading cover expects; a container with
    composed children collapses in the cover) given a SOLID app-surface background so the
    cover shows the theme instead of bare terminal grey. The animation fills + centres itself;
    the descriptive phase text stays in the status header (``_set_busy``)."""

    DEFAULT_CSS = """
    _JobListLoading { color: $accent; }
    /* The cover gets the `-textual-loading-indicator` class, whose base rule sets a
       translucent `$boost` background (the bare-grey bleed). Match that selector's
       specificity (type + class) to override it with a SOLID surface. */
    _JobListLoading.-textual-loading-indicator { background: $surface; }
    """


class JobListTable(DataTable[str]):
    """The job-list table. Overrides the loading widget so a running search shows a themed
    indicator over the app surface instead of the framework default's bare grey overlay."""

    def get_loading_widget(self) -> Widget:
        return _JobListLoading()


class JobApplicatorApp(App[None]):
    """Navigable home screen over the funnel store — browse and act on jobs in-app."""

    TITLE = "job-applicator"

    CSS = """
    #statusline { height: 4; border: round $primary; padding: 0 1; }
    #body { height: 1fr; }
    /* height: 1fr so the table fills the body like the detail pane — a DataTable defaults
       to height:auto (sizes to its rows), which left the left side short of the bottom with
       few results AND made the loading cover only span the content rows. */
    #joblist { width: 45%; height: 1fr; border-right: solid $panel; }
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
        Binding("A", "ats_check", "ATS"),
        Binding("o", "open_url", "Open"),
        Binding("y", "copy_url", "Copy URL", show=False),
        Binding("slash", "filter", "Filter"),
        Binding("f", "cycle_stage_filter", "Stage"),
        Binding("b", "cycle_board_filter", "Board"),
        Binding("S", "cycle_sort", "Sort"),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "clear_filter", "Clear filter", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        # Scroll the detail pane (the job list owns focus / arrows, so a long posting needs
        # its own keys for keyboard users; the mouse wheel works over it too).
        Binding("right_square_bracket", "scroll_detail(1)", "Scroll posting", show=False),
        Binding("left_square_bracket", "scroll_detail(-1)", "Scroll posting up", show=False),
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
        self._filter = ""  # text filter (title/company substring)
        self._stage_filter: str | None = None  # funnel-stage filter (None = all stages)
        self._board_filter: str | None = None  # board filter (None = all boards)
        self._sort_mode = "match"  # list sort order (see _SORT_CYCLE)
        self._load_error = ""

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Static(id="statusline")
        with Horizontal(id="body"):
            yield JobListTable(id="joblist", cursor_type="row", zebra_stripes=True)
            yield VerticalScroll(Static(id="detail"), id="detailscroll")
        yield Input(id="filter", placeholder="filter title/company — Enter to apply, Esc to clear")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#joblist", DataTable)
        table.add_columns("Stage", "Score", "Board", "Title", "Company")
        self._reload()
        table.focus()  # the job list owns focus, not the (hidden) filter Input

    # -------------------------------------------------------------------- data
    def _reload(self, *, refresh_applied: bool = True) -> None:
        """(Re)load jobs from the store and repaint. Account-safe; errors are shown,
        not raised, so a DB hiccup never crashes the UI.

        ``refresh_applied=False`` skips the applied-URLs requery: during a search (which
        writes only the jobs table) the SUBMITTED set is invariant, so the per-streamed-row
        repaint reuses it instead of re-reading list_recent N times on the event loop.
        """
        self._load_error = ""
        try:
            self._all = self._store.list_jobs(limit=500)
            if refresh_applied:
                self._applied_urls = {
                    str(a.job.url)
                    for a in self._app_state.list_recent(limit=10_000)
                    if a.status == ApplicationStatus.SUBMITTED
                }
        except JobApplicatorError as exc:
            self._all = []
            self._load_error = str(exc)
            if refresh_applied:
                self._applied_urls = set()
        # Sort the VIEW (not the shared list_jobs query the CLI also uses) by the active
        # mode — keeps it TUI-local, and makes the post-scoring repaint visibly re-rank.
        self._apply_sort()
        self._repaint()

    def _apply_sort(self) -> None:
        """Sort ``self._all`` in place by the active sort mode. Runs on the freshly
        store-ordered list (``_reload`` re-queries updated_at-desc first), so "match" keeps
        unscored jobs newest-first via the stable sort."""
        if self._sort_mode == "recent":
            self._all.sort(key=lambda s: s.updated_at, reverse=True)
        elif self._sort_mode == "stage":
            # Funnel order (found→applied, mirroring the status-line counts), best score
            # first within a stage (negate for descending score under the ascending key).
            self._all.sort(
                key=lambda s: (
                    _STAGE_ORDER.get(self._effective_stage(s), 0),
                    -(s.match_score or 0.0),
                )
            )
        else:  # "match" — best opportunity first: scored desc, then unscored (newest-first).
            self._all.sort(
                key=lambda s: (s.match_score is not None, s.match_score or 0.0), reverse=True
            )

    def _visible(self) -> list[StoredJob]:
        """The rows to show: ``self._all`` narrowed by the stage filter then the text filter
        (both optional, composable). Sort order is already applied to ``self._all``."""
        jobs = self._all
        if self._stage_filter is not None:
            jobs = [s for s in jobs if self._effective_stage(s) == self._stage_filter]
        if self._board_filter is not None:
            jobs = [s for s in jobs if s.job.board.value == self._board_filter]
        if self._filter:
            needle = self._filter.lower()
            jobs = [
                s for s in jobs if needle in s.job.title.lower() or needle in s.job.company.lower()
            ]
        return jobs

    def _effective_stage(self, s: StoredJob) -> str:
        """The job's stage, overridden to 'applied' when ApplicationState has a SUBMITTED
        record for it — so the sidebar, the counts, and the CLI `status` agree (a job
        applied via the funnel stays in the store at its head stage)."""
        return "applied" if str(s.job.url) in self._applied_urls else s.funnel_status.value

    def _repaint(self) -> None:
        self.query_one("#statusline", Static).update(self._statusline())
        table = self.query_one("#joblist", DataTable)
        # Cap Title (and Company) so the auto-sized Title column can't expand to the longest
        # title and push Company off the right edge (the reported bug). Fixed caps are
        # deliberate over a pane-width-derived value: the latter reads table.size mid-layout
        # and proved racy. Lowered from 46/22 to make room for the Board column: 38+18 keeps
        # the whole table ≤ ~84 cols so every column (incl. Company) still fits the pane at a
        # ~190-wide terminal (measured); narrower terminals h-scroll as before. Cells
        # ellipsize cleanly instead of hard-cutting at the edge.
        title_cap, company_cap = 38, 18
        table.clear()
        self._by_key = {}
        visible = self._visible()
        for s in visible:
            key = str(s.id)
            self._by_key[key] = s
            stage = self._effective_stage(s)
            style = _STAGE_STYLE.get(stage, "white")
            score = f"{s.match_score:.0%}" if s.match_score is not None else "—"
            board = s.job.board
            board_cell = f"[{_BOARD_STYLE.get(board.value, 'white')}]{board.display_name}[/]"
            table.add_row(
                f"[{style}]{stage.replace('_', ' ')}[/{style}]",
                score,
                board_cell,
                escape(_elide(s.job.title, title_cap)),
                escape(_elide(s.job.company, company_cap)),
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
        # View state: the active sort (always — a control the user drives with `S`), plus the
        # stage/text filters when set, and the shown count when either narrows the list.
        view = [f"sort: {_SORT_LABEL[self._sort_mode]}"]
        if self._stage_filter is not None:
            view.append(f"stage: {self._stage_filter.replace('_', ' ')}")
        if self._board_filter is not None:
            view.append(f"board: {JobBoard(self._board_filter).display_name}")
        if self._filter:
            view.append(f"text: {escape(self._filter)}")
        narrowed = (
            self._stage_filter is not None or self._board_filter is not None or bool(self._filter)
        )
        tail = f" — {len(self._visible())} shown" if narrowed else ""
        suffix = f"   [dim]({' · '.join(view)}{tail})[/dim]"
        return f"Résumé {resume}\n{' · '.join(parts)}{suffix}"

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
                return (
                    "[dim]No jobs match the current filter.\n\nPress [/dim][cyan]f[/cyan]"
                    "[dim] stage · [/dim][cyan]b[/cyan][dim] board · [/dim][cyan]/[/cyan]"
                    "[dim] text · [/dim][cyan]Esc[/cyan][dim] to clear all.[/dim]"
                )
            return (
                "[dim]No jobs yet.\n\nPress [/dim][cyan]s[/cyan][dim] to search a board · "
                "[/dim][cyan]e[/cyan][dim] to set your résumé · [/dim][cyan]?[/cyan]"
                "[dim] for all keys.[/dim]"
            )
        j = s.job
        stage = self._effective_stage(s)
        style = _STAGE_STYLE.get(stage, "white")
        lines = [
            f"[bold]{escape(j.title)}[/bold]",
            f"[green]{escape(j.company)}[/green]   [dim]{j.board.display_name}[/dim]",
            "",
            f"Stage     [{style}]{stage.replace('_', ' ')}[/{style}]",
            f"Location  {escape(j.location) if j.location else '—'}",
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
            name = escape(_elide_mid(Path(s.tailored_resume_path).name, 40))
            lines.append(
                f"Résumé    [@click=app.open_tailored][dim]{name}[/dim][/]"
                "  [dim](click to open)[/dim]"
            )
        if s.cover_letter_path:
            name = escape(_elide_mid(Path(s.cover_letter_path).name, 40))
            lines.append(
                f"Cover     [@click=app.open_cover][dim]{name}[/dim][/]  [dim](click to open)[/dim]"
            )
        url = str(j.url)
        if url and "example.com/placeholder" not in url:  # hide the manual-tailor placeholder
            # Show a compact form of the (often tracking-heavy) URL; o/click/y still act on
            # the FULL stored URL (they read j.url, never this display text). The TUI captures
            # the mouse, so plain terminal select/copy doesn't work — hence o/y.
            shown = escape(_elide(url, 64))
            lines += [
                "",
                f"[@click=app.open_url][blue underline]{shown}[/blue underline][/]"
                "  [dim](o open · y copy)[/dim]",
            ]
        if j.description:
            # Full description (the store keeps up to ~5k chars); the detail pane scrolls.
            # Truncating here hid the rest of the posting even though the pane can scroll.
            lines += ["", "[bold]Description[/bold]", escape(j.description)]
        return "\n".join(lines)

    # ----------------------------------------------------------------- actions
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key.value
        if key is not None:
            self._update_detail(self._by_key.get(key))

    def action_refresh(self) -> None:
        self._reload()

    def action_cycle_stage_filter(self) -> None:
        """Cycle the funnel-stage filter (all → each stage → all). View-only; composes with
        the board + text filters and respects the 'applied' overlay via ``_effective_stage``."""
        i = _STAGE_CYCLE.index(self._stage_filter)
        self._stage_filter = _STAGE_CYCLE[(i + 1) % len(_STAGE_CYCLE)]
        self._repaint()

    def action_cycle_board_filter(self) -> None:
        """Cycle the board filter (all → each board → all). View-only; composes with the
        stage + text filters."""
        i = _BOARD_CYCLE.index(self._board_filter)
        self._board_filter = _BOARD_CYCLE[(i + 1) % len(_BOARD_CYCLE)]
        self._repaint()

    def action_cycle_sort(self) -> None:
        """Cycle the list sort order (best match → recent → funnel stage). View-only."""
        i = _SORT_CYCLE.index(self._sort_mode)
        self._sort_mode = _SORT_CYCLE[(i + 1) % len(_SORT_CYCLE)]
        # Re-query store-ordered, then re-sort, so "match" keeps its newest-first tiebreak;
        # a keypress can't change the applied set, so skip its requery.
        self._reload(refresh_applied=False)

    def action_help(self) -> None:
        """Show the in-app key reference (read-only modal; touches nothing)."""
        from job_applicator.tui.screens import HelpScreen

        self.push_screen(HelpScreen())

    def action_scroll_detail(self, direction: int) -> None:
        """Page the detail/posting pane (the job list owns focus + the arrow keys, so a long
        posting needs its own keys for keyboard users)."""
        pane = self.query_one("#detailscroll", VerticalScroll)
        if direction > 0:
            pane.scroll_page_down()
        else:
            pane.scroll_page_up()

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

    @staticmethod
    def _headless() -> bool:
        """True on a Linux box with no graphical display, where opening a GUI app would
        block the loop / draw over the TUI (the default "browser" may be a terminal one
        like w3m). Callers point at a fallback instead of risking it."""
        import os
        import sys

        return sys.platform == "linux" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )

    def action_open_url(self) -> None:
        """Open the selected job's posting in the default browser, OFF the UI thread. Uses
        YOUR browser (not the tool's automation session) — a normal human click, no
        anti-bot risk."""
        url = self._current_url()
        if url is None:
            return
        if self._headless():
            self.notify(
                f"No graphical browser here — press 'y' to copy:  {escape(url)}",
                severity="warning",
                timeout=8,
            )
            return
        # escape() the (scraped) URL: notify renders Rich markup, so a bracketed URL
        # (IPv6 host, or a ?filter[0]= query param) would otherwise drop that span.
        safe = escape(url)
        self._open_external_worker(
            url,
            opened_msg=f"Opened in browser:  {safe}",
            fail_msg=f"No browser available — press 'y' to copy:  {safe}",
        )

    def action_open_tailored(self) -> None:
        """Open the tailored-résumé artifact for the selected job (if one was generated)."""
        s = self._current
        self._open_artifact(s.tailored_resume_path if s else None, "tailored résumé")

    def action_open_cover(self) -> None:
        """Open the cover-letter artifact for the selected job (if one was generated)."""
        s = self._current
        self._open_artifact(s.cover_letter_path if s else None, "cover letter")

    def _open_artifact(self, path: str | None, label: str) -> None:
        """Open a generated artifact file in YOUR default viewer (off-thread), with coherent
        toasts. Account-safe — a local file in your own viewer, nothing touches an account.
        Opens via a ``file://`` URI through the same worker as URLs (a .txt opens as text)."""
        if not path:
            self.notify(f"No {label} generated yet for this job.", severity="warning")
            return
        p = Path(path)
        if not p.exists():
            self.notify(
                f"{label.capitalize()} file is missing: {escape(p.name)}", severity="warning"
            )
            return
        if self._headless():
            self.notify(
                f"No graphical viewer here — {label} at {escape(str(p))}",
                severity="warning",
                timeout=8,
            )
            return
        self._open_external_worker(
            p.resolve().as_uri(),
            opened_msg=f"Opened {label}:  {escape(p.name)}",
            fail_msg=f"No viewer available — {label} at {escape(str(p))}",
        )

    @work(thread=True)
    def _open_external_worker(self, target: str, *, opened_msg: str, fail_msg: str) -> None:
        """Open ``target`` (a posting URL or a ``file://`` artifact URI) in the user's
        default app, OFF the UI thread (a cold launch can block). ``opened_msg`` / ``fail_msg``
        are pre-formatted by the caller so the toast reads coherently for either kind."""
        import webbrowser

        try:
            ok = webbrowser.open(target)
        except Exception:  # any launch failure → the caller's fallback hint
            ok = False
        if ok:
            self.call_from_thread(self.notify, opened_msg, timeout=4)
        else:
            self.call_from_thread(self.notify, fail_msg, severity="warning", timeout=8)

    def action_copy_url(self) -> None:
        """Copy the selected job's URL to the clipboard (OSC 52; the TUI captures the mouse,
        so plain terminal selection can't)."""
        url = self._current_url()
        if url is None:
            return
        self.copy_to_clipboard(url)  # the clipboard gets the raw URL; only the toast escapes
        self.notify(f"URL copied (OSC 52):  {escape(url)}", timeout=5)

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
        """Write a cover letter for the selected job in a background worker. Account-safe.
        If the job was already tailored, the letter draws on the tailored résumé."""
        job = self._selected_job_with_resume()
        if job is not None:
            tailored = self._current.tailored_resume_path if self._current else ""
            self._cover_letter_worker(job, tailored)

    @work(exclusive=True, group="action")
    async def _cover_letter_worker(self, job: JobListing, tailored_resume_path: str) -> None:
        from job_applicator.tui import actions

        self._set_busy(f"Writing a cover letter for {job.title}…")
        try:
            result = await self._run_action(
                "Cover letter",
                actions.cover_letter_job(
                    self._settings, job, tailored_resume_path=tailored_resume_path
                ),
            )
        finally:
            self._set_busy("")
        if result is None:
            return
        self._store.set_cover_letter(str(job.url), result.output_path)
        self._reload()
        self.notify(f"Cover letter ✓  →  {result.output_path}", timeout=6)

    def action_ats_check(self) -> None:
        """Check the selected job's résumé (the tailored one if available, else the
        configured résumé) for ATS compatibility, and show the result. Offline; account-safe."""
        if self._selected_job_with_resume() is None:
            return
        tailored = self._current.tailored_resume_path if self._current else ""
        self._ats_worker(tailored)

    @work(exclusive=True, group="action")
    async def _ats_worker(self, tailored_resume_path: str) -> None:
        from job_applicator.tui import actions

        self._set_busy("Checking ATS compatibility…")
        try:
            result = await self._run_action(
                "ATS check", actions.ats_check(self._settings, tailored_resume_path)
            )
        finally:
            self._set_busy("")
        if result is None:
            return
        from job_applicator.tui.screens import AtsScreen

        source = "tailored résumé" if tailored_resume_path else "configured résumé"
        self.push_screen(AtsScreen(result, source))

    def action_search(self) -> None:
        """Open the search modal. Touches the account only on submit — the modal collects
        the query and shows the 'opens a browser on your real account' warning."""
        if self._account_busy_refused():
            return
        from job_applicator.tui.screens import SearchScreen

        self.push_screen(SearchScreen(), self._search_then)

    def _search_then(self, plans: list[SearchParams] | None) -> None:
        # None/[] = cancelled or no board selected. Re-check busy: a second search modal can
        # stack over the first (the app keybind fires under a modal), and both could dismiss.
        if plans and not self._account_busy_refused():
            self._search_worker(plans)

    @work(group="account")
    async def _search_worker(self, plans: list[SearchParams]) -> None:
        """Scrape each selected board SEQUENTIALLY in this single account worker — never two
        browsers at once, and never interrupted (so a real submission elsewhere can't be
        cancelled mid-flight). One board failing (e.g. Indeed can't clear Cloudflare) is
        toasted by ``_run_action`` and does NOT abort the remaining board(s)."""
        from job_applicator.tui import actions

        table = self.query_one("#joblist", DataTable)
        table.loading = True  # spinner until the FIRST result streams in (across all boards)
        self._set_busy("Searching…")
        streamed = False

        def on_job(_job: JobListing) -> None:
            # Each scraped listing is already persisted (actions.emit) before this fires, so
            # a repaint from the store shows it. Drop the spinner on the first one (any
            # board), then let rows accumulate live. On the event loop → direct UI update.
            nonlocal streamed
            if not streamed:
                streamed = True
                table.loading = False
            # The SUBMITTED set can't change during a search, so skip its requery per row.
            self._reload(refresh_applied=False)

        total = 0
        failed: list[str] = []
        try:
            for params in plans:
                found = await self._run_action(
                    f"Search {params.board.display_name}",
                    actions.search_jobs(
                        self._settings,
                        self._store,
                        params,
                        on_progress=self._set_busy,
                        on_job=on_job,
                    ),
                )
                if found is None:  # this board errored (already toasted); try the next
                    failed.append(params.board.display_name)
                else:
                    total += found
        finally:
            table.loading = False
            self._set_busy("")
        # Final repaint after scoring: the streamed (found) rows now carry scores and
        # re-sort best-match-first (the reorder-on-score).
        self._reload()
        if total == 0 and failed:
            return  # every attempted board errored — _run_action already toasted each
        # Name only the boards that succeeded in the "across" clause; failed ones go in the
        # suffix (so a board isn't listed twice).
        succeeded = ", ".join(
            p.board.display_name for p in plans if p.board.display_name not in failed
        )
        suffix = f"  ({', '.join(failed)} failed)" if failed else ""
        self.notify(
            f"Found {total} job(s) across {succeeded}{suffix} — added to your funnel.",
            severity="warning" if failed else "information",
            timeout=8,
        )

    def action_apply(self) -> None:
        """Open the apply modal for the selected job. Dry-run by default; a real submit is
        gated behind the modal's danger checkbox. Account-touching only on confirm."""
        if self._current is None:
            self.notify("No job selected.", severity="warning")
            return
        if self._current.job.board is not JobBoard.LINKEDIN:
            self.notify(
                f"Automated apply is LinkedIn-only — apply to "
                f"{self._current.job.board.value} jobs manually.",
                severity="warning",
                timeout=8,
            )
            return
        if self._account_busy_refused():
            return
        from job_applicator.tui.screens import ApplyScreen

        job = self._current.job
        self.push_screen(ApplyScreen(job), partial(self._apply_dispatch, job))

    def _apply_dispatch(self, job: JobListing, submit: bool | None) -> None:
        # None = cancelled. Re-check busy at dispatch too (a stacked confirm could otherwise
        # start a second account worker past the modal-open gate).
        if submit is None or self._account_busy_refused():
            return
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

    def _account_busy_refused(self) -> bool:
        """True (after a toast) when an account action is already running. Checked at BOTH the
        modal open AND the post-confirm dispatch, so a stacked second modal can't slip a second
        account worker past the one-at-a-time rule (the modal gate alone isn't enough — the
        app keybind still fires while a modal is up if focus is off an Input)."""
        if self._account_busy():
            self.notify(
                "An account action is already running — wait for it to finish.",
                severity="warning",
            )
            return True
        return False

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
        self._stage_filter = None  # Esc resets ALL filters → back to the full list
        self._board_filter = None
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
