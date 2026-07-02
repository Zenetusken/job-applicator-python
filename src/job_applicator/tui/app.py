"""The Textual application: a navigable home over the job-funnel store.

Layout: a status line (résumé + funnel summary), a job-list sidebar (the funnel head from
``JobStore``), a detail pane for the highlighted job, and a footer keybar. Actions run on
the selected job in background workers — tailor / cover-letter (LLM, account-safe),
search / apply (account-touching, behind explicit confirms), and résumé setup. Launching,
navigating, and filtering touch only local state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markup import escape
from rich.measure import Measurement
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Footer, Input, LoadingIndicator, OptionList, Static, Tab, Tabs
from textual.widgets.option_list import Option

from job_applicator.exceptions import JobApplicatorError
from job_applicator.models import (
    ApplicationStatus,
    Format,
    FunnelStatus,
    JobBoard,
    coverage_measured,
    parse_salary_to_annual_min,
)
from job_applicator.tui.textfmt import format_job_description

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

# View tabs across the top mirror the stage filter: "All" + each funnel stage, in funnel order.
# The tab id encodes the stage value ("stage-all" = no filter); the label is the short caption.
_STAGE_TAB_LABEL: dict[str | None, str] = {
    None: "All",
    "found": "Found",
    "matched": "Matched",
    "tailored": "Tailored",
    "cover_letter": "Cover letter",
    "applied": "Applied",
}


def _stage_to_tab(stage: str | None) -> str:
    """Tab id for a stage filter value (None → the All tab)."""
    return f"stage-{stage or 'all'}"


def _tab_to_stage(tab_id: str | None) -> str | None:
    """Stage filter value for a tab id (the All tab → None)."""
    if not tab_id or tab_id == "stage-all":
        return None
    return tab_id.removeprefix("stage-")


# Board-filter cycle: None (all) then each board; compared against `s.job.board.value`.
# `_BOARD_STYLE` colours the full board name as a badge for an instant LinkedIn-vs-Indeed scan.
# (Stage colour now lives on the card's left spine, well away from this mid-line badge, so the
# board can keep its own brand colour without reading as a stage.)
_BOARD_CYCLE: list[str | None] = [None, *(b.value for b in JobBoard)]
_BOARD_STYLE: dict[str, str] = {"linkedin": "bold blue", "indeed": "bold yellow"}


def _score_style(score: float | None) -> str:
    """Sentiment palette for a match score: green (strong) / yellow (moderate) / red (weak),
    so the confidence reads at a glance. Unknown (unscored) → dim."""
    if score is None:
        return "dim"
    if score >= 0.70:
        return "bold green"
    if score >= 0.50:
        return "bold yellow"
    return "bold red"


# Sort orders cycled by `S`; "match" (best opportunity first) is the default. The labels are
# what the status line shows.
_SORT_CYCLE: list[str] = ["match", "recent", "stage", "salary"]
_SORT_LABEL: dict[str, str] = {
    "match": "best match",
    "recent": "recent",
    "stage": "funnel stage",
    "salary": "salary (high→low)",
}

# Minimum-salary floors cycled by `m` (annual; 0 = off). Compared against each job's parsed
# annual minimum (see parse_salary_to_annual_min).
_SALARY_CYCLE: list[int] = [0, 40_000, 60_000, 80_000, 100_000, 120_000, 150_000]


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


class _JobCard:
    """One job's list row, as a custom Rich renderable so the stage-coloured left spine ``▌``
    runs down EVERY line of the card — including wrapped continuations — and the card re-wraps to
    the OptionList's width automatically (no manual width math / resize handling). Takes the card
    lines (title · company · meta) so each gets the spine and wraps independently.

    ``__str__`` returns the plain text so callers (and tests) can read the content."""

    _GUTTER = "▌ "  # the spine + one space; 2 cells

    def __init__(self, lines: list[Text], spine_style: str, divider_style: str) -> None:
        self._lines = lines
        self._spine_style = spine_style
        self._divider_style = divider_style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        inner = max(4, options.max_width - len(self._GUTTER))  # content width past the spine
        for source in self._lines:
            for line in source.wrap(console, inner):
                row = Text(self._GUTTER, style=self._spine_style)
                row.append_text(line)
                yield row
        # A horizontal rule in the theme's $panel colour (the same hue as the pane border) divides
        # the rows into a table — compact structure in place of the old blank gap. The spine rail
        # continues through it so the stage-coloured left edge stays unbroken.
        divider = Text(self._GUTTER, style=self._spine_style)
        divider.append("─" * inner, style=self._divider_style)
        yield divider

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        # Never ask for more than the available width — so the OptionList wraps us, never scrolls.
        return Measurement(min(20, options.max_width), options.max_width)

    def __str__(self) -> str:
        return "\n".join(line.plain for line in self._lines)


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


class JobList(OptionList):
    """The job list. An OptionList (not a DataTable) so each job is a multi-line card that
    WRAPS to the pane width instead of overflowing horizontally, with a full-width highlight.
    Overrides the loading widget so a running search shows a themed indicator over the app
    surface instead of the framework default's bare grey overlay."""

    def get_loading_widget(self) -> Widget:
        return _JobListLoading()


class JobApplicatorApp(App[None]):
    """Navigable home screen over the funnel store — browse and act on jobs in-app."""

    TITLE = "job-applicator"

    CSS = """
    #statusline { height: 4; border: round $primary; padding: 0 1; }
    /* Tabs read as a clickable control, not flat indicators: inactive tabs are dimmed and the
       active one is a filled pill (accent background). */
    #stagetabs { height: 1; }
    #stagetabs Tab { color: $text-muted; }
    #stagetabs Tab.-active { background: $accent; color: $background; text-style: bold; }
    #body { height: 1fr; }
    /* height: 1fr so the list fills the body like the detail pane. border: none clears the
       OptionList default all-side border (which otherwise eats 2 rows / shrinks the list);
       keep only the right divider between the list and the detail pane. */
    #joblist { width: 45%; height: 1fr; border: none; border-right: solid $panel; padding: 0; }
    /* max-width caps the reading measure: a full-pane-wide description line (~110 cols on a
       wide terminal) is hard to read; ~90 is a comfortable column. Narrower terminals are
       unaffected (pane < 90). */
    #detail { width: 1fr; max-width: 90; padding: 0 1; }
    #filter { dock: bottom; display: none; border: round $accent; }
    #filter.visible { display: block; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        # The footer shows PRIMARY keys only; the secondary long-tail (PDF exports, ATS, style
        # guide, copy) is show=False — still bound, still in Help (?) — so the footer reads as a
        # lean bar, not a wall. This also drops the a/A footer collision (ATS is hidden).
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("t", "tailor", "Tailor"),
        Binding("T", "tailor_pdf", "Tailor PDF", show=False),
        Binding("c", "cover_letter", "Cover letter"),
        Binding("C", "cover_letter_pdf", "Cover PDF", show=False),
        Binding("p", "open_pdf", "Open PDF", show=False),
        Binding("s", "search", "Search"),
        Binding("a", "apply", "Apply"),
        Binding("e", "set_resume", "Résumé"),
        Binding("g", "set_style_guide", "Style guide", show=False),
        Binding("A", "ats_check", "ATS", show=False),
        Binding("o", "open_url", "Open"),
        Binding("y", "copy_url", "Copy URL", show=False),
        Binding("slash", "filter", "Find"),
        Binding("f", "open_filters", "Filter"),
        # The individual cycle keys still work for power users, but are off the footer now that
        # `f` opens a grouped Filter & sort panel (board/salary/sort/text/stage in one place).
        Binding("b", "cycle_board_filter", "Board", show=False),
        Binding("m", "cycle_min_salary", "Min $", show=False),
        Binding("u", "toggle_unlisted", "Unlisted", show=False),
        Binding("S", "cycle_sort", "Sort", show=False),
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
        self._min_salary = 0  # min annual-salary floor (0 = off; see _SALARY_CYCLE)
        self._hide_unlisted = False  # when True, hide jobs with no parseable salary
        self._load_error = ""
        # True while we programmatically move the stage tab to mirror _stage_filter, so the
        # tab-activated handler doesn't re-apply/reload. Starts True to swallow the auto-activation
        # of the first tab during mount (on_mount drives the first real reload itself).
        self._tab_sync = True

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Static(id="statusline")
        yield Tabs(
            *(Tab(label, id=_stage_to_tab(stage)) for stage, label in _STAGE_TAB_LABEL.items()),
            id="stagetabs",
        )
        with Horizontal(id="body"):
            yield JobList(id="joblist")
            yield VerticalScroll(Static(id="detail"), id="detailscroll")
        yield Input(id="filter", placeholder="filter title/company — Enter to apply, Esc to clear")
        yield Footer()

    def on_mount(self) -> None:
        self._reload()  # builds the job cards + seeds the stage-tab counts + sets the highlight
        self._tab_sync = False  # mount done — real tab clicks now apply + reload
        self.query_one("#joblist", JobList).focus()  # the list owns focus, not tabs / filter

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """A stage tab was clicked/selected → apply it as the stage filter. Suppressed while we
        sync the tab to ``_stage_filter`` (see ``_sync_stage_tab``) so it can't loop."""
        if self._tab_sync:
            return
        stage = _tab_to_stage(event.tab.id)
        if stage != self._stage_filter:
            self._stage_filter = stage
            self._reload(refresh_applied=False)
        self.query_one("#joblist", JobList).focus()  # clicking a tab returns focus to the list

    def _sync_stage_tab(self) -> None:
        """Move the active stage tab to mirror ``_stage_filter`` WITHOUT triggering a reload."""
        try:
            tabs = self.query_one("#stagetabs", Tabs)
        except NoMatches:
            return
        self._tab_sync = True
        try:
            tabs.active = _stage_to_tab(self._stage_filter)
        finally:
            self._tab_sync = False

    def _update_tab_counts(self) -> None:
        """Show the live per-stage counts in the tab labels (All = total)."""
        try:
            tabs = self.query_one("#stagetabs", Tabs)
        except NoMatches:
            return
        counts: dict[str, int] = {}
        for s in self._all:
            stage = self._effective_stage(s)
            counts[stage] = counts.get(stage, 0) + 1
        head_urls = {str(s.job.url) for s in self._all}
        counts["applied"] = counts.get("applied", 0) + len(self._applied_urls - head_urls)
        for tab_stage, label in _STAGE_TAB_LABEL.items():
            n = len(self._all) if tab_stage is None else counts.get(tab_stage, 0)
            try:
                tabs.query_one(f"#{_stage_to_tab(tab_stage)}", Tab).label = f"{label} {n}"
            except NoMatches:
                pass

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
        self._update_tab_counts()  # refresh the per-stage counts shown on the tabs
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
        elif self._sort_mode == "salary":
            # Highest parsed annual salary first; jobs with no listed/parseable salary sort
            # last (the `is not None` flag dominates the descending sort).
            def _salary_key(s: StoredJob) -> tuple[bool, int]:
                value = parse_salary_to_annual_min(s.job.salary)
                return (value is not None, value or 0)

            self._all.sort(key=_salary_key, reverse=True)
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
        if self._min_salary > 0 or self._hide_unlisted:
            # Both salary filters in ONE pass (parse each job's salary once): an unlisted job is
            # kept unless 'hide unlisted' is on; a listed job is kept when it clears the floor.
            def _salary_ok(s: StoredJob) -> bool:
                value = parse_salary_to_annual_min(s.job.salary)
                if value is None:
                    return not self._hide_unlisted
                return self._min_salary == 0 or value >= self._min_salary

            jobs = [s for s in jobs if _salary_ok(s)]
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

    def _job_card(self, s: StoredJob) -> _JobCard:
        """A three-line job card (rendered by ``_JobCard`` so the stage-coloured spine spans all
        lines, incl. wraps), giving a clear hierarchy:
          line 1 — the FULL title (bold)
          line 2 — the company (green, its own line so it isn't lost in the metadata)
          line 3 — match score (sentiment colour) · board (brand badge) · location · found-via query
        """
        spine = _STAGE_STYLE.get(self._effective_stage(s), "white")
        title = Text(s.job.title, style="bold")
        company = Text(s.job.company or "—", style="green")
        meta = Text()
        score = s.match_score
        meta.append(f"{score:.0%}" if score is not None else "—", style=_score_style(score))
        # Mark a coverage-unknown score (no requirements → semantic-only) so a low rank isn't
        # misread as weak skills at list-scan time; the detail pane spells it out.
        if score is not None and not coverage_measured(s.matched_skills, s.missing_skills):
            meta.append("*", style="dim")
        meta.append("  ·  ", style="dim")
        meta.append(s.job.board.display_name, style=_BOARD_STYLE.get(s.job.board.value, "white"))
        if s.job.location:
            meta.append("  ·  ", style="dim")
            meta.append(s.job.location, style="dim")
        if s.source_query:  # provenance — the search that first surfaced this job
            meta.append("  ·  via ", style="dim")
            meta.append(f"'{s.source_query}'", style="dim italic")
        # Match the row divider to the pane border ($panel) so the table grid reads as one piece.
        divider = self.theme_variables.get("panel", "#242f38")
        return _JobCard([title, company, meta], spine, divider)

    def _repaint(self) -> None:
        self.query_one("#statusline", Static).update(self._statusline())
        jobs = self.query_one("#joblist", JobList)
        # Remember the selected job so it survives the rebuild (OptionList.highlighted is None
        # until set, so we re-select explicitly below — else the detail pane goes blank on load
        # and after every refresh/filter/sort/scrape).
        prev_key = str(self._current.id) if self._current is not None else None
        jobs.clear_options()
        self._by_key = {}
        visible = self._visible()
        for s in visible:
            key = str(s.id)
            self._by_key[key] = s
            jobs.add_option(Option(self._job_card(s), id=key))
        # Re-select the SAME job (the option set changes under a filter/sort); fall back to the
        # first option only when the previously-selected job is no longer visible.
        if visible:
            restore = next((i for i, s in enumerate(visible) if str(s.id) == prev_key), 0)
            jobs.highlighted = restore
            self._update_detail(visible[restore])
        else:
            jobs.highlighted = None
            self._update_detail(None)

    # ------------------------------------------------------------------ render
    def _statusline(self) -> str:
        if self._load_error:
            return f"[red]⚠ {escape(self._load_error)}[/red]"
        # Show the résumé/style-guide BASENAME (the full path is long and ate the whole line);
        # the pre-styled sentinel stays as-is when unset (escape only the real value).
        path = self._settings.resume_path
        resume = (
            f"[cyan]{escape(Path(path).name)}[/cyan]"
            if path
            else "[dim]not set — press 'e' to set[/dim]"
        )
        sg_path = self._settings.style_guide_path
        style = (
            f"[cyan]{escape(Path(sg_path).name)}[/cyan]"
            if sg_path
            else "[dim]none — press 'g'[/dim]"
        )
        if self._busy:  # a worker is running — show live progress instead of static counts
            return f"Résumé {resume}   Style: {style}\n[yellow]⏳ {escape(self._busy)}[/yellow]"
        # Line 2 carries the position (row N/M), the sort, the board/salary/text filters, and the
        # resulting "N shown". Per-stage counts and the active stage live on the tabs.
        view: list[str] = []
        try:  # current position in the list (updates on every cursor move; see the highlight hook)
            jobs = self.query_one("#joblist", JobList)
            if jobs.highlighted is not None and jobs.option_count:
                view.append(f"{jobs.highlighted + 1}/{jobs.option_count}")
        except (NoMatches, ScreenStackError):  # not mounted yet (e.g. a unit-test _statusline call)
            pass
        view.append(f"sort: {_SORT_LABEL[self._sort_mode]}")
        if self._board_filter is not None:
            view.append(f"board: {JobBoard(self._board_filter).display_name}")
        if self._min_salary > 0:
            view.append(f"min ${self._min_salary // 1000}k")
        if self._hide_unlisted:
            view.append("listed pay only")
        if self._filter:
            view.append(f"text: {escape(self._filter)}")
        # "N shown" is for the NON-stage filters only: a stage tab already shows its own count,
        # and when a stage tab + a board/etc. filter combine, len(visible) legitimately differs.
        narrowed = (
            self._board_filter is not None
            or self._min_salary > 0
            or self._hide_unlisted
            or bool(self._filter)
        )
        tail = f" — {len(self._visible())} shown" if narrowed else ""
        return f"Résumé {resume}   Style: {style}\n[dim]{' · '.join(view)}{tail}[/dim]"

    def _set_busy(self, msg: str) -> None:
        """Show a live '⏳ <msg>' progress line while a worker runs (empty string clears
        it, restoring the sort/filter line) — so latency never looks like a freeze."""
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
            "[bold]Overview[/bold]",  # section header, matching the Description header below
            f"Stage     [{style}]{stage.replace('_', ' ')}[/{style}]",
            f"Location  {escape(j.location) if j.location else '—'}",
            f"Salary    {escape(j.salary) if j.salary else '—'}",
        ]
        if s.match_score is not None:
            sem = s.semantic_score or 0
            if coverage_measured(s.matched_skills, s.missing_skills):
                skill = s.skill_score or 0
                breakdown = f"(semantic {sem:.0%} · skill {skill:.0%})"
            else:
                # No requirements listed → skill_score is 0.0 by convention, not a real 0%.
                breakdown = f"(semantic {sem:.0%} · coverage n/a — none listed)"
            lines.append(
                f"Match     [{_score_style(s.match_score)}]{s.match_score:.0%}[/]  "
                f"[dim]{breakdown}[/dim]"
            )
            lines.append(
                "[dim]Score = skill-overlap, not role-fit — sparse/junior roles score low.[/dim]"
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
        if s.pdf_path:
            name = escape(_elide_mid(Path(s.pdf_path).name, 40))
            lines.append(
                f"Résumé PDF  [@click=app.open_pdf][dim]{name}[/dim][/]  [dim](click to open)[/dim]"
            )
        if s.cover_letter_path:
            name = escape(_elide_mid(Path(s.cover_letter_path).name, 40))
            lines.append(
                f"Cover     [@click=app.open_cover][dim]{name}[/dim][/]  [dim](click to open)[/dim]"
            )
        if s.cover_letter_pdf_path:
            name = escape(_elide_mid(Path(s.cover_letter_pdf_path).name, 40))
            lines.append(
                f"Cover PDF  [@click=app.open_cover_pdf][dim]{name}[/dim][/]"
                "  [dim](click to open)[/dim]"
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
            # Reflow the raw scrape into readable markup (the formatter escapes its own text and
            # adds the header [bold], so it is NOT re-escaped here). Store keeps ~5k chars; the
            # pane scrolls. Truncating here would hide the rest of the posting.
            lines += ["", "[bold]Description[/bold]", "", format_job_description(j.description)]
        return "\n".join(lines)

    # ----------------------------------------------------------------- actions
    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        key = event.option.id
        if key is not None:
            self._update_detail(self._by_key.get(key))
        # Refresh the status so the "row N/M" position tracks the cursor.
        self.query_one("#statusline", Static).update(self._statusline())

    def action_refresh(self) -> None:
        self._reload()

    def action_open_filters(self) -> None:
        """Open the grouped Filter & sort panel (`f`) — stage · board · min salary · hide-unlisted
        · sort · text in one place. The app builds the option lists from its constants and applies
        the result in ``_apply_filters`` (so the panel screen needs no import back here)."""
        from job_applicator.tui.screens import FilterScreen

        stage_opts = [
            (label, "all" if st is None else st) for st, label in _STAGE_TAB_LABEL.items()
        ]
        board_opts = [("All boards", "all"), *((b.display_name, b.value) for b in JobBoard)]
        salary_opts = [("Any", "0"), *((f"${v // 1000}k+", str(v)) for v in _SALARY_CYCLE[1:])]
        sort_opts = [(label, key) for key, label in _SORT_LABEL.items()]
        self.push_screen(
            FilterScreen(
                stage_options=stage_opts,
                stage_value="all" if self._stage_filter is None else self._stage_filter,
                board_options=board_opts,
                board_value="all" if self._board_filter is None else self._board_filter,
                salary_options=salary_opts,
                salary_value=str(self._min_salary),
                sort_options=sort_opts,
                sort_value=self._sort_mode,
                hide_unlisted=self._hide_unlisted,
                text=self._filter,
            ),
            self._apply_filters,
        )

    def _apply_filters(self, result: dict[str, object] | None) -> None:
        """Apply the Filter panel's result (None = cancelled). Sets every view control at once,
        mirrors stage on the tabs, and re-queries. View-only; never touches the account."""
        joblist = self.query_one("#joblist", JobList)
        if result is None:
            joblist.focus()
            return
        stage = result["stage"]
        board = result["board"]
        self._stage_filter = None if stage == "all" else str(stage)
        self._board_filter = None if board == "all" else str(board)
        self._min_salary = int(str(result["min_salary"]))
        self._hide_unlisted = bool(result["hide_unlisted"])
        self._sort_mode = str(result["sort_mode"])
        self._filter = str(result["text"])
        self._sync_stage_tab()
        self._reload(refresh_applied=False)
        joblist.focus()

    def action_cycle_stage_filter(self) -> None:
        """Cycle the funnel-stage filter (all → each stage → all) and move the matching tab.
        View-only; kept for the tabs/`f`-panel to share; composes with the other filters."""
        i = _STAGE_CYCLE.index(self._stage_filter)
        self._stage_filter = _STAGE_CYCLE[(i + 1) % len(_STAGE_CYCLE)]
        self._sync_stage_tab()  # mirror the cycle on the tab bar (no reload — view-only)
        self._repaint()

    def action_cycle_board_filter(self) -> None:
        """Cycle the board filter (all → each board → all). View-only; composes with the
        stage + text filters."""
        i = _BOARD_CYCLE.index(self._board_filter)
        self._board_filter = _BOARD_CYCLE[(i + 1) % len(_BOARD_CYCLE)]
        self._repaint()

    def action_cycle_min_salary(self) -> None:
        """Cycle the minimum-salary floor (off → $40k → … → $150k → off). View-only filter;
        jobs with no listed salary are kept unless 'hide unlisted' (u) is on."""
        i = _SALARY_CYCLE.index(self._min_salary) if self._min_salary in _SALARY_CYCLE else 0
        self._min_salary = _SALARY_CYCLE[(i + 1) % len(_SALARY_CYCLE)]
        self._repaint()

    def action_toggle_unlisted(self) -> None:
        """Toggle hiding jobs that have no listed/parseable salary. View-only."""
        self._hide_unlisted = not self._hide_unlisted
        self._repaint()

    def action_cycle_sort(self) -> None:
        """Cycle the list sort order (best match → recent → funnel stage → salary). View-only."""
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

    def action_open_pdf(self) -> None:
        """Open the selected job's generated PDF.

        Prefers the tailored-résumé PDF and falls back to the cover-letter PDF so a single
        keybinding provides a quick preview of whichever artifact exists.
        """
        s = self._current
        path = s.pdf_path if s else None
        label = "résumé PDF"
        if not path and s:
            path = s.cover_letter_pdf_path
            label = "cover letter PDF"
        self._open_artifact(path, label)

    def action_open_cover_pdf(self) -> None:
        """Open the cover-letter PDF for the selected job (if one was generated)."""
        s = self._current
        self._open_artifact(s.cover_letter_pdf_path if s else None, "cover letter PDF")

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

    def action_set_style_guide(self) -> None:
        """Set the style-guide path(s) in-app — opens a modal, saves to config."""
        from job_applicator.tui.screens import StyleGuideScreen

        self.push_screen(
            StyleGuideScreen(self._settings.style_guide_path), self._set_style_guide_then
        )

    def _set_style_guide_then(self, path: str | None) -> None:
        if path is None:  # cancelled (empty string is a deliberate clear)
            return
        self._settings.style_guide_path = path
        saved = self._persist_style_guide_path(path)
        self._reload()
        if saved is not None:
            self.notify(f"Style guide set ✓ — saved to {saved}", timeout=6)
        else:
            self.notify(
                "Style guide set for this session (couldn't write config — set "
                "style_guide_path manually to persist).",
                severity="warning",
                timeout=8,
            )

    def _persist_config_key(self, key: str, value: str) -> Path | None:
        """Best-effort: write a top-level ``key = value`` into the config file.

        Safe over a credentialed config: the result is re-parsed and must yield exactly
        this top-level key (so a line accidentally rewritten inside a ``[table]`` or a
        duplicate key that won't parse is rejected), and the swap is ATOMIC
        (temp file + ``os.replace``) so a torn write can never destroy the config. Returns
        the file written, or None on any failure.
        """
        import json
        import os
        import re
        import tomllib

        from job_applicator.config import CONFIG_FILE_ENV_VAR, DEFAULT_CONFIG_FILE

        cfg = Path(os.environ.get(CONFIG_FILE_ENV_VAR, DEFAULT_CONFIG_FILE))
        line = f"{key} = {json.dumps(value)}"
        try:
            if cfg.exists():
                text = cfg.read_text(encoding="utf-8")
                # Match an ACTIVE line only (no leading '#', so we never un-comment an
                # example into a duplicate key); else prepend at the very top (top-level).
                pattern = re.compile(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$")
                new_text = (
                    pattern.sub(line, text, count=1) if pattern.search(text) else f"{line}\n{text}"
                )
            else:
                new_text = f"# job-applicator config\n{line}\n"
            # Validate before committing: must parse AND set the key at top level.
            if tomllib.loads(new_text).get(key) != value:
                return None
            tmp = cfg.with_name(f"{cfg.name}.tmp")
            tmp.write_text(new_text, encoding="utf-8")
            os.replace(tmp, cfg)  # atomic swap
        except (OSError, ValueError):  # ValueError covers tomllib.TOMLDecodeError
            return None
        return cfg

    def _persist_resume_path(self, path: str) -> Path | None:
        """Persist ``resume_path`` to config.toml."""
        return self._persist_config_key("resume_path", path)

    def _persist_style_guide_path(self, path: str) -> Path | None:
        """Persist ``style_guide_path`` to config.toml."""
        return self._persist_config_key("style_guide_path", path)

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
            self._tailor_worker(job, output_format=Format.TXT)

    def action_tailor_pdf(self) -> None:
        """Tailor and render a PDF résumé for the selected job. Account-safe."""
        job = self._selected_job_with_resume()
        if job is not None:
            self._tailor_worker(job, output_format=Format.PDF)

    @work(exclusive=True, group="action")
    async def _tailor_worker(self, job: JobListing, *, output_format: Format = Format.TXT) -> None:
        from job_applicator.tui import actions

        label = (
            f"Tailoring PDF for {job.title}…"
            if output_format == Format.PDF
            else f"Tailoring {job.title}…"
        )
        self._set_busy(label)
        try:
            tailored = await self._run_action(
                "Tailor",
                actions.tailor_job(
                    self._settings,
                    job,
                    style_guide_path=self._settings.style_guide_path,
                    output_format=output_format,
                ),
            )
        finally:
            self._set_busy("")
        if tailored is None:
            return
        try:
            self._store.mark_tailored(
                job,
                tailored_resume_path=tailored.output_path,
                pdf_path=tailored.pdf_path,
            )
            self._reload()
        except JobApplicatorError as exc:
            # The artifact IS written; only the funnel update failed (e.g. DB locked). Surface it
            # as a warning so the success isn't lost to a crashed worker + no toast.
            self.notify(
                f"Tailored ✓  →  {tailored.output_path}  "
                f"(funnel update failed: {escape(str(exc))})",
                severity="warning",
                timeout=8,
            )
            return
        self.notify(f"Tailored ✓  →  {tailored.output_path}", timeout=6)

    def action_cover_letter(self) -> None:
        """Write a cover letter for the selected job in a background worker. Account-safe.
        If the job was already tailored, the letter draws on the tailored résumé."""
        job = self._selected_job_with_resume()
        if job is not None:
            tailored = self._current.tailored_resume_path if self._current else ""
            self._cover_letter_worker(job, tailored, output_format=Format.TXT)

    def action_cover_letter_pdf(self) -> None:
        """Write a PDF cover letter for the selected job. Account-safe."""
        job = self._selected_job_with_resume()
        if job is not None:
            tailored = self._current.tailored_resume_path if self._current else ""
            self._cover_letter_worker(job, tailored, output_format=Format.PDF)

    @work(exclusive=True, group="action")
    async def _cover_letter_worker(
        self,
        job: JobListing,
        tailored_resume_path: str,
        *,
        output_format: Format = Format.TXT,
    ) -> None:
        from job_applicator.tui import actions

        label = (
            f"Writing PDF cover letter for {job.title}…"
            if output_format == Format.PDF
            else f"Writing a cover letter for {job.title}…"
        )
        self._set_busy(label)
        try:
            result = await self._run_action(
                "Cover letter",
                actions.cover_letter_job(
                    self._settings,
                    job,
                    tailored_resume_path=tailored_resume_path,
                    style_guide_path=self._settings.style_guide_path,
                    output_format=output_format,
                ),
            )
        finally:
            self._set_busy("")
        if result is None:
            return
        try:
            self._store.set_cover_letter(
                str(job.url),
                result.output_path,
                cover_letter_pdf_path=result.pdf_path,
            )
            self._reload()
        except JobApplicatorError as exc:
            self.notify(
                f"Cover letter ✓  →  {result.output_path}  "
                f"(funnel update failed: {escape(str(exc))})",
                severity="warning",
                timeout=8,
            )
            return
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

        table = self.query_one("#joblist", JobList)
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
        # Per-board (name, count|None) so the summary can show EACH board's contribution —
        # a board that returns 0 (e.g. Indeed blocked upstream of extraction → 0 cards, which
        # is NOT an error and so isn't caught as a failure) must be visible as "Indeed: 0",
        # never silently folded into a total that credits it as a succeeded board.
        results: list[tuple[str, int | None]] = []
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
                results.append((params.board.display_name, found))
                if found is not None:  # None = this board errored (already toasted); skip it
                    total += found
        finally:
            table.loading = False
            self._set_busy("")
        # Final repaint after scoring: the streamed (found) rows now carry scores and
        # re-sort best-match-first (the reorder-on-score).
        self._reload()
        if all(found is None for _, found in results):
            return  # every attempted board errored — _run_action already toasted each
        # Per-board breakdown ("LinkedIn: 10 · Indeed: 0 · Foo: failed") so a zero/failed board
        # is honestly surfaced instead of a lumped total that hides which board came up empty.
        breakdown = " · ".join(
            f"{name}: {found if found is not None else 'failed'}" for name, found in results
        )
        empty_or_failed = any(found is None or found == 0 for _, found in results)
        self.notify(
            f"Found {total} job(s) — {breakdown} — added to your funnel.",
            severity="warning" if empty_or_failed else "information",
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

        # Capture the job AND its cover-letter path together, here — so they can't diverge if the
        # selection (_current) changes while the modal is open / the worker runs later.
        job = self._current.job
        cover_letter_path = self._current.cover_letter_path
        self.push_screen(ApplyScreen(job), partial(self._apply_dispatch, job, cover_letter_path))

    def _apply_dispatch(self, job: JobListing, cover_letter_path: str, submit: bool | None) -> None:
        # None = cancelled. Re-check busy at dispatch too (a stacked confirm could otherwise
        # start a second account worker past the modal-open gate).
        if submit is None or self._account_busy_refused():
            return
        self._apply_worker(job, cover_letter_path, submit=submit)

    @work(group="account")
    async def _apply_worker(self, job: JobListing, cover_letter_path: str, *, submit: bool) -> None:
        from job_applicator.tui import actions

        mode = "Submitting a real application to" if submit else "Dry-run for"
        self._set_busy(f"{mode} {job.title} — a browser will open…")
        # Attach the cover letter generated for this job (matching the CLI). The path was captured
        # WITH the job at action time, so it always belongs to the job being applied. Read the
        # stored TXT off the event loop; a missing/unreadable file just applies without one.
        cover_letter: str | None = None
        if cover_letter_path:
            try:
                cover_letter = await asyncio.to_thread(
                    Path(cover_letter_path).read_text, encoding="utf-8"
                )
            except OSError:
                cover_letter = None
        try:
            result = await self._run_action(
                "Apply",
                actions.apply_job(self._settings, job, submit=submit, cover_letter=cover_letter),
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
        self.query_one("#joblist", JobList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#joblist", JobList).action_cursor_up()

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
        self._min_salary = 0
        self._hide_unlisted = False
        self._sync_stage_tab()  # reset the tab bar to "All" too
        self._repaint()
        self.query_one("#joblist", JobList).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._filter = event.value.strip()
        event.input.remove_class("visible")
        self._repaint()
        self.query_one("#joblist", JobList).focus()


def run_tui(settings: AppSettings) -> None:
    """Build the local store(s) and run the TUI. Offline + account-safe."""
    from job_applicator.jobs_store import JobStore
    from job_applicator.state import ApplicationState

    JobApplicatorApp(settings=settings, store=JobStore(), app_state=ApplicationState()).run()
