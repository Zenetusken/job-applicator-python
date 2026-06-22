"""The Textual application: a read-only browser over the job-funnel store.

Layout: a status line (résumé + funnel summary), a job-list sidebar (the funnel head
from ``JobStore``), a detail pane for the highlighted job, and a footer keybar. This
increment is the *shell* — navigation only; tailor/cover-letter/apply actions land in a
later increment. Launch is offline and account-safe (reads the local SQLite store only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import DataTable, Footer, Input, Static

from job_applicator.exceptions import JobApplicatorError
from job_applicator.models import ApplicationStatus, FunnelStatus

if TYPE_CHECKING:
    from job_applicator.config import AppSettings
    from job_applicator.jobs_store import JobStore
    from job_applicator.models import StoredJob
    from job_applicator.state import ApplicationState

# Stage → display glyph/colour for the sidebar + detail.
_STAGE_STYLE: dict[str, str] = {
    "found": "white",
    "matched": "cyan",
    "tailored": "yellow",
    "cover_letter": "magenta",
    "applied": "green",
}


class JobApplicatorApp(App[None]):
    """Navigable home screen over the funnel store (read-only shell)."""

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
        self._applied_count = 0
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
            self._applied_count = sum(
                1
                for a in self._app_state.list_recent(limit=10_000)
                if a.status == ApplicationStatus.SUBMITTED
            )
        except JobApplicatorError as exc:
            self._all, self._applied_count = [], 0
            self._load_error = str(exc)
        self._repaint()

    def _visible(self) -> list[StoredJob]:
        if not self._filter:
            return self._all
        needle = self._filter.lower()
        return [
            s for s in self._all if needle in s.job.title.lower() or needle in s.job.company.lower()
        ]

    def _repaint(self) -> None:
        self.query_one("#statusline", Static).update(self._statusline())
        table = self.query_one("#joblist", DataTable)
        table.clear()
        self._by_key = {}
        visible = self._visible()
        for s in visible:
            key = str(s.id)
            self._by_key[key] = s
            stage = s.funnel_status.value
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
        resume = self._settings.resume_path or "[dim]not set — configure resume_path[/dim]"
        if self._load_error:
            return f"[red]⚠ {escape(self._load_error)}[/red]"
        counts: dict[str, int] = {}
        for s in self._all:
            counts[s.funnel_status.value] = counts.get(s.funnel_status.value, 0) + 1
        parts = [
            f"{counts.get(st.value, 0)} {st.value.replace('_', ' ')}"
            for st in FunnelStatus
            if st is not FunnelStatus.APPLIED
        ]
        parts.append(f"{self._applied_count} applied")
        shown = len(self._visible())
        filt = (
            f"   [dim](filter: {escape(self._filter)} — {shown} shown)[/dim]"
            if self._filter
            else ""
        )
        return f"Résumé [cyan]{escape(str(resume))}[/cyan]\n{' · '.join(parts)}{filt}"

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
        stage = s.funnel_status.value
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
            lines.append(f"Résumé    [dim]{escape(s.tailored_resume_path)}[/dim]")
        if s.cover_letter_path:
            lines.append(f"Cover     [dim]{escape(s.cover_letter_path)}[/dim]")
        lines += ["", f"[blue underline]{escape(str(j.url))}[/blue underline]"]
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
