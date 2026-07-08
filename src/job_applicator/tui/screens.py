"""Modal screens for the TUI.

Account-safety note: ``SearchScreen`` only *collects* parameters. Submitting it — the
deliberate act, taken with the "opens a browser on your real account" warning visible —
is what authorizes the scrape; the screen itself touches nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from job_applicator.models import JobBoard
from job_applicator.scrapers.base import SearchParams

if TYPE_CHECKING:
    from job_applicator.models import ATSCompatibilityResult, JobListing
    from job_applicator.tui.actions import DocumentQualityResult

T = TypeVar("T")


class _FadeModalScreen(ModalScreen[T]):
    """Base for the app's dialogs: a subtle fade-in on mount so a modal reads as a layer
    that *appears* over the dimmed app, rather than snapping in. Animating the screen's
    opacity fades the dim backdrop and the box in together.

    The fade honours reduced-motion: ``styles.animate`` respects the app's ``AnimationLevel``
    (``TEXTUAL_ANIMATIONS``), degrading to an instant show when animations are disabled.
    Subclasses that override ``on_mount`` (e.g. to focus an input) must call
    ``super().on_mount()``.
    """

    _FADE_DURATION = 0.18

    def on_mount(self) -> None:
        self.styles.opacity = 0.0
        self.styles.animate("opacity", value=1.0, duration=self._FADE_DURATION)


class SearchScreen(_FadeModalScreen[list[SearchParams] | None]):
    """A search form across one or both boards. Dismisses with a list of ``SearchParams`` —
    one per selected board, each carrying that board's own result count — which authorizes
    the scrape, or ``None`` on cancel/Esc.

    Account-safety: only LinkedIn reuses the real account; Indeed search is public (a clean,
    windowless browser, no login). The warning is board-aware, so the user always sees which
    of the two they are about to touch."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    _MAX_RESULTS_CAP = 50

    # (checkbox id, board, count-input id, warning shown while that board is selected). The
    # default count is 25 (matches SearchParams.max_results and the CLI `search --max`).
    _BOARDS: ClassVar[list[tuple[str, JobBoard, str, str]]] = [
        (
            "bd_linkedin",
            JobBoard.LINKEDIN,
            "n_linkedin",
            "⚠  LinkedIn opens a browser on your real account.",
        ),
        (
            "bd_indeed",
            JobBoard.INDEED,
            "n_indeed",
            "Indeed search is public - a clean, windowless browser, no login.",
        ),
    ]

    # The box auto-sizes to its content (snug on a normal terminal) but caps at 90% of the
    # screen and SCROLLS as a whole when the terminal is too short — so the Search/Cancel
    # buttons can never be pushed off-screen. The warning sits directly above the buttons, so
    # it's on-screen whenever they are (account-safety stays visible at the moment of submit).
    CSS = """
    SearchScreen { align: center middle; }
    #searchbox {
        width: 68; height: auto; max-height: 90%; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #searchbox Input, #searchbox Checkbox { margin: 1 0; }
    .boardrow { height: auto; }
    .boardrow Checkbox { width: 18; }
    .ncount { width: 14; }
    #warn { color: $warning; margin: 1 0; }
    #buttons { height: auto; align: right middle; }
    #buttons Button { margin-left: 2; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="searchbox"):
            yield Label("[bold]Search jobs[/bold]")
            yield Input(placeholder="query - e.g. senior python engineer", id="q")
            yield Input(placeholder="location (optional)", id="loc")
            yield Checkbox("Remote only", id="remote")
            yield Label("[bold]Boards[/bold]   [dim]results per board (1-50)[/dim]")
            with Horizontal(classes="boardrow"):
                yield Checkbox("LinkedIn", value=True, id="bd_linkedin")
                yield Input(value="25", id="n_linkedin", type="integer", classes="ncount")
            with Horizontal(classes="boardrow"):
                yield Checkbox("Indeed", value=False, id="bd_indeed")
                yield Input(value="25", id="n_indeed", type="integer", classes="ncount")
            yield Static("", id="warn")
            with Horizontal(id="buttons"):
                yield Button("Search", variant="primary", id="go")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        super().on_mount()  # fade-in
        self.query_one("#q", Input).focus()
        self._update_warning()  # seed the board-aware warning for the default selection

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        # Any board toggle re-derives the warning (the Remote checkbox is harmless here).
        self._update_warning()

    def _warning_text(self) -> str:
        """The board-aware warning for the current selection (or a prompt when none) — pure,
        so it reads the same whether or not the ``#warn`` Static is mounted yet."""
        lines = [msg for cb, _b, _n, msg in self._BOARDS if self._checked(cb)]
        return "\n".join(lines) if lines else "[red]Select at least one board.[/red]"

    def _update_warning(self) -> None:
        """Refresh the warning Static. Guarded so a ``Checkbox.Changed`` fired during mount
        (before ``#warn`` exists) is a no-op — ``on_mount`` seeds it once the tree is up."""
        try:
            warn = self.query_one("#warn", Static)
        except NoMatches:
            return
        warn.update(self._warning_text())

    def _checked(self, checkbox_id: str) -> bool:
        return self.query_one(f"#{checkbox_id}", Checkbox).value

    def _count(self, input_id: str) -> int:
        """A board's result count - clamped to 1…cap. Empty/invalid → the default (the Input
        is integer-only, so a non-numeric value shouldn't reach here)."""
        raw = self.query_one(f"#{input_id}", Input).value.strip()
        try:
            n = int(raw) if raw else 25
        except ValueError:
            n = 25
        return max(1, min(self._MAX_RESULTS_CAP, n))

    def _submit(self) -> None:
        query = self.query_one("#q", Input).value.strip()
        if not query:
            self.notify("Enter a search query.", severity="warning")
            return
        location = self.query_one("#loc", Input).value.strip()
        remote = self.query_one("#remote", Checkbox).value
        plans = [
            SearchParams(
                query=query,
                location=location,
                remote_only=remote,
                max_results=self._count(n_id),
                board=board,
            )
            for cb, board, n_id, _msg in self._BOARDS
            if self._checked(cb)
        ]
        if not plans:
            self.notify("Select at least one board to search.", severity="warning")
            return
        self.dismiss(plans)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in any field submits the form (same as the Search button).
        self._submit()


class FilterScreen(_FadeModalScreen["dict[str, object] | None"]):
    """One panel for ALL the list's view controls — stage, board, minimum salary, hide-unlisted,
    sort, and a text filter — opened with ``f`` so they're discoverable in one place instead of a
    row of cycle keys. Dismisses with a dict of the chosen values to apply, or ``None`` on
    cancel/Esc. The option lists + current values are passed in by the app (which owns the
    constants), so this screen has no import back to ``app.py``.

    View-only: changing filters/sort never touches the account; it only re-queries the store."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    CSS = """
    FilterScreen { align: center middle; }
    #filterbox {
        width: 64; height: auto; max-height: 90%; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #filterbox Select, #filterbox Input { margin: 1 0; }
    #fbuttons { height: auto; align: right middle; margin-top: 1; }
    #fbuttons Button { margin-left: 2; }
    """

    def __init__(
        self,
        *,
        stage_options: list[tuple[str, str]],
        stage_value: str,
        board_options: list[tuple[str, str]],
        board_value: str,
        salary_options: list[tuple[str, str]],
        salary_value: str,
        sort_options: list[tuple[str, str]],
        sort_value: str,
        hide_unlisted: bool,
        text: str,
    ) -> None:
        super().__init__()
        self._stage_options = stage_options
        self._stage_value = stage_value
        self._board_options = board_options
        self._board_value = board_value
        self._salary_options = salary_options
        self._salary_value = salary_value
        self._sort_options = sort_options
        self._sort_value = sort_value
        self._hide_unlisted = hide_unlisted
        self._text = text

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="filterbox"):
            yield Label("[bold]Filter & sort[/bold]")
            yield Label("Stage")
            yield Select(
                self._stage_options, value=self._stage_value, allow_blank=False, id="f_stage"
            )
            yield Label("Board")
            yield Select(
                self._board_options, value=self._board_value, allow_blank=False, id="f_board"
            )
            yield Label("Minimum salary")
            yield Select(
                self._salary_options, value=self._salary_value, allow_blank=False, id="f_salary"
            )
            yield Label("Sort by")
            yield Select(self._sort_options, value=self._sort_value, allow_blank=False, id="f_sort")
            yield Checkbox(
                "Hide jobs with no listed salary", value=self._hide_unlisted, id="f_unlisted"
            )
            yield Input(value=self._text, placeholder="text filter — title / company", id="f_text")
            with Horizontal(id="fbuttons"):
                yield Button("Apply", variant="primary", id="apply")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        super().on_mount()  # fade-in
        self.query_one("#f_stage", Select).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _submit(self) -> None:
        self.dismiss(
            {
                "stage": self.query_one("#f_stage", Select).value,
                "board": self.query_one("#f_board", Select).value,
                "min_salary": int(str(self.query_one("#f_salary", Select).value)),
                "sort_mode": self.query_one("#f_sort", Select).value,
                "hide_unlisted": self.query_one("#f_unlisted", Checkbox).value,
                "text": self.query_one("#f_text", Input).value.strip(),
            }
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()  # Enter in the text field applies


class ApplyScreen(_FadeModalScreen[bool | None]):
    """Confirm applying to a job. Dismisses ``True`` (real submit), ``False`` (dry run),
    or ``None`` (cancel). A real submit requires explicitly ticking the danger checkbox —
    never a single keypress."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    CSS = """
    ApplyScreen { align: center middle; }
    #applybox {
        width: 72; height: auto; padding: 1 2;
        border: thick $error; background: $surface;
    }
    #applybox Checkbox { margin: 1 0; }
    #warn { color: $warning; margin: 1 0; }
    #buttons { height: auto; align: right middle; }
    #buttons Button { margin-left: 2; }
    """

    def __init__(self, job: JobListing) -> None:
        super().__init__()
        self._job = job

    def compose(self) -> ComposeResult:
        with Vertical(id="applybox"):
            yield Label(f"[bold]Apply to {escape(self._job.title)}[/bold]")
            yield Static(f"{escape(self._job.company)} · {self._job.board.value}")
            yield Static("⚠  Opens a browser on your real LinkedIn account.", id="warn")
            yield Checkbox(
                "Send a REAL application — leave unchecked for a dry run "
                "(fills the form, never submits)",
                id="real",
            )
            with Horizontal(id="buttons"):
                yield Button("Apply", variant="primary", id="go")
                yield Button("Cancel", id="cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self.dismiss(self.query_one("#real", Checkbox).value)  # True = real, False = dry run
        else:
            self.dismiss(None)


class SetupScreen(_FadeModalScreen[str | None]):
    """Set the résumé path in-app. Dismisses with the entered path or ``None`` (cancel)."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    CSS = """
    SetupScreen { align: center middle; }
    #setupbox {
        width: 72; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #setupbox Input { margin: 1 0; }
    #hint { color: $text-muted; }
    #buttons { height: auto; align: right middle; }
    #buttons Button { margin-left: 2; }
    """

    def __init__(self, current: str = "") -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="setupbox"):
            yield Label("[bold]Set your résumé[/bold]")
            yield Input(
                value=self._current,
                placeholder="/path/to/your/cv.pdf  (or .docx / .txt)",
                id="path",
            )
            yield Static("Saved to config.toml so you only set it once.", id="hint")
            with Horizontal(id="buttons"):
                yield Button("Save", variant="primary", id="go")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        super().on_mount()  # fade-in
        self.query_one("#path", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _submit(self) -> None:
        path = self.query_one("#path", Input).value.strip()
        if not path:
            self.notify("Enter a résumé path.", severity="warning")
            return
        self.dismiss(path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()


class StyleGuideScreen(_FadeModalScreen[str | None]):
    """Set the style-guide path(s) in-app. Dismisses with the entered path(s) or ``None``.

    Accepts a single file or a comma-separated list of résumé/cover-letter examples
    whose writing style the LLM should mimic.
    """

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    CSS = """
    StyleGuideScreen { align: center middle; }
    #stylebox {
        width: 72; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #stylebox Input { margin: 1 0; }
    #hint { color: $text-muted; }
    #buttons { height: auto; align: right middle; }
    #buttons Button { margin-left: 2; }
    """

    def __init__(self, current: str = "") -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="stylebox"):
            yield Label("[bold]Set style guide[/bold]")
            yield Input(
                value=self._current,
                placeholder="/path/to/example.pdf  (comma-separated for multiple)",
                id="path",
            )
            yield Static(
                "Saved to config.toml. Tailor and cover-letter actions will mimic this style.",
                id="hint",
            )
            with Horizontal(id="buttons"):
                yield Button("Save", variant="primary", id="go")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        super().on_mount()  # fade-in
        self.query_one("#path", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _submit(self) -> None:
        # Empty string is a deliberate clear; None means Cancel.
        self.dismiss(self.query_one("#path", Input).value.strip())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()


class AtsScreen(_FadeModalScreen[None]):
    """Read-only ATS-compatibility result for the selected job's résumé. Esc / Close
    dismisses; the body scrolls when there are many warnings."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "close", "Close")]

    CSS = """
    AtsScreen { align: center middle; }
    #atsbox {
        width: 80; height: auto; max-height: 80%; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #atsbody { height: auto; max-height: 20; }
    #buttons { height: auto; align: right middle; }
    """

    def __init__(self, result: ATSCompatibilityResult, source: str = "") -> None:
        super().__init__()
        self._result = result
        self._source = source

    def compose(self) -> ComposeResult:
        r = self._result
        verdict = (
            "[green]✓ ATS-compatible[/green]"
            if r.is_compatible
            else "[red]✗ not ATS-compatible[/red]"
        )
        lines = [f"[bold]ATS check — {r.score:.0%}[/bold]   {verdict}"]
        if self._source:
            lines.append(f"[dim]{escape(self._source)}[/dim]")
        if r.warnings:
            lines += [
                "",
                "[bold]Warnings[/bold]",
                *(f"[yellow]⚠[/yellow] {escape(w)}" for w in r.warnings),
            ]
        if r.suggestions:
            lines += ["", "[bold]Suggestions[/bold]", *(f"→ {escape(s)}" for s in r.suggestions)]
        if not r.warnings and not r.suggestions:
            lines += ["", "[green]No issues found.[/green]"]
        with Vertical(id="atsbox"):
            yield VerticalScroll(Static("\n".join(lines)), id="atsbody")
            with Horizontal(id="buttons"):
                yield Button("Close", variant="primary", id="close")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class DocumentQualityScreen(_FadeModalScreen[None]):
    """Read-only generated document-quality result for the selected job."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "close", "Close")]

    CSS = """
    DocumentQualityScreen { align: center middle; }
    #qualitybox {
        width: 88; height: auto; max-height: 85%; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #qualitybody { height: auto; max-height: 24; }
    #buttons { height: auto; align: right middle; }
    """

    def __init__(self, result: DocumentQualityResult) -> None:
        super().__init__()
        self._result = result

    def compose(self) -> ComposeResult:
        result = self._result
        verdict = "[green]✓ passed[/green]" if result.passed else "[red]✗ failed[/red]"
        mode_text = (
            "Packet quality check passed"
            if result.packet is not None and result.passed
            else "Packet quality check failed"
            if result.packet is not None
            else "Limited smoke check passed"
            if result.passed
            else "Limited smoke check failed"
        )
        lines = [
            f"[bold]Document quality[/bold]   {verdict}",
            f"[bold]{mode_text}[/bold]",
            f"[dim]Source: {escape(result.source)}[/dim]",
            f"[dim]Keywords: {escape(result.keyword_source)}[/dim]",
            f"[dim]Matched skills: {result.matched_skill_count}[/dim]",
        ]
        if result.artifact_paths:
            lines.append("")
            lines.append("[bold]Artifacts[/bold]")
            for path in result.artifact_paths:
                mtime = result.artifact_mtimes.get(path, "unknown mtime")
                lines.append(f"  {escape(path)}  [dim]mtime={escape(mtime)}[/dim]")
        if result.packet is not None:
            lines.append("")
            lines.append(f"[bold]Packet[/bold]  overall={result.packet.overall:.2f}/4")
            for name, score in result.packet.dimensions.items():
                lines.append(f"  {escape(name)}: {score:.2f}/4")

        for report in result.reports:
            lines.append("")
            lines.append(
                f"[bold]{escape(report.kind.replace('_', ' ').title())}[/bold]  "
                f"score={report.score}"
            )
            for item in report.failures:
                lines.append(f"  [red]failure:[/red] {escape(item)}")
            for item in report.warnings:
                lines.append(f"  [yellow]warning:[/yellow] {escape(item)}")
            if not report.failures and not report.warnings:
                lines.append("  [green]No issues found.[/green]")

        if result.packet is not None:
            for item in result.packet.failures:
                lines.append(f"  [red]packet failure:[/red] {escape(item)}")
            for item in result.packet.warnings:
                lines.append(f"  [yellow]packet warning:[/yellow] {escape(item)}")

        with Vertical(id="qualitybox"):
            yield VerticalScroll(Static("\n".join(lines)), id="qualitybody")
            with Horizontal(id="buttons"):
                yield Button("Close", variant="primary", id="close")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class HelpScreen(_FadeModalScreen[None]):
    """Read-only key reference, grouped by account-safety tier so the safe/local keys read
    as distinct from the account-touching ones. Esc / Close dismisses."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "close", "Close")]

    CSS = """
    HelpScreen { align: center middle; }
    #helpbox {
        width: 72; height: auto; max-height: 90%; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #helpbody { height: auto; max-height: 24; }
    #buttons { height: auto; align: right middle; }
    """

    # One source of truth for the key reference (the BINDINGS table drives behaviour; this
    # explains it). Grouped by what each key TOUCHES — local/LLM vs the real account.
    _HELP = "\n".join(
        [
            "[bold]job-applicator — keys[/bold]",
            "",
            "[bold]Navigate[/bold]",
            "  ↑ ↓ · j k   move selection",
            "  ] \\[         scroll the posting / detail pane",  # \\[ → literal '[' (not a tag)
            "  /           quick text filter (title/company) · Esc clears all filters",
            "  f           Filter & sort panel — stage · board · salary · sort · text",
            "  tabs        click a stage tab at the top (or use the f panel)",
            "  [dim]quick keys: b board · m min$ · u unlisted · S sort (all in the f panel)[/dim]",
            "  r           refresh from the store",
            "  q           quit",
            "",
            "[bold]Act on the selected job[/bold]  [dim](LLM + local files — account-safe)[/dim]",
            "  t           tailor résumé",
            "  T           tailor résumé PDF",
            "  c           cover letter",
            "  C           cover letter PDF",
            "  p           open generated PDF (résumé, or cover letter as fallback)",
            "  A           ATS-compatibility check",
            "  D           document-quality check",
            "  e           set résumé path",
            "  g           set style-guide path",
            "",
            "[bold]Links[/bold]",
            "  o           open the posting in your browser",
            "  y           copy the posting URL",
            "  click       open a generated résumé / cover-letter artifact",
            "",
            "[bold]Account-touching[/bold]  "
            "[yellow](opens a real browser on your account)[/yellow]",
            "  s           search a board — explicit confirm before any browser opens",
            "  a           apply — [yellow]dry-run by default[/yellow]; a real submit needs the "
            "danger checkbox",
            "",
            "[dim]search/apply never auto-login; apply respects the daily cap.[/dim]",
        ]
    )

    def compose(self) -> ComposeResult:
        with Vertical(id="helpbox"):
            yield VerticalScroll(Static(self._HELP), id="helpbody")
            with Horizontal(id="buttons"):
                yield Button("Close", variant="primary", id="close")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)
