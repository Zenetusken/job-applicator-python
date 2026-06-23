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
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Static

from job_applicator.models import JobBoard
from job_applicator.scrapers.base import SearchParams

if TYPE_CHECKING:
    from job_applicator.models import ATSCompatibilityResult, JobListing

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


class SearchScreen(_FadeModalScreen[SearchParams | None]):
    """A search form. Dismisses with ``SearchParams`` on submit (authorizing the
    account-touching scrape) or ``None`` on cancel/Esc."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    CSS = """
    SearchScreen { align: center middle; }
    #searchbox {
        width: 68; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #searchbox Input, #searchbox Checkbox { margin: 1 0; }
    #warn { color: $warning; margin: 1 0; }
    #buttons { height: auto; align: right middle; }
    #buttons Button { margin-left: 2; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="searchbox"):
            yield Label("[bold]Search jobs[/bold]")
            yield Input(placeholder="query — e.g. senior python engineer", id="q")
            yield Input(placeholder="location (optional)", id="loc")
            yield Input(value="25", placeholder="max results (1-50)", id="maxn", type="integer")
            yield Checkbox("Remote only", id="remote")
            yield Static("⚠  Opens a browser on your real LinkedIn account.", id="warn")
            with Horizontal(id="buttons"):
                yield Button("Search", variant="primary", id="go")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        super().on_mount()  # fade-in
        self.query_one("#q", Input).focus()

    _MAX_RESULTS_CAP = 50

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _max_results(self) -> int:
        """How many results to scrape — clamped to 1…cap. Empty/invalid falls back to the
        default (the Input is integer-only, so non-numeric shouldn't reach here)."""
        raw = self.query_one("#maxn", Input).value.strip()
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
        self.dismiss(
            SearchParams(
                query=query,
                location=self.query_one("#loc", Input).value.strip(),
                remote_only=self.query_one("#remote", Checkbox).value,
                max_results=self._max_results(),
                board=JobBoard.LINKEDIN,
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in any field submits the form (same as the Search button).
        self._submit()


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
            "  /           filter title/company · Esc clears",
            "  r           refresh from the store",
            "  q           quit",
            "",
            "[bold]Act on the selected job[/bold]  [dim](LLM + local files — account-safe)[/dim]",
            "  t           tailor résumé",
            "  c           cover letter",
            "  A           ATS-compatibility check",
            "  e           set résumé path",
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
