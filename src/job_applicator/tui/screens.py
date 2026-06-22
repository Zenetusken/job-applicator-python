"""Modal screens for the TUI.

Account-safety note: ``SearchScreen`` only *collects* parameters. Submitting it — the
deliberate act, taken with the "opens a browser on your real account" warning visible —
is what authorizes the scrape; the screen itself touches nothing.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Static

from job_applicator.models import JobBoard
from job_applicator.scrapers.base import SearchParams


class SearchScreen(ModalScreen[SearchParams | None]):
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
            yield Checkbox("Remote only", id="remote")
            yield Static("⚠  Opens a browser on your real LinkedIn account.", id="warn")
            with Horizontal(id="buttons"):
                yield Button("Search", variant="primary", id="go")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#q", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

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
