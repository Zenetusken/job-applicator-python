"""Unit tests for the construction factories."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import typer

import job_applicator.factories as factories


def test_make_scraper_unknown_site_message_to_err_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown --site message must go to stderr (err_console), not stdout — so it can't
    pollute a --json consumer's stdout (CLAUDE.md stdout contract)."""
    printed: list[object] = []
    monkeypatch.setattr(
        factories, "err_console", MagicMock(print=lambda *a, **k: printed.append(a))
    )
    with pytest.raises(typer.Exit):
        factories._make_scraper("bogus", None, None)  # type: ignore[arg-type]
    assert printed  # routed to err_console (stderr)


def test_make_applicator_unknown_site_message_to_err_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(
        factories, "err_console", MagicMock(print=lambda *a, **k: printed.append(a))
    )
    with pytest.raises(typer.Exit):
        factories._make_applicator("bogus", None, None)  # type: ignore[arg-type]
    assert printed
