"""Unit tests for the job-description formatter (tui/textfmt.py)."""

from __future__ import annotations

from job_applicator.tui.textfmt import format_job_description


def test_empty_is_empty() -> None:
    assert format_job_description("") == ""
    assert format_job_description("   \n  \n") == ""


def test_reflows_lowercase_continuation() -> None:
    """A hard-wrapped sentence (continuation starts lowercase) is rejoined into one line."""
    raw = "Design secure architectures that are\nrecoverable and sized appropriately"
    out = format_job_description(raw)
    assert out == "Design secure architectures that are recoverable and sized appropriately"


def test_does_not_merge_separate_items() -> None:
    """Two list items (each starts uppercase) stay on their own lines — not merged to a wall."""
    raw = "Design complex architectures\nWork directly with customers\nOwn solutions end to end"
    out = format_job_description(raw)
    assert out.split("\n") == [
        "Design complex architectures",
        "Work directly with customers",
        "Own solutions end to end",
    ]


def test_collapses_blank_runs() -> None:
    raw = "First paragraph.\n\n\n\nSecond paragraph."
    assert format_job_description(raw) == "First paragraph.\n\nSecond paragraph."


def test_bolds_known_headers() -> None:
    out = format_job_description("Requirements:\n5+ years Python")
    assert out.startswith("[bold]Requirements[/bold]")
    assert "5+ years Python" in out


def test_bolds_titlecase_subheaders() -> None:
    """A short, all-content-capitalized line is a section header even when not in the known list
    (the apostrophe + merged-subheader case the known list misses)."""
    assert "[bold]Incident Response & Ownership[/bold]" in format_job_description(
        "Incident Response & Ownership\nParticipate in the on-call rotation"
    )
    assert "[bold]What You'll Do[/bold]" in format_job_description("What You'll Do\nbuild things")


def test_does_not_bold_a_sentence_starting_with_a_header_word() -> None:
    """A long line that merely STARTS with a header word ('Experience in a …') is prose, not a
    header — guards against the measured false positive."""
    line = "Experience in a control function (having a control mind-set) is required"
    out = format_job_description(line)
    assert "[bold]" not in out


def test_escapes_markup_in_the_text() -> None:
    """Square brackets in the posting must be escaped so they don't inject Rich markup."""
    out = format_job_description("Pay is [negotiable] and [red]urgent[/red]")
    assert "\\[negotiable]" in out  # bracket escaped, not interpreted as markup
    assert "\\[red]urgent\\[/red]" in out
