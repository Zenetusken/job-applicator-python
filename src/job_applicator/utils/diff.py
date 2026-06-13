"""Diff rendering utilities for terminal output."""

from __future__ import annotations

import difflib

from rich.console import Console


def render_diff(console: Console, original: str, tailored: str, max_lines: int = 30) -> None:
    """Render a color-coded diff between original and tailored resume.

    Args:
        console: Rich console instance
        original: Original resume text
        tailored: Tailored resume text
        max_lines: Maximum diff lines to show (0 = unlimited)
    """
    original_lines = original.splitlines(keepends=True)
    tailored_lines = tailored.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            original_lines,
            tailored_lines,
            fromfile="original",
            tofile="tailored",
            lineterm="",
        )
    )

    if not diff:
        console.print("[dim]No differences found.[/dim]")
        return

    shown = 0
    for line in diff:
        if max_lines and shown >= max_lines:
            console.print(
                f"[dim]... {len(diff) - shown} more lines (use [D] to see full diff)[/dim]"
            )
            break
        if line.startswith("+++") or line.startswith("---"):
            console.print(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{line}[/cyan]")
        elif line.startswith("+"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{line}[/red]")
        else:
            console.print(f"[dim]{line}[/dim]")
        shown += 1
