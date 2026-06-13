"""Tests for diff rendering utility."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from job_applicator.utils.diff import render_diff


class TestRenderDiff:
    def _get_output(self, original: str, tailored: str, max_lines: int = 0) -> str:
        """Helper to capture render_diff output."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, no_color=True)
        render_diff(console, original, tailored, max_lines=max_lines)
        return buf.getvalue()

    def test_no_differences(self):
        output = self._get_output("same text", "same text")
        assert "No differences found" in output

    def test_additions_shown(self):
        output = self._get_output("line one", "line one\nline two")
        assert "line two" in output
        assert "+" in output

    def test_removals_shown(self):
        output = self._get_output("line one\nline two", "line one")
        assert "line two" in output
        assert "-" in output

    def test_max_lines_truncates(self):
        original = "\n".join(f"line {i}" for i in range(20))
        tailored = "\n".join(f"line {i} modified" for i in range(20))
        output = self._get_output(original, tailored, max_lines=5)
        assert "more lines" in output

    def test_empty_original(self):
        output = self._get_output("", "new content")
        assert "new content" in output

    def test_empty_tailored(self):
        output = self._get_output("old content", "")
        assert "old content" in output
