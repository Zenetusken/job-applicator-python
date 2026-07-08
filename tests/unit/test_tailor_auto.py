"""Tests for shared unattended tailoring guardrails."""

from __future__ import annotations

from job_applicator.workflows.tailor_auto import source_only_instructions


def test_source_only_instructions_preserve_source_owned_terms_in_translation() -> None:
    instructions = source_only_instructions()

    assert "job titles" in instructions
    assert "course names" in instructions
    assert "skill/tool names" in instructions
    assert "one long translated prose claim" in instructions
