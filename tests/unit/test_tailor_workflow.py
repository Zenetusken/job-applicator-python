"""Characterization tests for the interactive `tailor` command loop.

These pin the CURRENT behavior of the tailor interactive loop ([A]/[R]/[I]/[D]/[V]/[S]/[Q])
BEFORE it is extracted to a workflow module, so any logic/behavioral drift in the
extraction is caught. They drive the real `tailor` command with a mocked ResumeTailor +
scripted console input, mocking only the heavy setup deps.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from job_applicator.models import ResumeData, TailoredResume


def _tailored(text: str = "TAILORED RESUME", **kw: object) -> TailoredResume:
    return TailoredResume(
        original_path="r.pdf",
        tailored_text=text,
        job_title="Dev",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.9,
        skill_score=0.7,
        changes_summary="reworded summary",
        **kw,  # type: ignore[arg-type]
    )


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inputs: list[str],
    *,
    sections: list[object] | None = None,
):
    """Drive the `tailor` command through its interactive loop.

    Returns (CliRunner result, engine mock, cover-letter-workflow mock).
    """
    import job_applicator.cli as cli

    engine = MagicMock()
    engine.tailor = AsyncMock(return_value=_tailored("INITIAL"))
    engine.refine = AsyncMock(return_value=_tailored("REFINED"))

    audit = MagicMock(
        entries=[],
        warnings=[],
        staleness_issues=[],
        ordering_issues=[],
        is_stale=False,
        earliest_date="2020",
        latest_date="2023",
    )
    validator = MagicMock()
    validator.audit.return_value = audit
    loader = MagicMock()
    loader.load.return_value = ResumeData(raw_text="John Doe\njohn@example.com\nPython, SQL")
    tone = MagicMock(primary="professional", confidence=0.9)
    cl_workflow = AsyncMock(return_value=Path("output/cover.txt"))

    monkeypatch.setattr(cli.console, "input", MagicMock(side_effect=inputs))

    with (
        patch("job_applicator.documents.resume.ResumeLoader", return_value=loader),
        patch("job_applicator.documents.resume_tailor.ResumeTailor", return_value=engine),
        patch(
            "job_applicator.documents.resume_tailor.ResumeDateValidator",
            return_value=validator,
        ),
        patch(
            "job_applicator.documents.resume_tailor.parse_sections",
            return_value=(sections if sections is not None else []),
        ),
        patch.object(cli, "_detect_tone", return_value=tone),
        patch("job_applicator.workflows.tailor._cover_letter_workflow", cl_workflow),
    ):
        result = CliRunner().invoke(
            cli.app,
            ["tailor", "-t", "Dev", "-c", "Acme", "--resume", "r.pdf", "--min-score", "0"],
            env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
        )
    return result, engine, cl_workflow


def test_tailor_quit_discards_without_saving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[Q] quits after the initial tailor: no refine, no files written."""
    result, engine, cl = _drive(monkeypatch, tmp_path, ["Q"])
    assert result.exit_code == 0, result.output
    engine.tailor.assert_awaited_once()
    engine.refine.assert_not_awaited()
    cl.assert_not_awaited()
    assert not list(tmp_path.glob("tailored_*.txt"))


def test_tailor_accept_saves_without_cover_letter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[A] then 'N': saves the tailored résumé + meta.json, no cover letter."""
    result, engine, cl = _drive(monkeypatch, tmp_path, ["A", "N"])
    assert result.exit_code == 0, result.output
    txts = list(tmp_path.glob("tailored_*.txt"))
    assert len(txts) == 1
    assert txts[0].read_text(encoding="utf-8") == "INITIAL"
    assert len(list(tmp_path.glob("tailored_*.meta.json"))) == 1
    cl.assert_not_awaited()
    engine.refine.assert_not_awaited()


def test_tailor_accept_then_cover_letter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """[A] then 'Y': saves, runs the cover-letter workflow, records its path in meta."""
    result, _engine, cl = _drive(monkeypatch, tmp_path, ["A", "Y"])
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("tailored_*.txt"))) == 1
    cl.assert_awaited_once()
    meta = next(tmp_path.glob("tailored_*.meta.json")).read_text(encoding="utf-8")
    assert "output/cover.txt" in meta  # cover_letter_path recorded


def test_tailor_retry_refines_then_quit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """[R] refines via the engine (empty instructions), then [Q] discards."""
    result, engine, _cl = _drive(monkeypatch, tmp_path, ["R", "Q"])
    assert result.exit_code == 0, result.output
    engine.refine.assert_awaited_once()
    assert not list(tmp_path.glob("tailored_*.txt"))


def test_tailor_input_refines_with_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[I] refines with the user's instructions threaded into engine.refine."""
    result, engine, _ = _drive(monkeypatch, tmp_path, ["I", "emphasize customer service", "Q"])
    assert result.exit_code == 0, result.output
    engine.refine.assert_awaited_once()
    assert "emphasize customer service" in engine.refine.await_args.args


def test_tailor_section_edit_refines_target_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[S] parses sections, targets one, and refines it with section-scoped instructions."""
    sections = [
        SimpleNamespace(name="Summary", text="summary text"),
        SimpleNamespace(name="Experience", text="experience text"),
    ]
    result, engine, _ = _drive(
        monkeypatch, tmp_path, ["S", "1", "fix the summary", "Q"], sections=sections
    )
    assert result.exit_code == 0, result.output
    engine.refine.assert_awaited_once()
    instructions = engine.refine.await_args.args[2]
    assert "fix the summary" in instructions
    assert "Summary" in instructions  # section-scoped


def test_tailor_diff_then_quit_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[D] is display-only: no refine, no save, loop continues to [Q]."""
    result, engine, _ = _drive(monkeypatch, tmp_path, ["D", "Q"])
    assert result.exit_code == 0, result.output
    engine.refine.assert_not_awaited()
    assert not list(tmp_path.glob("tailored_*.txt"))


def test_tailor_history_with_one_attempt_then_quit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[V] with a single attempt reports none-yet and continues; [Q] then discards."""
    result, engine, _ = _drive(monkeypatch, tmp_path, ["V", "Q"])
    assert result.exit_code == 0, result.output
    assert "No previous attempts yet" in result.output
    engine.refine.assert_not_awaited()


def test_tailor_invalid_choice_then_quit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unrecognized choice is rejected and the loop re-prompts; [Q] then discards."""
    result, engine, _ = _drive(monkeypatch, tmp_path, ["X", "Q"])
    assert result.exit_code == 0, result.output
    assert "Invalid choice" in result.output
    engine.refine.assert_not_awaited()
    assert not list(tmp_path.glob("tailored_*.txt"))
