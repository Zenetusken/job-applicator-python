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
    yes: bool = False,
    as_json: bool = False,
    min_score: str = "0",
    match_score: float = 0.8,
    staleness: list[str] | None = None,
    ordering: list[str] | None = None,
    output_format: str | None = None,
    template: str | None = None,
    category: str | None = None,
):
    """Drive the `tailor` command through its interactive loop.

    Returns (CliRunner result, engine mock, cover-letter-workflow mock).
    """
    import job_applicator.cli as cli

    engine = MagicMock()
    engine.tailor_verified = AsyncMock(return_value=_tailored("INITIAL"))
    engine.refine = AsyncMock(return_value=_tailored("REFINED"))

    audit = MagicMock(
        entries=[],
        warnings=[],
        staleness_issues=staleness or [],
        ordering_issues=ordering or [],
        is_stale=bool(staleness),  # mirrors resume_tailor.py: is_stale = bool(staleness_issues)
        earliest_date="2020",
        latest_date="2023",
    )
    validator = MagicMock()
    validator.audit.return_value = audit
    loader = MagicMock()
    loader.load.return_value = ResumeData(raw_text="John Doe\njohn@example.com\nPython, SQL")
    tone = MagicMock(primary="professional", confidence=0.9)
    cl_workflow = AsyncMock(return_value=Path("output/cover.txt"))
    matcher = MagicMock()
    matcher.match_resume_to_job.return_value = MagicMock(score=match_score)

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
        patch("job_applicator.embeddings.matching.JobMatcher", return_value=matcher),
    ):
        args = ["tailor", "-t", "Dev", "-c", "Acme", "--resume", "r.pdf", "--min-score", min_score]
        if yes:
            args.append("--yes")
        if as_json:
            args.append("--json")
        if output_format:
            args.extend(["--format", output_format])
        if template:
            args.extend(["--template", template])
        if category:
            args.extend(["--category", category])
        result = CliRunner().invoke(
            cli.app,
            args,
            env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
        )
    return result, engine, cl_workflow


def test_tailor_quit_discards_without_saving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[Q] quits after the initial tailor: no refine, no files written."""
    result, engine, cl = _drive(monkeypatch, tmp_path, ["Q"])
    assert result.exit_code == 0, result.output
    engine.tailor_verified.assert_awaited_once()
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


def test_tailor_pdf_format_writes_pdf_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``tailor --format pdf --template classic`` renders a PDF artifact and sidecar."""
    from job_applicator.documents.pdf_renderer import PDFRenderer

    fake_pdf = tmp_path / "tailored_Acme_Dev_20260625_120000_000000_classic.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with patch.object(
        PDFRenderer, "render_resume", new=AsyncMock(return_value=fake_pdf)
    ) as mock_render:
        result, _engine, _cl = _drive(
            monkeypatch,
            tmp_path,
            [],
            yes=True,
            output_format="pdf",
            template="classic",
            category="cybersecurity",
        )

    assert result.exit_code == 0, result.output
    pdfs = list(tmp_path.glob("tailored_*.pdf"))
    assert len(pdfs) == 1
    assert pdfs[0].name == fake_pdf.name
    meta = next(tmp_path.glob("tailored_*.meta.json")).read_text(encoding="utf-8")
    assert str(fake_pdf) in meta
    mock_render.assert_awaited_once()
    assert mock_render.await_args.kwargs["template"] == "classic"
    assert mock_render.await_args.kwargs["category"] == "cybersecurity"


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


def test_tailor_yes_is_non_interactive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`--yes` runs the whole flow with NO prompts: auto-accept the tailored résumé AND
    skip the (interactive) cover-letter offer. Regression guard — the flag was not threaded
    into _tailor_workflow, so `tailor --yes` blocked on the action menu (hung in CI/non-tty).
    """
    import job_applicator.cli as cli

    # inputs=[] → if ANY prompt fires it raises StopIteration and the command fails.
    result, engine, cl = _drive(monkeypatch, tmp_path, [], yes=True)
    assert result.exit_code == 0, result.output
    cli.console.input.assert_not_called()  # type: ignore[attr-defined]  # zero prompts
    txts = list(tmp_path.glob("tailored_*.txt"))
    assert len(txts) == 1 and txts[0].read_text(encoding="utf-8") == "INITIAL"  # accepted + saved
    assert list(tmp_path.glob("tailored_*.meta.json"))  # meta written
    cl.assert_not_awaited()  # cover-letter offer skipped, not dragged into a 2nd interactive loop
    engine.tailor_verified.assert_awaited_once()


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


def test_tailor_stale_but_ordered_cv_triggers_confirm_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale-but-correctly-ordered CV (staleness, NO ordering issues) must still trigger the
    'Proceed anyway?' confirm gate. Regression guard — the gate was nested under
    `if audit.ordering_issues:`, so a stale-but-ordered CV silently skipped it and tailored anyway.
    """
    # Answer the confirm prompt "n" → abort BEFORE tailoring.
    result, engine, _cl = _drive(
        monkeypatch, tmp_path, ["n"], staleness=["Most recent role ended 2019 (5+ years ago)"]
    )
    assert result.exit_code == 0, result.output  # typer.Exit(0) user-abort
    assert "Aborted. Please update your CV." in result.output
    engine.tailor_verified.assert_not_awaited()  # gate fired + aborted before any tailoring


def test_tailor_clean_dates_show_coherent_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A CV with NO staleness/ordering issues shows the '✓ Dates look coherent' confirmation.
    Regression guard — that else-branch was dead (nested under `if ordering_issues:` + an
    always-true `or`), so the message never printed for a clean CV.
    """
    result, _engine, _cl = _drive(monkeypatch, tmp_path, ["A", "N"])  # default audit = all clean
    assert result.exit_code == 0, result.output
    assert "Dates look coherent and current." in result.output
    assert len(list(tmp_path.glob("tailored_*.txt"))) == 1  # tailoring proceeded normally


def test_tailor_yes_auto_proceeds_through_stale_date_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--yes` on a stale CV fires the date gate and auto-proceeds (no prompt), then tailors —
    covers the gate's `yes` branch (the other date tests exercise the else / interactive paths).
    """
    result, engine, _cl = _drive(
        monkeypatch, tmp_path, [], yes=True, staleness=["Most recent role ended 2019"]
    )
    assert result.exit_code == 0, result.output
    assert "--yes flag set, proceeding automatically." in result.output
    engine.tailor_verified.assert_awaited_once()
    assert len(list(tmp_path.glob("tailored_*.txt"))) == 1  # proceeded past the gate + tailored


def test_tailor_ordering_only_cv_fires_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An ordering-issue-only CV (no staleness) fires the gate (closes the 4-state matrix:
    clean / stale-only / ordering-only / both — the last two share the same boolean branch)."""
    result, _engine, _cl = _drive(
        monkeypatch, tmp_path, ["y", "A", "N"], ordering=["Roles listed out of chronological order"]
    )
    assert result.exit_code == 0, result.output
    assert "Ordering Issues" in result.output
    assert "Dates look coherent" not in result.output  # gate fired, not the else
    assert len(list(tmp_path.glob("tailored_*.txt"))) == 1  # 'y' → proceeded + tailored


def test_tailor_json_emits_tailored_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`tailor --json` is non-interactive (implies --yes) and emits the TailoredResume as JSON
    on stdout; the Rich preview/diff/date-audit all go to stderr (so `tailor --json | jq` works).
    inputs=[] → any prompt would StopIteration; --json must auto-accept without prompting."""
    import json

    result, engine, cl = _drive(monkeypatch, tmp_path, [], as_json=True)
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)  # raises if any Rich output leaked onto stdout
    assert parsed["tailored_text"] == "INITIAL"  # the auto-accepted version
    assert parsed["job_title"] == "Dev" and parsed["job_company"] == "Acme"
    assert parsed["output_path"]  # the saved artifact path is carried in the JSON
    assert "Tailored Resume Preview" not in result.stdout  # Rich preview lives on stderr
    engine.tailor_verified.assert_awaited_once()
    cl.assert_not_awaited()  # cover-letter offer skipped (non-interactive)


def test_tailor_json_min_score_abort_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`tailor --json` with a sub-threshold match score exits NON-ZERO (not a silent
    empty-stdout success). Under --json the abort message is on stderr, so the exit code is the
    only signal a `| jq` pipeline gets — Exit(0) there would look like a working-but-empty run."""
    result, engine, _cl = _drive(
        monkeypatch, tmp_path, [], as_json=True, min_score="0.99", match_score=0.10
    )
    assert result.exit_code != 0  # sub-threshold → non-zero (the gate-2a finding)
    assert result.stdout.strip() == ""  # nothing tailored → no JSON emitted
    engine.tailor_verified.assert_not_awaited()  # aborted before tailoring
