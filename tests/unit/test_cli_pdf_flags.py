"""Tests for the PDF rendering CLI flags introduced in Task 13.

These are intentionally minimal smoke tests: they verify that ``--format``,
``--template``, and ``--category`` are accepted by the four affected commands
and are passed through to the workflow / artifact helpers. They mock the PDF
helpers and heavy LLM/browser dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from job_applicator.models import JobListing, ResumeData, TailoredResume


def _tailored(text: str = "TAILORED RESUME", **kw: object) -> TailoredResume:
    defaults = {
        "original_path": "r.pdf",
        "tailored_text": text,
        "job_title": "Dev",
        "job_company": "Acme",
        "match_score": 0.8,
        "semantic_score": 0.9,
        "skill_score": 0.7,
        "changes_summary": "reworded summary",
    }
    defaults.update(kw)
    return TailoredResume(**defaults)  # type: ignore[arg-type]


def _patch_tailor_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    """Mock the heavy dependencies of the ``tailor`` command."""
    import job_applicator.cli as cli

    engine = MagicMock()
    engine.tailor_verified = AsyncMock(return_value=_tailored("INITIAL"))
    engine.refine_verified = AsyncMock(return_value=_tailored("REFINED"))

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
    matcher = MagicMock()
    matcher.match_resume_to_job = AsyncMock(return_value=MagicMock(score=0.8))

    monkeypatch.setattr(cli.console, "input", MagicMock(side_effect=["A", "N"]))

    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    monkeypatch.setattr(
        "job_applicator.documents.resume_tailor.ResumeTailor", lambda *a, **k: engine
    )
    monkeypatch.setattr(
        "job_applicator.documents.resume_tailor.ResumeDateValidator",
        lambda: validator,
    )
    monkeypatch.setattr(cli, "_detect_tone", lambda job: tone)
    monkeypatch.setattr(cli, "_run_ats_preflight", lambda r: MagicMock(score=0.8))
    monkeypatch.setattr("job_applicator.embeddings.matching.JobMatcher", lambda *a, **k: matcher)
    return engine


def test_tailor_pdf_flags_passed_to_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``tailor --format pdf --template classic --category cybersecurity`` forwards the
    flags to ``_tailor_workflow``.
    """
    import job_applicator.cli as cli

    _patch_tailor_stack(monkeypatch, tmp_path)
    workflow = AsyncMock()
    monkeypatch.setattr(cli, "_tailor_workflow", workflow)

    result = CliRunner().invoke(
        cli.app,
        [
            "tailor",
            "-t",
            "Dev",
            "-c",
            "Acme",
            "--resume",
            "r.pdf",
            "--format",
            "pdf",
            "--template",
            "classic",
            "--category",
            "cybersecurity",
        ],
        env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0, result.output
    workflow.assert_awaited_once()
    call_kwargs = workflow.await_args.kwargs
    assert call_kwargs["output_format"].value == "pdf"
    assert call_kwargs["resume_template"] == "classic"
    assert call_kwargs["cover_letter_template"] == "classic"
    assert call_kwargs["category"] == "cybersecurity"


def test_tailor_format_txt_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``tailor`` without ``--format`` defaults to ``txt`` and the configured templates."""
    import job_applicator.cli as cli
    from job_applicator.config import AppSettings

    _patch_tailor_stack(monkeypatch, tmp_path)
    workflow = AsyncMock()
    monkeypatch.setattr(cli, "_tailor_workflow", workflow)

    result = CliRunner().invoke(
        cli.app,
        ["tailor", "-t", "Dev", "-c", "Acme", "--resume", "r.pdf"],
        env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0, result.output
    call_kwargs = workflow.await_args.kwargs
    assert call_kwargs["output_format"].value == "txt"
    assert call_kwargs["resume_template"] == AppSettings().output.resume_template
    assert call_kwargs["cover_letter_template"] == AppSettings().output.cover_letter_template
    assert call_kwargs["category"] is None


def test_generate_cover_letter_pdf_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``generate-cover-letter --format pdf --template classic --category cybersecurity``
    calls the PDF artifact helper with the right template and category.
    """
    import job_applicator.cli as cli

    loader = MagicMock()
    loader.load.return_value = ResumeData(raw_text="John Doe\njohn@example.com\nPython")
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    monkeypatch.setattr(cli, "_run_ats_preflight", lambda r: MagicMock(score=0.8))
    monkeypatch.setattr(
        cli, "_detect_tone", lambda job: MagicMock(primary="professional", confidence=0.9)
    )

    fake_pdf = tmp_path / "cover_letter_Acme_Dev_20260625_000000.pdf"
    fake_pdf.write_bytes(b"pdf")

    with (
        patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls,
        patch(
            "job_applicator.documents.pdf_renderer.PDFRenderer.render_cover_letter",
            new=AsyncMock(return_value=fake_pdf),
        ) as mock_render,
    ):
        mock_gen = mock_gen_cls.return_value
        mock_gen.generate_verified = AsyncMock(return_value="Dear Hiring Manager,")

        result = CliRunner().invoke(
            cli.app,
            [
                "generate-cover-letter",
                "--job-title",
                "Dev",
                "--company",
                "Acme",
                "--resume",
                "r.pdf",
                "--format",
                "pdf",
                "--template",
                "classic",
                "--category",
                "cybersecurity",
            ],
            env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
        )

    assert result.exit_code == 0, result.output
    mock_render.assert_awaited_once()
    (rendered_result,) = mock_render.await_args.args
    assert rendered_result.job_title == "Dev"
    assert rendered_result.job_company == "Acme"
    assert mock_render.await_args.kwargs["template"] == "classic"
    assert mock_render.await_args.kwargs["category"] == "cybersecurity"


def test_generate_cover_letter_default_txt_writes_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``generate-cover-letter`` (default ``--format txt``) writes a text artifact and
    reports its path.
    """
    import job_applicator.cli as cli

    loader = MagicMock()
    loader.load.return_value = ResumeData(raw_text="John Doe\njohn@example.com\nPython")
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    monkeypatch.setattr(cli, "_run_ats_preflight", lambda r: MagicMock(score=0.8))
    monkeypatch.setattr(
        cli, "_detect_tone", lambda job: MagicMock(primary="professional", confidence=0.9)
    )

    with patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls:
        mock_gen = mock_gen_cls.return_value
        mock_gen.generate_verified = AsyncMock(return_value="Dear Hiring Manager,")

        result = CliRunner().invoke(
            cli.app,
            [
                "generate-cover-letter",
                "--job-title",
                "Dev",
                "--company",
                "Acme",
                "--resume",
                "r.pdf",
                "--json",
            ],
            env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["output_path"]
    assert Path(payload["output_path"]).exists()
    assert Path(payload["output_path"]).suffix == ".txt"
    assert "pdf_path" not in payload


def test_generate_cover_letter_both_writes_txt_and_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``generate-cover-letter --format both`` writes a text file, a PDF, and updates the
    text sidecar with ``pdf_path``.
    """
    import job_applicator.cli as cli

    loader = MagicMock()
    loader.load.return_value = ResumeData(raw_text="John Doe\njohn@example.com\nPython")
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    monkeypatch.setattr(cli, "_run_ats_preflight", lambda r: MagicMock(score=0.8))
    monkeypatch.setattr(
        cli, "_detect_tone", lambda job: MagicMock(primary="professional", confidence=0.9)
    )

    fake_pdf = tmp_path / "cover_letter_Acme_Dev_20260625_000000.pdf"
    fake_pdf.write_bytes(b"pdf")

    with (
        patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls,
        patch(
            "job_applicator.documents.pdf_renderer.PDFRenderer.render_cover_letter",
            new=AsyncMock(return_value=fake_pdf),
        ),
    ):
        mock_gen = mock_gen_cls.return_value
        mock_gen.generate_verified = AsyncMock(return_value="Dear Hiring Manager,")

        result = CliRunner().invoke(
            cli.app,
            [
                "generate-cover-letter",
                "--job-title",
                "Dev",
                "--company",
                "Acme",
                "--resume",
                "r.pdf",
                "--format",
                "both",
                "--json",
            ],
            env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["output_path"]
    assert payload["pdf_path"] == str(fake_pdf)
    meta_path = Path(payload["output_path"]).with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["pdf_path"] == str(fake_pdf)


def test_batch_pdf_flags_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``batch --format pdf --template classic --category cybersecurity`` is accepted and
    the PDF artifact helper is invoked for the résumé.
    """
    import job_applicator.cli as cli
    from job_applicator.embeddings.matching import MatchResult
    from job_applicator.models import JobBoard, JobListing

    loader = MagicMock()
    loader.load.return_value = ResumeData(raw_text="John Doe\njohn@example.com\nPython")
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    monkeypatch.setattr(cli, "_run_ats_preflight", lambda r: MagicMock(score=0.8))
    monkeypatch.setattr(
        cli, "_run_ats_post_tailor", lambda original, tailored: MagicMock(score=1.0)
    )

    job = JobListing(
        title="Dev",
        company="Acme",
        url="https://example.com/1",
        board=JobBoard.LINKEDIN,
        description="cybersecurity role",
    )
    match = MatchResult(
        job=job,
        score=0.9,
        semantic_score=0.9,
        skill_score=0.9,
        matched_skills=["Python"],
        missing_skills=[],
        summary="",
    )
    monkeypatch.setattr(
        "job_applicator.embeddings.matching.JobMatcher",
        lambda *a, **k: MagicMock(rank_jobs=AsyncMock(return_value=[match])),
    )

    engine = MagicMock()
    engine.tailor_verified = AsyncMock(
        return_value=_tailored(job_title="Dev", job_company="Acme", job_url=str(job.url))
    )
    monkeypatch.setattr(
        "job_applicator.documents.resume_tailor.ResumeTailor", lambda *a, **k: engine
    )

    fake_pdf = tmp_path / "tailored_Acme_Dev_20260625_000000.pdf"
    fake_pdf.write_bytes(b"pdf")

    with patch(
        "job_applicator.documents.pdf_renderer.PDFRenderer.render_resume",
        new=AsyncMock(return_value=fake_pdf),
    ) as mock_render:
        result = CliRunner().invoke(
            cli.app,
            [
                "batch",
                "--resume",
                "r.pdf",
                "--jobs-file",
                str(_write_jobs_file(tmp_path, [job])),
                "--format",
                "pdf",
                "--template",
                "classic",
                "--category",
                "cybersecurity",
                "--no-cover-letter",
            ],
            env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
        )

    assert result.exit_code == 0, result.output
    mock_render.assert_awaited_once()
    assert mock_render.await_args.kwargs["template"] == "classic"
    assert mock_render.await_args.kwargs["category"] == "cybersecurity"


def test_apply_pdf_flags_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``apply --format pdf --template classic --category cybersecurity`` is accepted and
    the PDF artifact helper is invoked for cover letters.
    """
    import job_applicator.cli as cli
    from job_applicator.models import JobBoard, JobListing

    job = JobListing(
        title="Dev",
        company="Acme",
        url="https://example.com/1",
        board=JobBoard.LINKEDIN,
        description="cybersecurity role",
    )

    loader = MagicMock()
    loader.load.return_value = ResumeData(raw_text="John Doe\njohn@example.com\nPython")
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    monkeypatch.setattr(cli, "_run_ats_preflight", lambda r: MagicMock(score=0.8))
    monkeypatch.setattr(
        cli, "_detect_tone", lambda job: MagicMock(primary="professional", confidence=0.9)
    )
    monkeypatch.setattr(
        cli,
        "_get_jobs_store",
        lambda: MagicMock(list_jobs=lambda *a, **k: []),
    )

    class _BrowserCtx:
        async def __aenter__(self) -> MagicMock:
            return MagicMock()

        async def __aexit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(cli, "_make_browser", lambda *a, **k: _BrowserCtx())
    monkeypatch.setattr(
        cli,
        "_make_scraper",
        lambda *a, **k: MagicMock(
            scrape=AsyncMock(return_value=[job]),
        ),
    )
    monkeypatch.setattr(
        cli,
        "_make_applicator",
        lambda *a, **k: MagicMock(
            apply=AsyncMock(
                return_value=MagicMock(
                    job=job,
                    status=MagicMock(value="pending"),
                    error_message=None,
                    notes="",
                    cover_letter="Dear Hiring Manager,",
                    dry_run=None,
                )
            )
        ),
    )

    fake_pdf = tmp_path / "cover_letter_Acme_Dev_20260625_000000.pdf"
    fake_pdf.write_bytes(b"pdf")

    with (
        patch("job_applicator.documents.cover_letter.CoverLetterGenerator") as mock_gen_cls,
        patch(
            "job_applicator.documents.pdf_renderer.PDFRenderer.render_cover_letter",
            new=AsyncMock(return_value=fake_pdf),
        ) as mock_render,
    ):
        mock_gen = mock_gen_cls.return_value
        mock_gen.generate_verified = AsyncMock(return_value="Dear Hiring Manager,")

        result = CliRunner().invoke(
            cli.app,
            [
                "apply",
                "--query",
                "python",
                "--resume",
                "r.pdf",
                "--format",
                "pdf",
                "--template",
                "classic",
                "--category",
                "cybersecurity",
                "--limit",
                "1",
            ],
            env={"JOB_APPLICATOR_OUTPUT_DIR": str(tmp_path)},
        )

    assert result.exit_code == 0, result.output
    mock_render.assert_awaited_once()
    assert mock_render.await_args.kwargs["template"] == "classic"
    assert mock_render.await_args.kwargs["category"] == "cybersecurity"


def _write_jobs_file(tmp_path: Path, jobs: list[JobListing]) -> Path:
    import json

    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "title": j.title,
                    "company": j.company,
                    "url": str(j.url),
                    "board": j.board.value,
                    "description": j.description,
                }
                for j in jobs
            ]
        ),
        encoding="utf-8",
    )
    return path
