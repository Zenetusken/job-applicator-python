"""Integration tests for deterministic Typst rendering and PDF text extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("typst")
pytest.importorskip("fitz")

import fitz

from job_applicator.documents.ats_checker import ATSChecker
from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.documents.resume import ResumeLoader
from job_applicator.models import CoverLetterResult, TailoredResume


def _extract_text(path: Path) -> str:
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


@pytest.fixture
def sample_tailored() -> TailoredResume:
    return TailoredResume(
        original_path="resume.pdf",
        tailored_text=(
            "Alex Rivera\n"
            "alex.rivera@example.com | 416-555-0199 | Toronto, ON\n\n"
            "SUMMARY\n"
            "Senior Python engineer focused on async systems and data pipelines.\n\n"
            "EXPERIENCE\n"
            "Staff Engineer | Acme Corp | 2020 - Present\n"
            "• Built async microservices handling production requests.\n"
            "• Improved API latency by 40%.\n\n"
            "EDUCATION\n"
            "Bachelor of Computer Science | University of Toronto | 2018\n\n"
            "SKILLS\n"
            "Python, Asyncio, PostgreSQL"
        ),
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        match_score=0.85,
        semantic_score=0.80,
        skill_score=0.90,
        changes_summary="Replaced the source-backed summary.",
    )


@pytest.fixture
def sample_cover_letter() -> CoverLetterResult:
    return CoverLetterResult(
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        cover_letter_text=(
            "Dear Hiring Manager,\n\n"
            "I am applying for the Senior Python Engineer role at Acme Corp.\n\n"
            "I built async microservices and worked with Python, Asyncio, and PostgreSQL.\n\n"
            "I would welcome the opportunity to discuss my application.\n\n"
            "Sincerely,\n"
            "Alex Rivera"
        ),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_render_resume_to_pdf_and_extract_text(
    sample_tailored: TailoredResume,
    app_settings: object,
    tmp_path: Path,
) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    pdf_path = await renderer.render_resume(sample_tailored)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    text = _extract_text(pdf_path)
    for expected in (
        "Alex Rivera",
        "alex.rivera@example.com",
        "416-555-0199",
        "Acme Corp",
        "Python",
        "Asyncio",
        "PostgreSQL",
        "Built async microservices",
    ):
        assert expected in text

    ats = ATSChecker().check(ResumeLoader().parse_text(text))
    assert ats.is_compatible, ats.warnings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_render_cover_letter_to_pdf_and_extract_text(
    sample_cover_letter: CoverLetterResult,
    app_settings: object,
    tmp_path: Path,
) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    pdf_path = await renderer.render_cover_letter(sample_cover_letter)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    text = _extract_text(pdf_path)
    for expected in (
        "Acme Corp",
        "Dear Hiring Manager",
        "Python",
        "Asyncio",
        "PostgreSQL",
        "Sincerely",
        "Alex Rivera",
    ):
        assert expected in text
