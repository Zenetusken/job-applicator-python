"""Integration test for PDF rendering: real Typst compile + PyMuPDF text extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("typst")
pytest.importorskip("fitz")

import fitz

from job_applicator.documents.ats_checker import ATSChecker
from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedEducationEntry,
    FormattedExperienceEntry,
    FormattedResume,
    FormattedSkillGroup,
)
from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.documents.resume import ResumeLoader
from job_applicator.models import CoverLetterResult, TailoredResume


def _extract_text(path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
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
            "Senior Python Engineer\n\n"
            "Experience\n"
            "Acme Corp - Staff Engineer, 2020-Present\n"
            "Built async microservices\n"
            "Improved API latency by 40%\n\n"
            "Skills\n"
            "Python, Asyncio, PostgreSQL"
        ),
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        match_score=0.85,
        semantic_score=0.80,
        skill_score=0.90,
        changes_summary="Emphasized Python async and PostgreSQL experience for senior role.",
    )


@pytest.fixture
def sample_cover_letter() -> CoverLetterResult:
    return CoverLetterResult(
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        cover_letter_text=(
            "Dear Hiring Manager,\n\n"
            "I am excited to apply for the Senior Python Engineer role at Acme Corp. "
            "My experience building async microservices aligns well with your needs.\n\n"
            "I have deep expertise in Python, Asyncio, and PostgreSQL. "
            "I look forward to contributing to your team.\n\n"
            "Sincerely,\n"
            "Alex Rivera"
        ),
    )


@pytest.fixture
def formatted_resume() -> FormattedResume:
    return FormattedResume(
        name="Alex Rivera",
        title="Senior Python Engineer",
        email="alex.rivera@example.com",
        phone="416-555-0199",
        location="Toronto, ON",
        summary="Senior Python engineer with a focus on async systems and data pipelines.",
        experience=[
            FormattedExperienceEntry(
                title="Staff Engineer",
                company="Acme Corp",
                location="Toronto, ON",
                start_date="2020",
                end_date="Present",
                bullets=[
                    "Built async microservices handling 10k req/s",
                    "Improved API latency by 40%",
                ],
            ),
        ],
        education=[
            FormattedEducationEntry(
                degree="Bachelor of Computer Science",
                institution="University of Toronto",
                location="Toronto, ON",
                start_date="2014",
                end_date="2018",
            ),
        ],
        skills=[
            FormattedSkillGroup(
                category="Languages & Frameworks",
                skills=["Python", "Asyncio", "PostgreSQL"],
            ),
        ],
    )


@pytest.fixture
def formatted_cover_letter() -> FormattedCoverLetter:
    return FormattedCoverLetter(
        recipient_company="Acme Corp",
        date="2026-06-25",
        greeting="Dear Hiring Manager,",
        paragraphs=[
            "I am excited to apply for the Senior Python Engineer role at Acme Corp. "
            "My experience building async microservices aligns well with your needs.",
            "I have deep expertise in Python, Asyncio, and PostgreSQL. "
            "I look forward to contributing to your team.",
        ],
        closing="Sincerely",
        signature="Alex Rivera",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_render_resume_to_pdf_and_extract_text(
    sample_tailored: TailoredResume,
    formatted_resume: FormattedResume,
    app_settings: object,
    tmp_path: Path,
) -> None:
    """Render a tailored résumé to PDF and verify key content is extractable."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)

    with patch.object(
        renderer, "_format_resume_with_instructor", new_callable=AsyncMock
    ) as mock_format:
        mock_format.return_value = formatted_resume
        pdf_path = await renderer.render_resume(sample_tailored)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    assert pdf_path.suffix == ".pdf"

    text = _extract_text(pdf_path)
    assert "Alex Rivera" in text
    assert "Senior Python Engineer" in text
    assert "alex.rivera@example.com" in text
    assert "416-555-0199" in text
    assert "Acme Corp" in text
    assert "Python" in text
    assert "Asyncio" in text
    assert "PostgreSQL" in text
    assert "Built async microservices" in text

    parsed = ResumeLoader().parse_text(text)
    ats = ATSChecker().check(parsed)
    assert ats.is_compatible, ats.warnings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_render_cover_letter_to_pdf_and_extract_text(
    sample_cover_letter: CoverLetterResult,
    formatted_cover_letter: FormattedCoverLetter,
    app_settings: object,
    tmp_path: Path,
) -> None:
    """Render a cover letter to PDF and verify key content is extractable."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)

    with patch.object(
        renderer, "_format_cover_letter_with_instructor", new_callable=AsyncMock
    ) as mock_format:
        mock_format.return_value = formatted_cover_letter
        pdf_path = await renderer.render_cover_letter(sample_cover_letter)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    assert pdf_path.suffix == ".pdf"

    text = _extract_text(pdf_path)
    assert "Acme Corp" in text
    assert "Dear Hiring Manager" in text
    assert "Python" in text
    assert "Asyncio" in text
    assert "PostgreSQL" in text
    assert "Sincerely" in text
    assert "Alex Rivera" in text
