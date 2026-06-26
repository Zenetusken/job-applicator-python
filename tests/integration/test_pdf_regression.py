"""Visual regression spike for PDF rendering.

Generates PDFs from fixed inputs, rasterizes them with PyMuPDF at a fixed DPI, and
compares against checked-in reference images. The test is skipped by default because
raster output can drift across OS, font, and HarfBuzz versions. Run with:

    JOB_APPLICATOR_PDF_REGRESSION=1 pytest tests/integration/test_pdf_regression.py

When references are missing, the test saves them as baselines and skips the assertion.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

pytest.importorskip("typst")
pytest.importorskip("fitz")

import fitz

from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedExperienceEntry,
    FormattedResume,
    FormattedSkillGroup,
)
from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.models import CoverLetterResult, TailoredResume

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("JOB_APPLICATOR_PDF_REGRESSION") != "1",
        reason="Visual regression is opt-in (set JOB_APPLICATOR_PDF_REGRESSION=1)",
    ),
]

_REF_DIR = Path(__file__).with_suffix("").parent / "references"
_REF_DIR.mkdir(parents=True, exist_ok=True)
_DPI = 150


def _render_page_to_numpy(pdf_path: Path, dpi: int = _DPI) -> np.ndarray:
    """Render the first page of a PDF to a numpy RGBA array at the given DPI."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        # pixmap.samples is RGB (alpha=False); reshape to (height, width, 3)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    finally:
        doc.close()


@pytest.fixture
def _fixed_tailored() -> TailoredResume:
    return TailoredResume(
        original_path="resume.pdf",
        tailored_text=(
            "Alex Rivera\nSenior Python Engineer\n\n"
            "Experience\nAcme Corp - Staff Engineer, 2020-Present\n"
            "Built async microservices\n\n"
            "Skills\nPython, Asyncio, PostgreSQL"
        ),
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        match_score=0.85,
        semantic_score=0.80,
        skill_score=0.90,
        changes_summary="Emphasized Python async and PostgreSQL.",
    )


@pytest.fixture
def _fixed_cover_letter() -> CoverLetterResult:
    return CoverLetterResult(
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        cover_letter_text=(
            "Dear Hiring Manager,\n\n"
            "I am excited to apply for the Senior Python Engineer role at Acme Corp.\n\n"
            "Sincerely,\nAlex Rivera"
        ),
    )


@pytest.fixture
def _fixed_formatted_resume() -> FormattedResume:
    return FormattedResume(
        name="Alex Rivera",
        title="Senior Python Engineer",
        email="alex.rivera@example.com",
        location="Toronto, ON",
        summary="Senior Python engineer with a focus on async systems.",
        experience=[
            FormattedExperienceEntry(
                title="Staff Engineer",
                company="Acme Corp",
                location="Toronto, ON",
                start_date="2020",
                end_date="Present",
                bullets=["Built async microservices handling 10k req/s"],
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
def _fixed_formatted_cover_letter() -> FormattedCoverLetter:
    return FormattedCoverLetter(
        recipient_company="Acme Corp",
        date="2026-06-25",
        greeting="Dear Hiring Manager,",
        paragraphs=[
            "I am excited to apply for the Senior Python Engineer role at Acme Corp.",
        ],
        closing="Sincerely",
        signature="Alex Rivera",
    )


@pytest.mark.asyncio
async def test_resume_pdf_visual_regression(
    _fixed_tailored: TailoredResume,
    _fixed_formatted_resume: FormattedResume,
    app_settings: object,
    tmp_path: Path,
) -> None:
    """Render a fixed résumé and compare the rasterized page to the reference."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    with patch.object(
        renderer, "_format_resume_with_instructor", new_callable=AsyncMock
    ) as mock_format:
        mock_format.return_value = _fixed_formatted_resume
        pdf_path = await renderer.render_resume(_fixed_tailored)

    rendered = _render_page_to_numpy(pdf_path)
    reference = _REF_DIR / "cv_modern.pdf"
    if not reference.exists():
        # Save baseline PDF on first run; do not assert.
        import shutil

        shutil.copy(pdf_path, reference)
        pytest.skip(f"Saved new reference: {reference}")

    expected = _render_page_to_numpy(reference)
    assert rendered.shape == expected.shape, (
        f"Dimension mismatch: rendered {rendered.shape} vs reference {expected.shape}"
    )
    diff = np.abs(rendered.astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 2, f"Visual diff too large (max={diff.max()})"


@pytest.mark.asyncio
async def test_cover_letter_pdf_visual_regression(
    _fixed_cover_letter: CoverLetterResult,
    _fixed_formatted_cover_letter: FormattedCoverLetter,
    app_settings: object,
    tmp_path: Path,
) -> None:
    """Render a fixed cover letter and compare the rasterized page to the reference."""
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    with patch.object(
        renderer, "_format_cover_letter_with_instructor", new_callable=AsyncMock
    ) as mock_format:
        mock_format.return_value = _fixed_formatted_cover_letter
        pdf_path = await renderer.render_cover_letter(_fixed_cover_letter)

    rendered = _render_page_to_numpy(pdf_path)
    reference = _REF_DIR / "cover_letter_modern.pdf"
    if not reference.exists():
        import shutil

        shutil.copy(pdf_path, reference)
        pytest.skip(f"Saved new reference: {reference}")

    expected = _render_page_to_numpy(reference)
    assert rendered.shape == expected.shape, (
        f"Dimension mismatch: rendered {rendered.shape} vs reference {expected.shape}"
    )
    diff = np.abs(rendered.astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 2, f"Visual diff too large (max={diff.max()})"
