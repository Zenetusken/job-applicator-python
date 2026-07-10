"""Raster repeatability checks for deterministic PDF rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("typst")
pytest.importorskip("fitz")

import fitz

from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.models import CoverLetterResult, TailoredResume

pytestmark = pytest.mark.integration
_DPI = 150


def _render_page(pdf_path: Path) -> np.ndarray:
    doc = fitz.open(pdf_path)
    try:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(_DPI / 72, _DPI / 72), alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    finally:
        doc.close()


def _tailored() -> TailoredResume:
    return TailoredResume(
        original_path="resume.pdf",
        tailored_text=(
            "Alex Rivera\n"
            "alex@example.com | 416-555-0199 | Toronto, ON\n\n"
            "SUMMARY\nSenior Python engineer focused on async systems.\n\n"
            "EXPERIENCE\nStaff Engineer | Acme Corp | 2020 - Present\n"
            "• Built async microservices.\n\n"
            "EDUCATION\nBachelor of Computer Science | University | 2018\n\n"
            "SKILLS\nPython, Asyncio, PostgreSQL"
        ),
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        match_score=0.85,
        semantic_score=0.80,
        skill_score=0.90,
        changes_summary="Source-backed summary overlay.",
    )


def _cover() -> CoverLetterResult:
    return CoverLetterResult(
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        cover_letter_text=(
            "Dear Hiring Manager,\n\n"
            "I am applying for the Senior Python Engineer role at Acme Corp.\n\n"
            "I built async microservices with Python and PostgreSQL.\n\n"
            "I would welcome the opportunity to discuss my application.\n\n"
            "Sincerely,\nAlex Rivera"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["modern", "classic", "minimal"])
async def test_resume_pdf_raster_is_repeatable(
    app_settings: object,
    tmp_path: Path,
    template: str,
) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    first = await renderer.render_resume(
        _tailored(), template=template, output_path=tmp_path / f"first-{template}.pdf"
    )
    second = await renderer.render_resume(
        _tailored(), template=template, output_path=tmp_path / f"second-{template}.pdf"
    )
    assert np.array_equal(_render_page(first), _render_page(second))


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["modern", "classic", "minimal"])
async def test_cover_letter_pdf_raster_is_repeatable(
    app_settings: object,
    tmp_path: Path,
    template: str,
) -> None:
    renderer = PDFRenderer(settings=app_settings, output_dir=tmp_path)
    first = await renderer.render_cover_letter(
        _cover(), template=template, output_path=tmp_path / f"first-letter-{template}.pdf"
    )
    second = await renderer.render_cover_letter(
        _cover(), template=template, output_path=tmp_path / f"second-letter-{template}.pdf"
    )
    assert np.array_equal(_render_page(first), _render_page(second))
