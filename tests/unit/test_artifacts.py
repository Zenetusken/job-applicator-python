"""Unit tests for the shared output-artifact helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from job_applicator.config import AppSettings
from job_applicator.documents.artifacts import (
    artifact_basename,
    write_cover_letter,
    write_cover_letter_pdf,
    write_tailored,
    write_tailored_pdf,
)
from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.exceptions import DocumentError, PDFRenderError
from job_applicator.models import CoverLetterResult, TailoredResume
from job_applicator.utils.path import safe_filename_slug

WHEN = datetime(2026, 6, 22, 14, 30, 0)
WHEN_US = datetime(2026, 6, 22, 14, 30, 0, 123456)


def _tailored() -> TailoredResume:
    return TailoredResume(
        original_path="/r.pdf",
        tailored_text="TAILORED BODY",
        job_title="Sr Eng",
        job_company="Ac/me Inc",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="c",
    )


def _cover_letter() -> CoverLetterResult:
    return CoverLetterResult(
        job_title="Sr Eng",
        job_company="Acme",
        job_url="https://x/1",
        cover_letter_text="Dear hiring manager,",
        attempt=1,
        prompt_version="1.0",
    )


def test_write_tailored_writes_files_and_sets_output_path(tmp_path: Path) -> None:
    tailored = _tailored()
    resume_path, meta_path = write_tailored(tmp_path, tailored, when=WHEN)
    assert Path(resume_path).read_text() == "TAILORED BODY"
    assert Path(meta_path).exists()
    assert tailored.output_path == resume_path
    assert Path(resume_path).name == "tailored_Ac_me_Inc_Sr_Eng_20260622_143000.txt"


def test_write_cover_letter_writes_files_and_sets_output_path(tmp_path: Path) -> None:
    result = _cover_letter()
    cl_path, meta_path = write_cover_letter(tmp_path, result, when=WHEN)
    assert Path(cl_path).read_text() == "Dear hiring manager,"
    assert Path(meta_path).exists()
    assert result.output_path == cl_path
    assert Path(cl_path).name == "cover_letter_Acme_Sr_Eng_20260622_143000.txt"


def test_safe_filename_slug_sanitizes_and_caps_at_30() -> None:
    slug = safe_filename_slug("A Very Long Company Name That Exceeds Thirty Characters")
    assert "/" not in slug and ":" not in slug  # specials → "_"
    assert len(slug) == 30
    assert slug == "A_Very_Long_Company_Name_That_"


def test_artifact_basename_uses_slug_and_timestamp() -> None:
    base = artifact_basename(
        "A Very Long Company Name That Exceeds Thirty Characters",
        "Title/With:Specials",
        when=WHEN,
    )
    assert base.startswith("tailored_")
    assert "/" not in base and ":" not in base
    assert base.endswith("_20260622_143000")


def test_write_tailored_wraps_oserror_as_document_error(tmp_path: Path) -> None:
    """A filesystem failure surfaces as a typed DocumentError, not a bare OSError."""
    not_a_dir = tmp_path / "afile"
    not_a_dir.write_text("x")  # output_dir is a FILE → writing under it fails
    tailored = TailoredResume(
        original_path="/r.pdf",
        tailored_text="T",
        job_title="T",
        job_company="C",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="c",
    )
    with pytest.raises(DocumentError):
        write_tailored(not_a_dir, tailored, when=WHEN)


async def test_write_tailored_pdf_renames_rendered_file_and_writes_meta(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()
    expected = output_dir / "tailored_Ac_me_Inc_Sr_Eng_20260622_143000_123456_modern.pdf"
    rendered = output_dir / "renderer_tmp_tailored.pdf"
    rendered.write_bytes(b"fake pdf bytes")

    with patch.object(
        PDFRenderer, "render_resume", new=AsyncMock(return_value=rendered)
    ) as mock_render:
        path = await write_tailored_pdf(output_dir, tailored, settings, when=WHEN_US)

    assert path == expected
    assert not rendered.exists()
    assert expected.read_bytes() == b"fake pdf bytes"
    assert tailored.pdf_path == str(expected)
    meta_path = expected.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["pdf_path"] == str(expected)
    mock_render.assert_awaited_once_with(tailored, job=None, template="modern", category=None)


async def test_write_cover_letter_pdf_renames_rendered_file_and_writes_meta(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()
    expected = output_dir / "cover_letter_Acme_Sr_Eng_20260622_143000_123456_modern.pdf"
    rendered = output_dir / "renderer_tmp_cl.pdf"
    rendered.write_bytes(b"fake cl bytes")

    with patch.object(
        PDFRenderer, "render_cover_letter", new=AsyncMock(return_value=rendered)
    ) as mock_render:
        path = await write_cover_letter_pdf(output_dir, result, settings, when=WHEN_US)

    assert path == expected
    assert not rendered.exists()
    assert expected.read_bytes() == b"fake cl bytes"
    assert result.pdf_path == str(expected)
    meta_path = expected.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["pdf_path"] == str(expected)
    mock_render.assert_awaited_once_with(result, job=None, template="modern", category=None)


async def test_write_tailored_pdf_uses_template_and_microseconds_in_name(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()
    when = datetime(2026, 6, 22, 14, 30, 0, 7)
    rendered = output_dir / "renderer_tmp_classic.pdf"
    rendered.write_bytes(b"x")

    with patch.object(PDFRenderer, "render_resume", new=AsyncMock(return_value=rendered)):
        path = await write_tailored_pdf(
            output_dir, tailored, settings, template="classic", when=when
        )

    assert re.fullmatch(r"tailored_Ac_me_Inc_Sr_Eng_20260622_143000_\d{6}_classic\.pdf", path.name)


async def test_write_cover_letter_pdf_uses_template_and_microseconds_in_name(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()
    when = datetime(2026, 6, 22, 14, 30, 0, 7)
    rendered = output_dir / "renderer_tmp_classic.pdf"
    rendered.write_bytes(b"x")

    with patch.object(PDFRenderer, "render_cover_letter", new=AsyncMock(return_value=rendered)):
        path = await write_cover_letter_pdf(
            output_dir, result, settings, template="classic", when=when
        )

    assert re.fullmatch(r"cover_letter_Acme_Sr_Eng_20260622_143000_\d{6}_classic\.pdf", path.name)


async def test_write_tailored_pdf_passes_category_to_renderer(tmp_path: Path) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()
    rendered = output_dir / "renderer_tmp_cat.pdf"
    rendered.write_bytes(b"x")

    with patch.object(
        PDFRenderer, "render_resume", new=AsyncMock(return_value=rendered)
    ) as mock_render:
        await write_tailored_pdf(
            output_dir, tailored, settings, category="tech-support", when=WHEN_US
        )

    mock_render.assert_awaited_once_with(
        tailored, job=None, template="modern", category="tech-support"
    )


async def test_write_cover_letter_pdf_passes_category_to_renderer(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()
    rendered = output_dir / "renderer_tmp_cat.pdf"
    rendered.write_bytes(b"x")

    with patch.object(
        PDFRenderer, "render_cover_letter", new=AsyncMock(return_value=rendered)
    ) as mock_render:
        await write_cover_letter_pdf(
            output_dir, result, settings, category="tech-support", when=WHEN_US
        )

    mock_render.assert_awaited_once_with(
        result, job=None, template="modern", category="tech-support"
    )


async def test_write_tailored_pdf_wraps_render_failure(tmp_path: Path) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()

    with patch.object(
        PDFRenderer, "render_resume", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        with pytest.raises(PDFRenderError, match="Failed to render tailored PDF"):
            await write_tailored_pdf(output_dir, tailored, settings, when=WHEN_US)

    assert not tailored.pdf_path
    assert not (output_dir / "tailored_Ac_me_Inc_Sr_Eng_20260622_143000_123456_modern.pdf").exists()


async def test_write_cover_letter_pdf_wraps_render_failure(tmp_path: Path) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()

    with patch.object(
        PDFRenderer,
        "render_cover_letter",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(PDFRenderError, match="Failed to render cover-letter PDF"):
            await write_cover_letter_pdf(output_dir, result, settings, when=WHEN_US)

    assert not result.pdf_path


async def test_write_tailored_pdf_propagates_typed_renderer_errors(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()

    with patch.object(
        PDFRenderer,
        "render_resume",
        new=AsyncMock(side_effect=PDFRenderError("template missing")),
    ):
        with pytest.raises(PDFRenderError, match="template missing") as exc_info:
            await write_tailored_pdf(output_dir, tailored, settings, when=WHEN_US)

    assert "Failed to render tailored PDF" not in str(exc_info.value)
    assert not tailored.pdf_path


async def test_write_cover_letter_pdf_propagates_typed_renderer_errors(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()

    with patch.object(
        PDFRenderer,
        "render_cover_letter",
        new=AsyncMock(side_effect=PDFRenderError("template missing")),
    ):
        with pytest.raises(PDFRenderError, match="template missing") as exc_info:
            await write_cover_letter_pdf(output_dir, result, settings, when=WHEN_US)

    assert "Failed to render cover-letter PDF" not in str(exc_info.value)
    assert not result.pdf_path


async def test_write_tailored_pdf_raises_when_rendered_file_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()
    missing = output_dir / "missing.pdf"

    with patch.object(PDFRenderer, "render_resume", new=AsyncMock(return_value=missing)):
        with pytest.raises(PDFRenderError, match="Renderer did not write a PDF"):
            await write_tailored_pdf(output_dir, tailored, settings, when=WHEN_US)

    assert not tailored.pdf_path
    assert not list(output_dir.glob("*.meta.json"))


async def test_write_cover_letter_pdf_raises_when_rendered_file_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()
    missing = output_dir / "missing.pdf"

    with patch.object(PDFRenderer, "render_cover_letter", new=AsyncMock(return_value=missing)):
        with pytest.raises(PDFRenderError, match="Renderer did not write a PDF"):
            await write_cover_letter_pdf(output_dir, result, settings, when=WHEN_US)

    assert not result.pdf_path
    assert not list(output_dir.glob("*.meta.json"))


async def test_write_tailored_pdf_raises_on_rename_collision(tmp_path: Path) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()
    rendered = output_dir / "rendered.pdf"
    rendered.write_bytes(b"rendered")
    target = output_dir / "tailored_Ac_me_Inc_Sr_Eng_20260622_143000_123456_modern.pdf"
    target.write_bytes(b"existing")

    with patch.object(PDFRenderer, "render_resume", new=AsyncMock(return_value=rendered)):
        with pytest.raises(DocumentError, match="Target PDF already exists"):
            await write_tailored_pdf(output_dir, tailored, settings, when=WHEN_US)

    assert not tailored.pdf_path
    assert not target.with_suffix(".meta.json").exists()


async def test_write_cover_letter_pdf_raises_on_rename_collision(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()
    rendered = output_dir / "rendered.pdf"
    rendered.write_bytes(b"rendered")
    target = output_dir / "cover_letter_Acme_Sr_Eng_20260622_143000_123456_modern.pdf"
    target.write_bytes(b"existing")

    with patch.object(PDFRenderer, "render_cover_letter", new=AsyncMock(return_value=rendered)):
        with pytest.raises(DocumentError, match="Target PDF already exists"):
            await write_cover_letter_pdf(output_dir, result, settings, when=WHEN_US)

    assert not result.pdf_path
    assert not target.with_suffix(".meta.json").exists()


async def test_write_tailored_pdf_returns_target_when_renderer_writes_exact_path(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()
    expected = output_dir / "tailored_Ac_me_Inc_Sr_Eng_20260622_143000_123456_modern.pdf"
    expected.write_bytes(b"exact")

    with patch.object(
        PDFRenderer, "render_resume", new=AsyncMock(return_value=expected)
    ) as mock_render:
        path = await write_tailored_pdf(output_dir, tailored, settings, when=WHEN_US)

    assert path == expected
    assert expected.read_bytes() == b"exact"
    assert tailored.pdf_path == str(expected)
    assert expected.with_suffix(".meta.json").exists()
    mock_render.assert_awaited_once_with(tailored, job=None, template="modern", category=None)


async def test_write_tailored_pdf_raises_when_exact_path_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    tailored = _tailored()
    expected = output_dir / "tailored_Ac_me_Inc_Sr_Eng_20260622_143000_123456_modern.pdf"

    with patch.object(PDFRenderer, "render_resume", new=AsyncMock(return_value=expected)):
        with pytest.raises(
            PDFRenderError, match="Renderer returned expected path but no PDF was written"
        ):
            await write_tailored_pdf(output_dir, tailored, settings, when=WHEN_US)

    assert not tailored.pdf_path
    assert not expected.with_suffix(".meta.json").exists()


async def test_write_cover_letter_pdf_returns_target_when_renderer_writes_exact_path(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()
    expected = output_dir / "cover_letter_Acme_Sr_Eng_20260622_143000_123456_modern.pdf"
    expected.write_bytes(b"exact")

    with patch.object(
        PDFRenderer, "render_cover_letter", new=AsyncMock(return_value=expected)
    ) as mock_render:
        path = await write_cover_letter_pdf(output_dir, result, settings, when=WHEN_US)

    assert path == expected
    assert expected.read_bytes() == b"exact"
    assert result.pdf_path == str(expected)
    assert expected.with_suffix(".meta.json").exists()
    mock_render.assert_awaited_once_with(result, job=None, template="modern", category=None)


async def test_write_cover_letter_pdf_raises_when_exact_path_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    result = _cover_letter()
    expected = output_dir / "cover_letter_Acme_Sr_Eng_20260622_143000_123456_modern.pdf"

    with patch.object(PDFRenderer, "render_cover_letter", new=AsyncMock(return_value=expected)):
        with pytest.raises(
            PDFRenderError, match="Renderer returned expected path but no PDF was written"
        ):
            await write_cover_letter_pdf(output_dir, result, settings, when=WHEN_US)

    assert not result.pdf_path
    assert not expected.with_suffix(".meta.json").exists()
