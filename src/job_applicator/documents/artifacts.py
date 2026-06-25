"""Output-artifact helpers for tailored résumés / cover letters.

The ``tailored_<company>_<title>_<timestamp>.txt`` + ``.meta.json`` convention in one
place, used by the TUI action layer. (The CLI's batch/tailor paths still inline
equivalent logic — a future cleanup could adopt these helpers to fully converge.)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.exceptions import DocumentError, PDFRenderError
from job_applicator.utils.path import safe_filename_slug

if TYPE_CHECKING:
    from job_applicator.config import AppSettings
    from job_applicator.models import CoverLetterResult, TailoredResume


def _write_text(path: Path, content: str) -> None:
    """Write text, wrapping a filesystem failure as a typed ``DocumentError`` (CLAUDE.md:
    every raised exception is a ``JobApplicatorError`` subclass)."""
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise DocumentError(f"Cannot write {path}: {exc}") from exc


def artifact_basename(company: str, title: str, *, when: datetime) -> str:
    """`tailored_<company>_<title>_<YYYYMMDD_HHMMSS>` (no extension)."""
    company_slug = safe_filename_slug(company)
    title_slug = safe_filename_slug(title)
    return f"tailored_{company_slug}_{title_slug}_{when.strftime('%Y%m%d_%H%M%S')}"


def write_tailored(
    output_dir: Path, tailored: TailoredResume, *, when: datetime
) -> tuple[str, str]:
    """Write the tailored résumé text + its ``.meta.json`` sidecar.

    Sets ``tailored.output_path`` and returns ``(resume_path, meta_path)``. ``when`` is
    passed in (not read from the clock) so callers control the timestamp / testability.
    """
    base = artifact_basename(tailored.job_company, tailored.job_title, when=when)
    resume_path = output_dir / f"{base}.txt"
    _write_text(resume_path, tailored.tailored_text)
    tailored.output_path = str(resume_path)
    meta_path = output_dir / f"{base}.meta.json"
    _write_text(meta_path, tailored.model_dump_json(indent=2))
    return str(resume_path), str(meta_path)


def write_cover_letter(
    output_dir: Path, result: CoverLetterResult, *, when: datetime
) -> tuple[str, str]:
    """Write the cover-letter text + its ``.meta.json`` sidecar; sets ``result.output_path``."""
    base = (
        f"cover_letter_{safe_filename_slug(result.job_company)}_{safe_filename_slug(result.job_title)}"
        f"_{when.strftime('%Y%m%d_%H%M%S')}"
    )
    cl_path = output_dir / f"{base}.txt"
    _write_text(cl_path, result.cover_letter_text)
    result.output_path = str(cl_path)
    meta_path = output_dir / f"{base}.meta.json"
    _write_text(meta_path, result.model_dump_json(indent=2))
    return str(cl_path), str(meta_path)


async def write_tailored_pdf(
    output_dir: Path,
    tailored: TailoredResume,
    settings: AppSettings,
    *,
    template: str = "modern",
    category: str | None = None,
    when: datetime,
) -> Path:
    """Render a tailored résumé to PDF and update its sidecar.

    Sets ``tailored.pdf_path`` and writes a ``.meta.json`` sidecar for the PDF so
    it reflects the model including the new ``pdf_path``. Returns the path to the
    generated PDF.

    Raises:
        PDFRenderError: if rendering fails or the renderer did not produce a PDF.
        DocumentError: if the sidecar cannot be written.
    """
    renderer = PDFRenderer(settings, output_dir=output_dir)
    base = artifact_basename(tailored.job_company, tailored.job_title, when=when)
    target = output_dir / f"{base}.pdf"
    try:
        rendered = await renderer.render_resume(
            tailored, job=None, template=template, category=category, output_path=target
        )
    except (PDFRenderError, DocumentError):
        raise
    except Exception as exc:
        raise PDFRenderError(f"Failed to render tailored PDF: {exc}") from exc
    if not rendered.exists():
        raise PDFRenderError(f"Renderer did not write a PDF at {rendered}")
    tailored.pdf_path = str(rendered)
    _write_text(rendered.with_suffix(".meta.json"), tailored.model_dump_json(indent=2))
    return rendered


async def write_cover_letter_pdf(
    output_dir: Path,
    result: CoverLetterResult,
    settings: AppSettings,
    *,
    template: str = "modern",
    category: str | None = None,
    when: datetime,
) -> Path:
    """Render a cover letter to PDF and update its sidecar.

    Sets ``result.pdf_path`` and writes a ``.meta.json`` sidecar for the PDF so
    it reflects the model including the new ``pdf_path``. Returns the path to the
    generated PDF.

    Raises:
        PDFRenderError: if rendering fails or the renderer did not produce a PDF.
        DocumentError: if the sidecar cannot be written.
    """
    renderer = PDFRenderer(settings, output_dir=output_dir)
    target = output_dir / (
        f"cover_letter_{safe_filename_slug(result.job_company)}_{safe_filename_slug(result.job_title)}"
        f"_{when.strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    try:
        rendered = await renderer.render_cover_letter(
            result, job=None, template=template, category=category, output_path=target
        )
    except (PDFRenderError, DocumentError):
        raise
    except Exception as exc:
        raise PDFRenderError(f"Failed to render cover-letter PDF: {exc}") from exc
    if not rendered.exists():
        raise PDFRenderError(f"Renderer did not write a PDF at {rendered}")
    result.pdf_path = str(rendered)
    _write_text(rendered.with_suffix(".meta.json"), result.model_dump_json(indent=2))
    return rendered
