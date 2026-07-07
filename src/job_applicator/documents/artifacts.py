"""Output-artifact helpers for tailored résumés / cover letters.

The plain-text convention is ``tailored_<company>_<title>_<YYYYMMDD_HHMMSS>.txt`` +
``.meta.json``. PDF artifacts include microseconds and the template suffix to avoid
collisions: ``tailored_<company>_<title>_<YYYYMMDD_HHMMSS>_<microseconds>_<template>.pdf``.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.exceptions import DocumentError, PDFRenderError
from job_applicator.utils.path import safe_filename_slug

if TYPE_CHECKING:
    from job_applicator.config import AppSettings
    from job_applicator.models import CoverLetterResult, TailoredResume
    from job_applicator.utils.llm import LLMRuntime


def strip_markdown_bold(text: str) -> str:
    """Strip paired ``**bold**`` markers for human-facing output (``**Skills**`` → ``Skills``).

    The tailoring model emits ``**bold**`` for résumé section headers/titles and in its
    "changes made" summary — intentional for the headers, which the section parsers
    (``_looks_like_section_header``) and the PDF formatter consume. But the saved ``.txt``
    artifact, the on-screen preview, and the change-log are for humans, so the markers are
    stripped there. The raw ``TailoredResume.tailored_text`` keeps them for the parser/PDF path.
    """
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


def _write_text(path: Path, content: str) -> None:
    """Write text, wrapping a filesystem failure as a typed ``DocumentError`` (CLAUDE.md:
    every raised exception is a ``JobApplicatorError`` subclass)."""
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise DocumentError(f"Cannot write {path}: {exc}") from exc


def _write_unique_text(path: Path, content: str) -> Path:
    """Write text to ``path`` or a suffixed sibling without overwriting an existing artifact."""
    candidate = path
    for index in range(1000):
        if index:
            candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        try:
            with candidate.open("x", encoding="utf-8") as handle:
                handle.write(content)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise DocumentError(f"Cannot write {candidate}: {exc}") from exc
    raise DocumentError(f"Cannot find an unused artifact path for {path}")


def artifact_basename(company: str, title: str, *, when: datetime) -> str:
    """`tailored_<company>_<title>_<YYYYMMDD_HHMMSS>` (no extension)."""
    company_slug = safe_filename_slug(company)
    title_slug = safe_filename_slug(title)
    return f"tailored_{company_slug}_{title_slug}_{when.strftime('%Y%m%d_%H%M%S')}"


def pdf_artifact_basename(company: str, title: str, *, when: datetime, template: str) -> str:
    """`tailored_<company>_<title>_<YYYYMMDD_HHMMSS>_<microseconds>_<template>` (no extension)."""
    return f"{artifact_basename(company, title, when=when)}_{when.microsecond:06d}_{template}"


def cover_letter_pdf_basename(company: str, title: str, *, when: datetime, template: str) -> str:
    """`cover_letter_<company>_<title>_<YYYYMMDD_HHMMSS>_<microseconds>_<template>`."""
    company_slug = safe_filename_slug(company)
    title_slug = safe_filename_slug(title)
    return (
        f"cover_letter_{company_slug}_{title_slug}_{when.strftime('%Y%m%d_%H%M%S')}"
        f"_{when.microsecond:06d}_{template}"
    )


def write_tailored(
    output_dir: Path, tailored: TailoredResume, *, when: datetime
) -> tuple[str, str]:
    """Write the tailored résumé text + its ``.meta.json`` sidecar.

    Sets ``tailored.output_path`` and returns ``(resume_path, meta_path)``. ``when`` is
    passed in (not read from the clock) so callers control the timestamp / testability.
    """
    base = artifact_basename(tailored.job_company, tailored.job_title, when=when)
    resume_path = output_dir / f"{base}.txt"
    resume_path = _write_unique_text(resume_path, strip_markdown_bold(tailored.tailored_text))
    tailored.output_path = str(resume_path)
    meta_path = resume_path.with_suffix(".meta.json")
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
    cl_path = _write_unique_text(cl_path, result.cover_letter_text)
    result.output_path = str(cl_path)
    meta_path = cl_path.with_suffix(".meta.json")
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
    write_meta: bool = True,
    runtime: LLMRuntime | None = None,
) -> Path:
    """Render a tailored résumé to PDF and update its sidecar.

    Sets ``tailored.pdf_path`` and, unless ``write_meta`` is ``False``, writes a
    ``.meta.json`` sidecar for the PDF so it reflects the model including the new
    ``pdf_path``. Returns the path to the generated PDF.

    Raises:
        PDFRenderError: if rendering fails or the renderer did not produce a PDF.
        DocumentError: if the sidecar cannot be written.
    """
    renderer = PDFRenderer(settings, output_dir=output_dir, runtime=runtime)
    base = pdf_artifact_basename(
        tailored.job_company, tailored.job_title, when=when, template=template
    )
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
    rendered_str = str(rendered)
    tailored.pdf_path = rendered_str
    if not tailored.output_path:
        tailored.output_path = rendered_str
    if write_meta:
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
    write_meta: bool = True,
    runtime: LLMRuntime | None = None,
) -> Path:
    """Render a cover letter to PDF and update its sidecar.

    Sets ``result.pdf_path`` and, unless ``write_meta`` is ``False``, writes a
    ``.meta.json`` sidecar for the PDF so it reflects the model including the new
    ``pdf_path``. Returns the path to the generated PDF.

    Raises:
        PDFRenderError: if rendering fails or the renderer did not produce a PDF.
        DocumentError: if the sidecar cannot be written.
    """
    renderer = PDFRenderer(settings, output_dir=output_dir, runtime=runtime)
    base = cover_letter_pdf_basename(
        result.job_company, result.job_title, when=when, template=template
    )
    target = output_dir / f"{base}.pdf"
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
    rendered_str = str(rendered)
    result.pdf_path = rendered_str
    if not result.output_path:
        result.output_path = rendered_str
    if write_meta:
        _write_text(rendered.with_suffix(".meta.json"), result.model_dump_json(indent=2))
    return rendered
