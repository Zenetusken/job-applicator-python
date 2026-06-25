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

if TYPE_CHECKING:
    from job_applicator.config import AppSettings
    from job_applicator.models import CoverLetterResult, TailoredResume


def _safe(text: str) -> str:
    """Filesystem-safe slug: alphanumerics/-/_ kept, everything else → '_', capped."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:30]


def _write_text(path: Path, content: str) -> None:
    """Write text, wrapping a filesystem failure as a typed ``DocumentError`` (CLAUDE.md:
    every raised exception is a ``JobApplicatorError`` subclass)."""
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise DocumentError(f"Cannot write {path}: {exc}") from exc


def artifact_basename(company: str, title: str, *, when: datetime) -> str:
    """`tailored_<company>_<title>_<YYYYMMDD_HHMMSS>` (no extension)."""
    return f"tailored_{_safe(company)}_{_safe(title)}_{when.strftime('%Y%m%d_%H%M%S')}"


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
        f"cover_letter_{_safe(result.job_company)}_{_safe(result.job_title)}"
        f"_{when.strftime('%Y%m%d_%H%M%S')}"
    )
    cl_path = output_dir / f"{base}.txt"
    _write_text(cl_path, result.cover_letter_text)
    result.output_path = str(cl_path)
    meta_path = output_dir / f"{base}.meta.json"
    _write_text(meta_path, result.model_dump_json(indent=2))
    return str(cl_path), str(meta_path)


def _pdf_path(
    output_dir: Path, prefix: str, company: str, title: str, template: str, when: datetime
) -> Path:
    """Build a spec-compliant PDF artifact path with a deterministic timestamp."""
    ts = when.strftime("%Y%m%d_%H%M%S")
    us = f"{when.microsecond:06d}"
    base = f"{prefix}_{_safe(company)}_{_safe(title)}_{ts}_{us}_{template}"
    return output_dir / f"{base}.pdf"


def _ensure_pdf_path(rendered: Path, target: Path) -> Path:
    """Ensure the rendered PDF ends up at the deterministic ``target`` path.

    ``PDFRenderer`` selects its own filename from the current clock; the helpers
    use a caller-supplied ``when`` for testability and stable output names. If
    the renderer wrote the file elsewhere, move it into place.

    Raises:
        PDFRenderError: if the renderer did not write a PDF.
        DocumentError: if the PDF cannot be moved to ``target``.
    """
    if rendered == target:
        if rendered.exists():
            return target
        raise PDFRenderError(f"Renderer returned expected path but no PDF was written: {rendered}")
    if rendered.exists():
        if target.exists():
            raise DocumentError(f"Target PDF already exists: {target}")
        try:
            rendered.rename(target)
        except FileExistsError as exc:
            raise DocumentError(f"Target PDF already exists: {target}") from exc
        except OSError as exc:
            raise DocumentError(
                f"Cannot rename rendered PDF {rendered} to {target}: {exc}"
            ) from exc
        return target
    raise PDFRenderError(f"Renderer did not write a PDF at {rendered}")


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
        DocumentError: if the rendered PDF cannot be moved or the sidecar cannot
            be written.
    """
    renderer = PDFRenderer(settings, output_dir=output_dir)
    target = _pdf_path(
        output_dir, "tailored", tailored.job_company, tailored.job_title, template, when
    )
    try:
        rendered = await renderer.render_resume(
            tailored, job=None, template=template, category=category
        )
    except (PDFRenderError, DocumentError):
        raise
    except Exception as exc:
        raise PDFRenderError(f"Failed to render tailored PDF: {exc}") from exc
    final = _ensure_pdf_path(rendered, target)
    tailored.pdf_path = str(final)
    _write_text(final.with_suffix(".meta.json"), tailored.model_dump_json(indent=2))
    return final


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
        DocumentError: if the rendered PDF cannot be moved or the sidecar cannot
            be written.
    """
    renderer = PDFRenderer(settings, output_dir=output_dir)
    target = _pdf_path(
        output_dir, "cover_letter", result.job_company, result.job_title, template, when
    )
    try:
        rendered = await renderer.render_cover_letter(
            result, job=None, template=template, category=category
        )
    except (PDFRenderError, DocumentError):
        raise
    except Exception as exc:
        raise PDFRenderError(f"Failed to render cover-letter PDF: {exc}") from exc
    final = _ensure_pdf_path(rendered, target)
    result.pdf_path = str(final)
    _write_text(final.with_suffix(".meta.json"), result.model_dump_json(indent=2))
    return final
