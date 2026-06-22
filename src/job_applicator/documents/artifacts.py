"""Shared output-artifact helpers for tailored résumés.

One place for the ``tailored_<company>_<title>_<timestamp>.txt`` + ``.meta.json``
convention so the TUI action layer and the CLI don't drift into separate copies.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from job_applicator.models import CoverLetterResult, TailoredResume


def _safe(text: str) -> str:
    """Filesystem-safe slug: alphanumerics/-/_ kept, everything else → '_', capped."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:30]


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
    resume_path.write_text(tailored.tailored_text, encoding="utf-8")
    tailored.output_path = str(resume_path)
    meta_path = output_dir / f"{base}.meta.json"
    meta_path.write_text(tailored.model_dump_json(indent=2), encoding="utf-8")
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
    cl_path.write_text(result.cover_letter_text, encoding="utf-8")
    result.output_path = str(cl_path)
    meta_path = output_dir / f"{base}.meta.json"
    meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return str(cl_path), str(meta_path)
