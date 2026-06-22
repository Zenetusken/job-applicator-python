"""Unit tests for the shared output-artifact helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from job_applicator.documents.artifacts import (
    artifact_basename,
    write_cover_letter,
    write_tailored,
)
from job_applicator.models import CoverLetterResult, TailoredResume

WHEN = datetime(2026, 6, 22, 14, 30, 0)


def test_write_tailored_writes_files_and_sets_output_path(tmp_path: Path) -> None:
    tailored = TailoredResume(
        original_path="/r.pdf",
        tailored_text="TAILORED BODY",
        job_title="Sr Eng",
        job_company="Ac/me Inc",  # "/" and space must sanitize to "_"
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="c",
    )
    resume_path, meta_path = write_tailored(tmp_path, tailored, when=WHEN)
    assert Path(resume_path).read_text() == "TAILORED BODY"
    assert Path(meta_path).exists()
    assert tailored.output_path == resume_path
    assert Path(resume_path).name == "tailored_Ac_me_Inc_Sr_Eng_20260622_143000.txt"


def test_write_cover_letter_writes_files_and_sets_output_path(tmp_path: Path) -> None:
    result = CoverLetterResult(
        job_title="Sr Eng",
        job_company="Acme",
        job_url="https://x/1",
        cover_letter_text="Dear hiring manager,",
        attempt=1,
        prompt_version="1.0",
    )
    cl_path, meta_path = write_cover_letter(tmp_path, result, when=WHEN)
    assert Path(cl_path).read_text() == "Dear hiring manager,"
    assert Path(meta_path).exists()
    assert result.output_path == cl_path
    assert Path(cl_path).name == "cover_letter_Acme_Sr_Eng_20260622_143000.txt"


def test_artifact_basename_sanitizes_and_caps() -> None:
    base = artifact_basename(
        "A Very Long Company Name That Exceeds Thirty Characters",
        "Title/With:Specials",
        when=WHEN,
    )
    assert base.startswith("tailored_")
    assert "/" not in base and ":" not in base  # specials → "_"
    assert base.endswith("_20260622_143000")
