#!/usr/bin/env python3
"""Live end-to-end tests for cover-letter sign-off enforcement.

These tests exercise the real CLI against a live vLLM endpoint to empirically
validate that every generated cover letter closes with a recognized sign-off
and a signature matching the applicant.

All tests are marked ``live`` and are skipped automatically when the vLLM
endpoint at ``localhost:8000`` is not reachable.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.documents.sign_off import _SIGN_OFFS, extract_sign_off

pytestmark = pytest.mark.live


RESUME_TEXT = """\
Sam Sample
sam.sample@example.com
555-0123

Summary
-------
Customer-focused IT support technician with 4 years of experience troubleshooting
Windows, Office 365, and hardware issues across phone, chat, and email channels.

Skills
------
Windows, Office 365, Hardware diagnostics, Customer service

Experience
----------
IT Support Specialist | HelpDesk Co | 2020-present
- Resolved tier-1 and tier-2 support tickets via phone, chat, and email.
- Diagnosed Windows and Office 365 issues for a 500-employee organization.

Education
---------
A.S. Information Technology, Example College
"""

JOB_DESCRIPTION = """\
We are seeking an IT Support Specialist to provide tier-1 and tier-2 technical
support via phone, chat, and email. Troubleshoot Windows, Office 365, and
hardware issues.
"""


def _write_resume(tmp_path: Path) -> Path:
    path = tmp_path / "resume.txt"
    path.write_text(RESUME_TEXT, encoding="utf-8")
    return path


def _is_vllm_up() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _signature_contains_name(signature: str, name: str) -> bool:
    """Token-level name check (e.g. ``Sam`` is not ``Samantha``)."""
    tokens = set(re.findall(r"[a-z0-9]+", signature.lower()))
    return all(token in tokens for token in name.lower().split())


@pytest.fixture
def resume_path(tmp_path: Path) -> Path:
    return _write_resume(tmp_path)


@pytest.mark.skipif(not _is_vllm_up(), reason="vLLM endpoint not reachable")
def test_generate_cover_letter_sign_off_matches_resume_name(resume_path: Path) -> None:
    """A real LLM call produces a cover letter signed with the parsed résumé name."""
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "generate-cover-letter",
            "--resume",
            str(resume_path),
            "--job-title",
            "IT Support Specialist",
            "--company",
            "TechCorp",
            "--description",
            JOB_DESCRIPTION,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    letter: str = parsed["cover_letter"]

    extracted = extract_sign_off(letter)
    assert extracted is not None, "Cover letter has no recognized sign-off"
    closing, signature = extracted
    assert closing in _SIGN_OFFS
    assert _signature_contains_name(signature, "Sam Sample")


@pytest.mark.skipif(not _is_vllm_up(), reason="vLLM endpoint not reachable")
def test_generate_cover_letter_sign_off_uses_profile_name_override(resume_path: Path) -> None:
    """When ``profile_name`` is configured, the signature uses it instead of the résumé name."""
    runner = CliRunner(env={**os.environ, "JOB_APPLICATOR_PROFILE_NAME": "Alexandra Quinn"})
    result = runner.invoke(
        cli.app,
        [
            "generate-cover-letter",
            "--resume",
            str(resume_path),
            "--job-title",
            "IT Support Specialist",
            "--company",
            "TechCorp",
            "--description",
            JOB_DESCRIPTION,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    letter: str = parsed["cover_letter"]

    extracted = extract_sign_off(letter)
    assert extracted is not None
    signature = extracted[1]
    assert _signature_contains_name(signature, "Alexandra Quinn")
    assert not _signature_contains_name(signature, "Sam Sample")
