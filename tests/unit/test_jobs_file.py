"""Tests for `_load_jobs_file` — every bad jobs-file input must raise a clean typed
`DocumentError` (a `JobApplicatorError`), so the `match`/`batch` commands' `except
JobApplicatorError` handler renders a one-line message instead of a raw traceback.

Regression guard for QA-pass-2 finding B6: `match --jobs-file <bad>` dumped a raw Python
traceback for all four bad inputs (no inner handler → the outer `except Exception` re-raised).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_applicator.cli import _load_jobs_file
from job_applicator.exceptions import DocumentError, JobApplicatorError


def _write(tmp_path: Path, name: str, content: object) -> str:
    p = tmp_path / name
    p.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return str(p)


def test_load_jobs_file_happy(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "jobs.json",
        [
            {
                "title": "Eng",
                "company": "Acme",
                "url": "https://example.com/1",
                "board": "linkedin",
            },
            {
                "title": "Dev",
                "company": "Globex",
                "url": "https://example.com/2",
                "board": "indeed",
            },
        ],
    )
    jobs = _load_jobs_file(f)
    assert [j.title for j in jobs] == ["Eng", "Dev"]


def test_load_jobs_file_missing(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="not found"):
        _load_jobs_file(str(tmp_path / "nope.json"))


def test_load_jobs_file_malformed_json(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="not valid JSON"):
        _load_jobs_file(_write(tmp_path, "bad.json", "{ not valid json "))


def test_load_jobs_file_not_a_list(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="must be a JSON array"):
        _load_jobs_file(_write(tmp_path, "obj.json", {"not": "a list"}))


def test_load_jobs_file_entry_not_object(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="not a job object"):
        _load_jobs_file(_write(tmp_path, "strs.json", ["just a string"]))


def test_load_jobs_file_missing_fields(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="invalid/missing fields"):
        _load_jobs_file(_write(tmp_path, "partial.json", [{"title": "Orphan Job"}]))


def test_load_jobs_file_directory(tmp_path: Path) -> None:
    """A directory path raises IsADirectoryError (OSError family, NOT FileNotFoundError) —
    must still be a clean DocumentError (gate-2a escape: only FileNotFoundError was caught)."""
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(DocumentError, match="Could not read"):
        _load_jobs_file(str(d))


def test_load_jobs_file_non_utf8(tmp_path: Path) -> None:
    """Non-UTF-8 bytes raise UnicodeDecodeError (a ValueError, NOT an OSError) — must still be
    a clean DocumentError (gate-2a escape: disjoint from the OSError widening)."""
    p = tmp_path / "latin1.json"
    p.write_bytes(b"\xff\xfe not utf-8 at all \x80\x81")
    with pytest.raises(DocumentError, match="Could not read"):
        _load_jobs_file(str(p))


def test_load_jobs_file_errors_are_typed(tmp_path: Path) -> None:
    """Every failure is a JobApplicatorError → caught by the commands' clean handler (B6)."""
    for blob in ("{ bad ", json.dumps({"x": 1}), json.dumps([{"title": "x"}])):
        with pytest.raises(JobApplicatorError):
            _load_jobs_file(_write(tmp_path, "f.json", blob))
