"""Unit tests for cookie read/write (utils/cookies.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_applicator.exceptions import CookieError
from job_applicator.utils.cookies import read_cookies, save_cookies


def test_read_cookies_missing_file_returns_empty(tmp_path: Path) -> None:
    """A genuinely absent cookie file is a legitimate empty (no seeded session), not a failure."""
    assert read_cookies(tmp_path / "nope.json") == []


def test_read_cookies_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    save_cookies(p, [{"name": "li_at", "value": "x"}])
    assert read_cookies(p) == [{"name": "li_at", "value": "x"}]


def test_read_cookies_corrupt_file_raises(tmp_path: Path) -> None:
    """A present-but-corrupt cookie file must RAISE CookieError — never silently return [] (which
    degrades a seeded LinkedIn session to unauthenticated and misleads the user into re-logging
    in when the file was merely unreadable)."""
    p = tmp_path / "c.json"
    p.write_text("{ this is not valid json")
    with pytest.raises(CookieError):
        read_cookies(p)


def test_read_cookies_wrong_shape_raises(tmp_path: Path) -> None:
    """A valid-JSON file missing the 'cookies' envelope is malformed → raise, not silently []."""
    p = tmp_path / "c.json"
    p.write_text('{"foo": "bar"}')
    with pytest.raises(CookieError):
        read_cookies(p)
