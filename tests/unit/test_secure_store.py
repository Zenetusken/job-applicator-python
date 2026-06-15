"""Tests for secure secret-file writing."""

from __future__ import annotations

import json
import os
from pathlib import Path

from job_applicator.utils.secure_store import write_secret_json


def test_write_secret_json_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "cookies" / "linkedin.json"
    write_secret_json(path, {"cookies": [{"name": "li_at", "value": "x"}]})

    assert json.loads(path.read_text())["cookies"][0]["name"] == "li_at"
    assert (path.stat().st_mode & 0o077) == 0  # file 0600 (no group/other)
    assert (path.parent.stat().st_mode & 0o077) == 0  # dir 0700


def test_write_secret_json_overwrites_atomically(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    write_secret_json(path, {"cookies": [1]})
    write_secret_json(path, {"cookies": [2]})

    assert json.loads(path.read_text())["cookies"] == [2]
    # No leftover temp files from the atomic-replace dance.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.json"]


def test_write_secret_json_replaces_symlink_not_target(tmp_path: Path) -> None:
    """A symlink planted at the path must be replaced, not followed (no write
    redirected into the link target)."""
    target = tmp_path / "attacker_target.json"
    target.write_text("untouched")
    link = tmp_path / "cookies.json"
    os.symlink(target, link)

    write_secret_json(link, {"cookies": ["secret"]})

    assert target.read_text() == "untouched"  # target not overwritten
    assert not link.is_symlink()  # link replaced by a real file
    assert json.loads(link.read_text())["cookies"] == ["secret"]
