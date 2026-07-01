"""SearchState — the per-day search-volume cap + inter-search cooldown store (anti-detection H1)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from job_applicator.search_state import SearchState


def test_record_and_count_today(tmp_path: Path) -> None:
    s = SearchState(db_path=tmp_path / "s.db")
    assert s.count_today() == 0
    s.record("linkedin", "python")
    s.record("linkedin", "rust")
    assert s.count_today() == 2


def test_count_today_filters_board(tmp_path: Path) -> None:
    s = SearchState(db_path=tmp_path / "s.db")
    s.record("linkedin", "a")
    s.record("indeed", "b")
    assert s.count_today("linkedin") == 1
    assert s.count_today("indeed") == 1
    assert s.count_today() == 2  # both, unfiltered


def test_count_today_excludes_prior_days(tmp_path: Path) -> None:
    """The cap is a DAILY budget — searches before UTC midnight must not count against today."""
    db = tmp_path / "s.db"
    s = SearchState(db_path=db)
    s.record("linkedin", "today")
    yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO searches (board, query, searched_at) VALUES (?, ?, ?)",
            ("linkedin", "old", yesterday),
        )
    assert s.count_today() == 1  # only today's search, not yesterday's


def test_seconds_since_last(tmp_path: Path) -> None:
    s = SearchState(db_path=tmp_path / "s.db")
    assert s.seconds_since_last() is None  # no prior search
    s.record("linkedin", "python")
    since = s.seconds_since_last("linkedin")
    assert since is not None and 0 <= since < 5
    assert s.seconds_since_last("indeed") is None  # board-scoped
