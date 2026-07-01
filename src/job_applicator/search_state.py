"""Persistent search-volume store — a proactive daily search cap + inter-search cooldown on
scraping.

Anti-detection rationale: the scraper reuses a human-authenticated LinkedIn session, so the board
already knows the account — the achievable goal is not "undetected" but **unremarkable** (low search
volume + velocity). The apply path already caps *submissions* per day (``ApplicationState``); this
is the same discipline for *searches*, which were otherwise uncapped in code (the top risk from the
2026-07-01 anti-detection audit). Backed by the same SQLite DB as the other local stores.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from job_applicator.exceptions import JobApplicatorError
from job_applicator.utils.logging import get_logger

logger = get_logger("search_state")

DEFAULT_DB_DIR = Path.home() / ".job-applicator"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "applications.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    board TEXT NOT NULL,
    query TEXT NOT NULL,
    searched_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_searches_searched_at ON searches(searched_at);
"""


class SearchStateError(JobApplicatorError):
    """Raised when the search-volume store cannot be read or written."""


class SearchState:
    """SQLite-backed store for scrape/search volume — the daily cap + inter-search cooldown."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or DEFAULT_DB_PATH
        self._ensure_dir()
        self._init_schema()

    def _ensure_dir(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SearchStateError(
                f"Cannot create state directory {self._path.parent}: {exc}"
            ) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(str(self._path), timeout=5.0)
        except sqlite3.Error as exc:
            raise SearchStateError(f"Cannot open state database {self._path}: {exc}") from exc
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")  # readers don't block on a writer
                conn.executescript(_CREATE_SQL)
        except sqlite3.Error as exc:
            raise SearchStateError(f"Cannot initialize search schema: {exc}") from exc

    def record(self, board: str, query: str) -> None:
        """Log one search against the daily budget."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO searches (board, query, searched_at) VALUES (?, ?, ?)",
                    (board, query, datetime.now(UTC).isoformat()),
                )
        except sqlite3.Error as exc:
            raise SearchStateError(f"Cannot record search: {exc}") from exc

    def count_today(self, board: str | None = None) -> int:
        """Count searches since UTC midnight (start of the UTC day), optionally for one board."""
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        params: list[Any] = [today.isoformat()]
        sql = "SELECT COUNT(*) FROM searches WHERE searched_at >= ?"
        if board:
            sql += " AND board = ?"
            params.append(board)
        try:
            with self._connect() as conn:
                row = conn.execute(sql, params).fetchone()
                return row[0] if row else 0
        except sqlite3.Error as exc:
            raise SearchStateError(f"Cannot count searches: {exc}") from exc

    def seconds_since_last(self, board: str | None = None) -> float | None:
        """Seconds since the most recent search (for the inter-search cooldown), or ``None`` if
        there is no prior search."""
        params: list[Any] = []
        sql = "SELECT MAX(searched_at) FROM searches"
        if board:
            sql += " WHERE board = ?"
            params.append(board)
        try:
            with self._connect() as conn:
                row = conn.execute(sql, params).fetchone()
        except sqlite3.Error as exc:
            raise SearchStateError(f"Cannot read the last search time: {exc}") from exc
        if not row or not row[0]:
            return None
        try:
            last = datetime.fromisoformat(row[0])
        except (
            ValueError,
            TypeError,
        ) as exc:  # a corrupt/foreign timestamp → typed, not a raw crash
            raise SearchStateError(f"Corrupt stored search timestamp {row[0]!r}: {exc}") from exc
        return (datetime.now(UTC) - last).total_seconds()
