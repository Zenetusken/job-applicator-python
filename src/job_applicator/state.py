"""Persistent application-state store.

Tracks which jobs have been applied to, when, and with what outcome. Backed by a
local SQLite database so the state survives restarts and prevents duplicate
applications.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from job_applicator.exceptions import JobApplicatorError
from job_applicator.models import ApplicationResult, ApplicationStatus
from job_applicator.utils.logging import get_logger

logger = get_logger("state")

DEFAULT_DB_DIR = Path.home() / ".job-applicator"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "applications.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    board TEXT NOT NULL,
    status TEXT NOT NULL,
    applied_at TIMESTAMP NOT NULL,
    cover_letter_path TEXT,
    error_message TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_applications_url ON applications(job_url);
CREATE INDEX IF NOT EXISTS idx_applications_applied_at ON applications(applied_at);
"""


class StateError(JobApplicatorError):
    """Raised when the application state store cannot be read or written."""


class ApplicationState:
    """SQLite-backed store for application attempts."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or DEFAULT_DB_PATH
        self._ensure_dir()
        self._init_schema()

    def _ensure_dir(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StateError(f"Cannot create state directory {self._path.parent}: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        try:
            return sqlite3.connect(str(self._path), timeout=5.0)
        except sqlite3.Error as exc:
            raise StateError(f"Cannot open state database {self._path}: {exc}") from exc

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(_CREATE_SQL)
        except sqlite3.Error as exc:
            raise StateError(f"Cannot initialize state schema: {exc}") from exc

    def record(self, result: ApplicationResult, cover_letter_path: str | None = None) -> None:
        """Persist an application attempt, upserting on job URL."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO applications (
                        job_url, title, company, board, status, applied_at,
                        cover_letter_path, error_message, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_url) DO UPDATE SET
                        status=excluded.status,
                        applied_at=excluded.applied_at,
                        cover_letter_path=excluded.cover_letter_path,
                        error_message=excluded.error_message,
                        notes=excluded.notes
                    """,
                    (
                        str(result.job.url),
                        result.job.title,
                        result.job.company,
                        result.job.board.value,
                        result.status.value,
                        result.timestamp.isoformat(),
                        cover_letter_path,
                        result.error_message,
                        result.notes,
                    ),
                )
        except sqlite3.Error as exc:
            raise StateError(f"Cannot record application state: {exc}") from exc

    def has_applied(
        self,
        url: str,
        *,
        statuses: set[ApplicationStatus] | None = None,
        since: datetime | None = None,
    ) -> bool:
        """Return True if an application for ``url`` exists matching the filters."""
        if statuses is None:
            statuses = {ApplicationStatus.SUBMITTED}
        status_placeholders = ", ".join(["?"] * len(statuses))
        status_values = [s.value for s in statuses]
        params: list[Any] = [url]
        since_clause = ""
        if since is not None:
            since_clause = " AND applied_at >= ?"
            params.append(since.isoformat())
        # Parameter order matches: url, [since], status_values.
        sql = (
            "SELECT 1 FROM applications WHERE job_url = ?"
            + since_clause
            + " AND status IN ("
            + status_placeholders
            + ") LIMIT 1"
        )
        try:
            with self._connect() as conn:
                row = conn.execute(sql, params + status_values).fetchone()
                return row is not None
        except sqlite3.Error as exc:
            raise StateError(f"Cannot read application state: {exc}") from exc

    def count_today(self, board: str | None = None) -> int:
        """Count applications recorded since local midnight."""
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        params: list[Any] = [today.isoformat()]
        board_clause = ""
        if board:
            board_clause = " AND board = ?"
            params.append(board)
        sql = "SELECT COUNT(*) FROM applications WHERE applied_at >= ?" + board_clause
        try:
            with self._connect() as conn:
                row = conn.execute(sql, params).fetchone()
                return row[0] if row else 0
        except sqlite3.Error as exc:
            raise StateError(f"Cannot count applications: {exc}") from exc

    def list_recent(self, limit: int = 50) -> list[ApplicationResult]:
        """Return the most recent application results (newest first)."""
        from job_applicator.models import JobBoard, JobListing

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT job_url, title, company, board, status, applied_at,
                           error_message, notes
                    FROM applications
                    ORDER BY applied_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise StateError(f"Cannot list applications: {exc}") from exc

        results: list[ApplicationResult] = []
        for row in rows:
            try:
                job = JobListing(
                    title=row[1],
                    company=row[2],
                    url=row[0],
                    board=JobBoard(row[3]),
                )
                results.append(
                    ApplicationResult(
                        job=job,
                        status=ApplicationStatus(row[4]),
                        timestamp=datetime.fromisoformat(row[5]),
                        error_message=row[6],
                        notes=row[7] or "",
                    )
                )
            except Exception as exc:
                logger.warning("Skipping corrupt application row: %s", exc)
        return results
