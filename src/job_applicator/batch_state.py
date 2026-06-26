"""Persistent progress store for batch runs.

Lets a long-running ``batch`` command resume after a crash instead of
re-tailoring every job from scratch.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from job_applicator.exceptions import JobApplicatorError
from job_applicator.models import BatchRunSpec, JobBoard, JobListing
from job_applicator.utils.logging import get_logger

logger = get_logger("batch_state")

DEFAULT_DB_DIR = Path.home() / ".job-applicator"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "applications.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS batch_runs (
    run_id TEXT PRIMARY KEY,
    site TEXT NOT NULL,
    query TEXT,
    jobs_file TEXT,
    resume_path TEXT NOT NULL,
    top_k INTEGER,
    min_score REAL,
    cover_letter BOOLEAN,
    status TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_jobs (
    run_id TEXT NOT NULL,
    job_url TEXT NOT NULL,
    title TEXT,
    company TEXT,
    board TEXT NOT NULL,
    status TEXT NOT NULL,
    resume_path TEXT,
    cover_letter_path TEXT,
    pdf_path TEXT,
    error_message TEXT,
    updated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, job_url)
);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_run_id ON batch_jobs(run_id);
CREATE INDEX IF NOT EXISTS idx_batch_runs_status ON batch_runs(status);
"""


class BatchStateError(JobApplicatorError):
    """Raised when the batch state store cannot be read or written."""


class BatchRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchJobStatus(StrEnum):
    PENDING = "pending"
    TAILORED = "tailored"
    COVER_LETTER = "cover_letter"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BatchState:
    """SQLite-backed store for batch-run progress."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or DEFAULT_DB_PATH
        self._ensure_dir()
        self._init_schema()

    def _ensure_dir(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BatchStateError(
                f"Cannot create state directory {self._path.parent}: {exc}"
            ) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction, always CLOSING it.

        ``with sqlite3.connect(...) as conn`` commits on exit but never closes the
        connection (it relies on CPython refcount GC). Wrapping it here closes
        deterministically while preserving the commit/rollback transaction —
        important for a store hit many times during a batch run.
        """
        try:
            conn = sqlite3.connect(str(self._path), timeout=5.0)
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot open state database {self._path}: {exc}") from exc
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(_CREATE_SQL)
                # Migration: older databases were created without pdf_path.
                self._migrate_add_pdf_path(conn)
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot initialize batch schema: {exc}") from exc

    @staticmethod
    def _migrate_add_pdf_path(conn: sqlite3.Connection) -> None:
        """Add the ``pdf_path`` column to ``batch_jobs`` if it is missing."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(batch_jobs)")}
        if "pdf_path" not in columns:
            conn.execute("ALTER TABLE batch_jobs ADD COLUMN pdf_path TEXT")

    def start_run(self, spec: BatchRunSpec, *, run_id: str, reset: bool = True) -> str:
        """Create or reset a batch run record. Returns the run_id."""
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO batch_runs (
                        run_id, site, query, jobs_file, resume_path, top_k, min_score,
                        cover_letter, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status=excluded.status,
                        updated_at=excluded.updated_at,
                        top_k=excluded.top_k,
                        min_score=excluded.min_score,
                        cover_letter=excluded.cover_letter
                    """,
                    (
                        run_id,
                        spec.site,
                        spec.query,
                        spec.jobs_file,
                        spec.resume_path,
                        spec.top_k,
                        spec.min_score,
                        spec.cover_letter,
                        BatchRunStatus.RUNNING,
                        now,
                        now,
                    ),
                )
                if reset:
                    conn.execute("DELETE FROM batch_jobs WHERE run_id = ?", (run_id,))
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot start batch run: {exc}") from exc
        return run_id

    def find_existing_run(self, spec: BatchRunSpec) -> str | None:
        """Return the most recent incomplete run_id matching the spec.

        Matches every spec identity field — the same source as ``BatchRunSpec.run_id()`` —
        so a resume can't bind a run created with different processing params and then
        silently adopt the new ones.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT run_id FROM batch_runs
                    WHERE site = ? AND query IS ? AND jobs_file IS ? AND resume_path = ?
                      AND top_k = ? AND min_score = ? AND cover_letter = ?
                      AND status = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (
                        spec.site,
                        spec.query,
                        spec.jobs_file,
                        spec.resume_path,
                        spec.top_k,
                        spec.min_score,
                        spec.cover_letter,
                        BatchRunStatus.RUNNING,
                    ),
                ).fetchone()
                return row[0] if row else None
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot search batch runs: {exc}") from exc

    def record_job(
        self,
        run_id: str,
        job: JobListing,
        status: BatchJobStatus,
        *,
        resume_path: str | None = None,
        cover_letter_path: str | None = None,
        pdf_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Persist the status of a single job within a batch run."""
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO batch_jobs (
                        run_id, job_url, title, company, board, status, resume_path,
                        cover_letter_path, pdf_path, error_message, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, job_url) DO UPDATE SET
                        status=excluded.status,
                        board=excluded.board,
                        resume_path=excluded.resume_path,
                        cover_letter_path=excluded.cover_letter_path,
                        pdf_path=excluded.pdf_path,
                        error_message=excluded.error_message,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        str(job.url),
                        job.title,
                        job.company,
                        job.board.value,
                        status,
                        resume_path,
                        cover_letter_path,
                        pdf_path,
                        error_message,
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE batch_runs SET updated_at = ? WHERE run_id = ?",
                    (now, run_id),
                )
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot record batch job: {exc}") from exc

    def get_job_status(self, run_id: str, url: str) -> BatchJobStatus | None:
        """Return the persisted status for a job, or None if not recorded."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM batch_jobs WHERE run_id = ? AND job_url = ?",
                    (run_id, str(url)),
                ).fetchone()
                return BatchJobStatus(row[0]) if row else None
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot read batch job status: {exc}") from exc

    def get_job(self, run_id: str, url: str) -> tuple[BatchJobStatus, str | None] | None:
        """Return (status, resume_path) for a persisted job, or None if not recorded.

        Used by mid-job resume to find a TAILORED job whose tailored artifact can be
        reused instead of re-tailoring.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT status, resume_path FROM batch_jobs WHERE run_id = ? AND job_url = ?",
                    (run_id, str(url)),
                ).fetchone()
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot read batch job: {exc}") from exc
        if row is None:
            return None
        return BatchJobStatus(row[0]), row[1]

    def list_completed_jobs(self, run_id: str) -> list[str]:
        """Return job URLs that are fully completed or explicitly skipped.

        TAILORED is intentionally excluded: a crash after tailoring but before
        the cover letter would leave the job half-done, and resuming must
        re-process it so the cover letter is generated.
        """
        completed = {
            BatchJobStatus.COMPLETED,
            BatchJobStatus.SKIPPED,
        }
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT job_url FROM batch_jobs
                    WHERE run_id = ? AND status IN (?, ?)
                    """,
                    (run_id, *(s.value for s in completed)),
                ).fetchall()
                return [r[0] for r in rows]
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot list completed batch jobs: {exc}") from exc

    def complete_run(self, run_id: str, status: BatchRunStatus = BatchRunStatus.COMPLETED) -> None:
        """Mark a batch run as completed or failed."""
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE batch_runs SET status = ?, updated_at = ? WHERE run_id = ?",
                    (status, now, run_id),
                )
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot complete batch run: {exc}") from exc

    def load_run_jobs(self, run_id: str) -> list[JobListing]:
        """Load the jobs recorded for a run (best-effort reconstruction)."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT job_url, title, company, board, status FROM batch_jobs
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise BatchStateError(f"Cannot load batch run jobs: {exc}") from exc

        jobs: list[JobListing] = []
        for url, title, company, board, _status in rows:
            try:
                jobs.append(
                    JobListing(
                        title=title or "",
                        company=company or "",
                        url=url,
                        board=JobBoard(board),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping corrupt batch job row: %s", exc)
        return jobs
