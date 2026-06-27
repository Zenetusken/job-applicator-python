"""Persistent job-funnel store — the head of the application pipeline.

Records jobs discovered via ``search`` and scored via ``match`` so they flow into
``tailor``/``apply`` without the user re-typing job metadata, and so ``status`` can
show where each job sits. Backed by the same local SQLite database as the
application-state store (``state.ApplicationState``), in a separate ``jobs`` table —
so a user's whole funnel lives in one file.

Division of authority: this store owns the funnel *head*
(found → matched → tailored → cover_letter). The authority for *submitted*
applications stays in ``state.ApplicationState`` (it drives the daily cap), so the
``status`` view composes the two rather than this store forking the applied state.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from job_applicator.embeddings.matching import MatchResult
from job_applicator.exceptions import JobApplicatorError
from job_applicator.models import FunnelStatus, JobBoard, JobListing, StoredJob
from job_applicator.utils.logging import get_logger

logger = get_logger("jobs_store")

DEFAULT_DB_DIR = Path.home() / ".job-applicator"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "applications.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    board TEXT NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    salary TEXT,
    seniority TEXT,
    description TEXT NOT NULL DEFAULT '',
    requirements TEXT NOT NULL DEFAULT '[]',
    match_score REAL,
    semantic_score REAL,
    skill_score REAL,
    matched_skills TEXT NOT NULL DEFAULT '[]',
    missing_skills TEXT NOT NULL DEFAULT '[]',
    funnel_status TEXT NOT NULL DEFAULT 'found',
    tailored_resume_path TEXT,
    cover_letter_path TEXT,
    pdf_path TEXT,
    cover_letter_pdf_path TEXT,
    source_query TEXT,
    first_seen_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_url ON jobs(job_url);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(funnel_status);
CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at);
"""


class JobStoreError(JobApplicatorError):
    """Raised when the job-funnel store cannot be read or written."""


def _now() -> str:
    """UTC ISO timestamp — matches ApplicationState so the shared DB compares cleanly."""
    return datetime.now(UTC).isoformat()


class JobStore:
    """SQLite-backed store for discovered/scored/tailored jobs (the funnel head)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or DEFAULT_DB_PATH
        self._ensure_dir()
        self._init_schema()

    def _ensure_dir(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise JobStoreError(
                f"Cannot create state directory {self._path.parent}: {exc}"
            ) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction, always CLOSING it.

        Mirrors ``state.ApplicationState._connect``: ``with conn`` commits/rolls back
        the transaction; the ``finally`` closes deterministically (CPython's
        connection ctx-manager commits but never closes). ``row_factory`` is set so
        rows support both index and column-name access.
        """
        try:
            conn = sqlite3.connect(str(self._path), timeout=5.0)
        except sqlite3.Error as exc:
            raise JobStoreError(f"Cannot open state database {self._path}: {exc}") from exc
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                # WAL: readers (status / the TUI) don't block on a long batch/apply writer on
                # this shared DB. Persists on the file, so setting it at init is enough.
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_CREATE_SQL)
                # Migration: older databases were created without pdf_path.
                self._migrate_add_pdf_path(conn)
                self._migrate_add_cover_letter_pdf_path(conn)
        except sqlite3.Error as exc:
            raise JobStoreError(f"Cannot initialize jobs schema: {exc}") from exc

    @staticmethod
    def _migrate_add_pdf_path(conn: sqlite3.Connection) -> None:
        """Add the ``pdf_path`` column to ``jobs`` if it is missing."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "pdf_path" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN pdf_path TEXT")

    @staticmethod
    def _migrate_add_cover_letter_pdf_path(conn: sqlite3.Connection) -> None:
        """Add the ``cover_letter_pdf_path`` column to ``jobs`` if it is missing."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "cover_letter_pdf_path" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN cover_letter_pdf_path TEXT")

    # ------------------------------------------------------------------ writes
    def upsert_job(self, job: JobListing, *, source_query: str = "") -> None:
        """Record a discovered job (from ``search``).

        Re-discovery refreshes the listing metadata but never downgrades the funnel
        stage (a job already tailored stays tailored) and never clobbers scores or
        artifact paths.
        """
        now = _now()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_url, title, company, board, location, salary, seniority,
                        description, requirements, funnel_status, source_query,
                        first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'found', ?, ?, ?)
                    ON CONFLICT(job_url) DO UPDATE SET
                        title=excluded.title,
                        company=excluded.company,
                        board=excluded.board,
                        location=excluded.location,
                        salary=COALESCE(excluded.salary, jobs.salary),
                        seniority=COALESCE(NULLIF(excluded.seniority, ''), jobs.seniority),
                        description=COALESCE(NULLIF(excluded.description, ''), jobs.description),
                        requirements=CASE
                            WHEN excluded.requirements = '[]' THEN jobs.requirements
                            ELSE excluded.requirements END,
                        source_query=COALESCE(NULLIF(excluded.source_query, ''), jobs.source_query),
                        updated_at=excluded.updated_at
                    """,
                    (
                        str(job.url),
                        job.title,
                        job.company,
                        job.board.value,
                        job.location,
                        job.salary,
                        job.seniority,
                        job.description,
                        json.dumps(job.requirements),
                        source_query,
                        now,
                        now,
                    ),
                )
        except sqlite3.Error as exc:
            raise JobStoreError(f"Cannot record job: {exc}") from exc

    def upsert_match(self, match: MatchResult, *, source_query: str = "") -> None:
        """Record a scored job (from ``match``): scores + skills, advancing to ``matched``.

        On re-match, scores/skills refresh; the stage advances to ``matched`` only if
        the job is still ``found`` — a job already ``tailored``/``cover_letter`` keeps
        its further stage.
        """
        now = _now()
        job = match.job
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_url, title, company, board, location, salary, seniority,
                        description, requirements, match_score, semantic_score, skill_score,
                        matched_skills, missing_skills, funnel_status, source_query,
                        first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'matched', ?, ?, ?)
                    ON CONFLICT(job_url) DO UPDATE SET
                        title=excluded.title,
                        company=excluded.company,
                        board=excluded.board,
                        location=excluded.location,
                        salary=COALESCE(excluded.salary, jobs.salary),
                        seniority=COALESCE(NULLIF(excluded.seniority, ''), jobs.seniority),
                        description=COALESCE(NULLIF(excluded.description, ''), jobs.description),
                        requirements=CASE
                            WHEN excluded.requirements = '[]' THEN jobs.requirements
                            ELSE excluded.requirements END,
                        match_score=excluded.match_score,
                        semantic_score=excluded.semantic_score,
                        skill_score=excluded.skill_score,
                        matched_skills=excluded.matched_skills,
                        missing_skills=excluded.missing_skills,
                        funnel_status=CASE
                            WHEN jobs.funnel_status IN ('tailored', 'cover_letter')
                            THEN jobs.funnel_status ELSE 'matched' END,
                        source_query=COALESCE(NULLIF(excluded.source_query, ''), jobs.source_query),
                        updated_at=excluded.updated_at
                    """,
                    (
                        str(job.url),
                        job.title,
                        job.company,
                        job.board.value,
                        job.location,
                        job.salary,
                        job.seniority,
                        job.description,
                        json.dumps(job.requirements),
                        match.score,
                        match.semantic_score,
                        match.skill_score,
                        json.dumps(match.matched_skills),
                        json.dumps(match.missing_skills),
                        source_query,
                        now,
                        now,
                    ),
                )
        except sqlite3.Error as exc:
            raise JobStoreError(f"Cannot record match: {exc}") from exc

    def mark_tailored(
        self,
        job: JobListing,
        *,
        tailored_resume_path: str,
        cover_letter_path: str = "",
        pdf_path: str = "",
        cover_letter_pdf_path: str = "",
    ) -> None:
        """Record that a job has been tailored (upsert + advance the funnel stage).

        Called when ``tailor`` saves an artifact for a job with a real identity (a
        stored ``--from`` job, or one given a ``--url``). Advances to ``cover_letter``
        when a cover letter was generated, else ``tailored`` — but never downgrades a
        job already at ``cover_letter`` (a re-tailor without a cover letter keeps it).
        """
        now = _now()
        status = (
            FunnelStatus.COVER_LETTER.value if cover_letter_path else FunnelStatus.TAILORED.value
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_url, title, company, board, location, salary, seniority,
                        description, requirements, funnel_status,
                        tailored_resume_path, cover_letter_path, pdf_path,
                        cover_letter_pdf_path,
                        first_seen_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_url) DO UPDATE SET
                        title=excluded.title,
                        company=excluded.company,
                        board=excluded.board,
                        location=excluded.location,
                        salary=COALESCE(excluded.salary, jobs.salary),
                        seniority=COALESCE(NULLIF(excluded.seniority, ''), jobs.seniority),
                        funnel_status=CASE
                            WHEN excluded.funnel_status = 'cover_letter'
                                OR jobs.funnel_status = 'cover_letter'
                            THEN 'cover_letter' ELSE 'tailored' END,
                        tailored_resume_path=excluded.tailored_resume_path,
                        cover_letter_path=COALESCE(
                            NULLIF(excluded.cover_letter_path, ''), jobs.cover_letter_path),
                        pdf_path=COALESCE(NULLIF(excluded.pdf_path, ''), jobs.pdf_path),
                        cover_letter_pdf_path=COALESCE(
                            NULLIF(excluded.cover_letter_pdf_path, ''), jobs.cover_letter_pdf_path),
                        updated_at=excluded.updated_at
                    """,
                    (
                        str(job.url),
                        job.title,
                        job.company,
                        job.board.value,
                        job.location,
                        job.salary,
                        job.seniority,
                        job.description,
                        json.dumps(job.requirements),
                        status,
                        tailored_resume_path,
                        cover_letter_path,
                        pdf_path,
                        cover_letter_pdf_path,
                        now,
                        now,
                    ),
                )
        except sqlite3.Error as exc:
            raise JobStoreError(f"Cannot record tailored job: {exc}") from exc

    def set_cover_letter(
        self,
        job_url: str,
        cover_letter_path: str,
        *,
        cover_letter_pdf_path: str = "",
    ) -> None:
        """Record a generated cover letter: store its path and advance the stage to
        ``cover_letter`` (the furthest head stage). The job must already exist.
        Optionally records a cover-letter PDF path as well.
        """
        now = _now()
        try:
            with self._connect() as conn:
                if cover_letter_pdf_path:
                    conn.execute(
                        "UPDATE jobs SET cover_letter_path = ?, cover_letter_pdf_path = ?, "
                        "funnel_status = 'cover_letter', updated_at = ? WHERE job_url = ?",
                        (cover_letter_path, cover_letter_pdf_path, now, job_url),
                    )
                else:
                    conn.execute(
                        "UPDATE jobs SET cover_letter_path = ?, funnel_status = 'cover_letter', "
                        "updated_at = ? WHERE job_url = ?",
                        (cover_letter_path, now, job_url),
                    )
        except sqlite3.Error as exc:
            raise JobStoreError(f"Cannot record cover letter: {exc}") from exc

    # ------------------------------------------------------------------- reads
    def get(self, ref: str) -> StoredJob | None:
        """Resolve a stored job by numeric id or exact job URL (for ``--from``)."""
        ref = ref.strip()
        if not ref:
            return None
        try:
            with self._connect() as conn:
                if ref.isdigit():
                    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (int(ref),)).fetchone()
                else:
                    row = conn.execute("SELECT * FROM jobs WHERE job_url = ?", (ref,)).fetchone()
        except sqlite3.Error as exc:
            raise JobStoreError(f"Cannot read job: {exc}") from exc
        if row is None:
            return None
        try:
            return self._row_to_stored(row)
        except Exception as exc:  # corrupt / enum-drift row → typed error, not a raw crash
            raise JobStoreError(f"Could not read stored job {ref!r}: {exc}") from exc

    def list_jobs(
        self,
        *,
        status: FunnelStatus | None = None,
        board: str | None = None,
        limit: int = 50,
    ) -> list[StoredJob]:
        """Return stored jobs, newest-updated first, optionally filtered by stage/board.

        Filters are applied in SQL *before* the LIMIT, so a board filter sees every
        matching row — not just the most-recent ``limit`` rows of any board.
        """
        sql = "SELECT * FROM jobs"
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("funnel_status = ?")
            params.append(status.value)
        if board is not None:
            clauses.append("board = ?")
            params.append(board)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Recency-first ("Recent jobs"), with match_score DESC as a tiebreak so jobs
        # scored in the SAME batch (identical updated_at) read best-first rather than in
        # arbitrary insertion order. NULL (unscored) scores sort last under DESC.
        sql += " ORDER BY updated_at DESC, match_score DESC LIMIT ?"
        params.append(limit)
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise JobStoreError(f"Cannot list jobs: {exc}") from exc

        out: list[StoredJob] = []
        for row in rows:
            try:
                out.append(self._row_to_stored(row))
            except Exception as exc:  # a single corrupt row must not break the listing
                logger.warning("Skipping corrupt job row: %s", exc)
        return out

    @staticmethod
    def _row_to_stored(row: sqlite3.Row) -> StoredJob:
        job = JobListing(
            title=row["title"],
            company=row["company"],
            url=row["job_url"],
            description=row["description"],
            location=row["location"],
            salary=row["salary"],
            requirements=json.loads(row["requirements"]),
            board=JobBoard(row["board"]),
            seniority=row["seniority"],
        )
        return StoredJob(
            id=int(row["id"]),
            job=job,
            funnel_status=FunnelStatus(row["funnel_status"]),
            match_score=row["match_score"],
            semantic_score=row["semantic_score"],
            skill_score=row["skill_score"],
            matched_skills=json.loads(row["matched_skills"]),
            missing_skills=json.loads(row["missing_skills"]),
            tailored_resume_path=row["tailored_resume_path"] or "",
            cover_letter_path=row["cover_letter_path"] or "",
            pdf_path=row["pdf_path"] or "",
            cover_letter_pdf_path=row["cover_letter_pdf_path"] or "",
            source_query=row["source_query"] or "",
            first_seen_at=datetime.fromisoformat(row["first_seen_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
