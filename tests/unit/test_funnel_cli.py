"""CLI integration tests for the job-funnel backbone.

Covers ``status`` (dashboard + JSON purity), search/match persistence into the store,
and the ``apply --from`` / saved-list paths — including the account-safety invariant
that an unresolved/empty target fails BEFORE any browser is constructed.

All tests isolate state (real stores on ``tmp_path``, or a mocked store) and mock the
browser/scraper/applicator/LLM — they never touch the real account or DB.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

import job_applicator.cli as cli
from job_applicator.cli import _resolve_output_format, _stage_label
from job_applicator.config import AppSettings
from job_applicator.embeddings.matching import MatchResult
from job_applicator.jobs_store import JobStore
from job_applicator.models import ApplicationResult, ApplicationStatus, Format, JobBoard, JobListing


def _job(n: int = 1) -> JobListing:
    return JobListing(
        title=f"Engineer {n}",
        company=f"Co{n}",
        url=f"https://linkedin.com/jobs/{n}",
        description="async pipelines",
        location="Remote",
        requirements=["python"],
        board=JobBoard.LINKEDIN,
    )


def _match(job: JobListing, score: float = 0.9) -> MatchResult:
    return MatchResult(
        job=job,
        score=score,
        semantic_score=score,
        skill_score=score,
        matched_skills=["python"],
        missing_skills=[],
        summary="ok",
    )


def _browser_cm() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


# --------------------------------------------------------------------- helpers
def test_resolve_output_format_uses_cli_value() -> None:
    settings = AppSettings(output={"default_format": "pdf"})  # type: ignore[arg-type]
    assert _resolve_output_format(Format.TXT, settings) is Format.TXT


def test_resolve_output_format_falls_back_to_config() -> None:
    settings = AppSettings(output={"default_format": "pdf"})  # type: ignore[arg-type]
    assert _resolve_output_format(None, settings) is Format.PDF


def test_resolve_output_format_falls_back_to_txt_on_bad_config() -> None:
    from unittest.mock import MagicMock

    settings = MagicMock()
    settings.output.default_format = "invalid"
    assert _resolve_output_format(None, settings) is Format.TXT


def test_stage_label_pluralizes_cover_letter() -> None:
    assert _stage_label("cover_letter", 1) == "cover letter"
    assert _stage_label("cover_letter", 2) == "cover letters"
    assert _stage_label("matched", 2) == "matched"


# --------------------------------------------------------------------- status
def test_status_json_composes_both_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    js = JobStore(db_path=tmp_path / "applications.db")
    js.upsert_job(_job(1))  # found
    js.mark_tailored(_job(2), tailored_resume_path="/out/2.txt")  # tailored
    from job_applicator.state import ApplicationState

    # Both stores share ONE db file (the production layout: jobs + applications tables
    # co-located in applications.db), so this exercises the real single-file composition.
    st = ApplicationState(db_path=tmp_path / "applications.db")
    st.record(
        ApplicationResult(
            job=_job(3), status=ApplicationStatus.SUBMITTED, timestamp=datetime.now(UTC)
        )
    )

    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    monkeypatch.setattr("job_applicator.state.ApplicationState", lambda *a, **k: st)

    result = CliRunner().invoke(cli.app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)  # raises if Rich leaked onto stdout
    assert data["counts"]["found"] == 1
    assert data["counts"]["tailored"] == 1
    assert data["counts"]["applied"] == 1
    assert data["total"] == 3
    stages = {r["stage"] for r in data["recent"]}
    assert {"found", "tailored", "applied"} <= stages


def test_status_table_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    js = JobStore(db_path=tmp_path / "applications.db")
    js.upsert_job(_job(1))
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    monkeypatch.setattr(
        "job_applicator.state.ApplicationState",
        lambda *a, **k: MagicMock(list_recent=lambda limit=0: []),
    )
    result = CliRunner().invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "Funnel" in result.stdout
    assert "Engineer 1" in result.stdout


def test_status_empty_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    js = JobStore(db_path=tmp_path / "applications.db")
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    monkeypatch.setattr(
        "job_applicator.state.ApplicationState",
        lambda *a, **k: MagicMock(list_recent=lambda limit=0: []),
    )
    result = CliRunner().invoke(cli.app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["total"] == 0
    assert data["counts"] == {
        "found": 0,
        "matched": 0,
        "tailored": 0,
        "cover_letter": 0,
        "applied": 0,
    }


# --------------------------------------------------------- search persistence
def test_search_persists_found_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = [_job(1), _job(2)]
    scraper = MagicMock(scrape=AsyncMock(return_value=jobs))
    store = MagicMock()
    monkeypatch.setattr(cli, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(cli, "_make_scraper", lambda *a, **k: scraper)
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: store)

    result = CliRunner().invoke(cli.app, ["search", "-q", "python", "--json"])
    assert result.exit_code == 0, result.output
    assert store.upsert_job.call_count == 2
    # the query is threaded through as source_query
    assert store.upsert_job.call_args_list[0].kwargs.get("source_query") == "python"


# ---------------------------------------------------------- match persistence
def test_match_persists_scored_jobs(
    tmp_path: Path, sample_resume: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(
        json.dumps([{"title": "E1", "company": "C1", "url": "https://x/1", "board": "linkedin"}])
    )
    matches = [_match(_job(1)), _match(_job(2))]
    matcher_cls = MagicMock()
    matcher_cls.return_value.rank_jobs = AsyncMock(return_value=matches)
    store = MagicMock()

    monkeypatch.setattr("job_applicator.embeddings.matching.JobMatcher", matcher_cls)
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: store)
    loader = MagicMock()
    loader.load.return_value = sample_resume  # a real ResumeData → ATSChecker is happy
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)

    result = CliRunner().invoke(
        cli.app, ["match", "--resume", "/tmp/r.txt", "--jobs-file", str(jobs_file), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert store.upsert_match.call_count == 2
    assert len(json.loads(result.stdout)) == 2  # JSON stays pure on stdout


# --------------------------------------------------- apply --from / saved-list
def _patch_apply_browser(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Patch the apply command's factories; return (make_browser_mock, applicator)."""
    applicator = MagicMock()

    async def _apply(job, letter=None, submit=False):  # type: ignore[no-untyped-def]
        return ApplicationResult(
            job=job, status=ApplicationStatus.PENDING, timestamp=datetime.now(UTC)
        )

    applicator.apply = AsyncMock(side_effect=_apply)
    make_browser = MagicMock(return_value=_browser_cm())
    monkeypatch.setattr(cli, "_make_browser", make_browser)
    monkeypatch.setattr(cli, "_make_scraper", MagicMock())
    monkeypatch.setattr(cli, "_make_applicator", lambda *a, **k: applicator)
    monkeypatch.setattr(
        "job_applicator.workflows.apply.ApplicationState",
        lambda *a, **k: MagicMock(
            **{"has_applied.return_value": False, "count_today.return_value": 0}
        ),
    )
    return make_browser, applicator


def test_apply_from_unknown_fails_before_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from with no matching stored job must error BEFORE launching a browser."""
    js = JobStore(db_path=tmp_path / "applications.db")  # empty
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    make_browser, applicator = _patch_apply_browser(monkeypatch)

    result = CliRunner().invoke(cli.app, ["apply", "--from", "999"])
    assert result.exit_code == 1, result.output
    assert "No stored job matches" in result.output
    make_browser.assert_not_called()  # account-safety: no browser on a bad target
    applicator.apply.assert_not_awaited()


def test_apply_empty_saved_list_fails_before_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bare `apply` with an empty store must error BEFORE launching a browser."""
    js = JobStore(db_path=tmp_path / "applications.db")  # empty
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    make_browser, _ = _patch_apply_browser(monkeypatch)

    result = CliRunner().invoke(cli.app, ["apply"])
    assert result.exit_code == 1, result.output
    assert "No saved" in result.output and "jobs to apply" in result.output
    make_browser.assert_not_called()


def test_apply_from_stored_job_applies_without_searching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from a stored job applies (dry-run) to exactly that job and never scrapes."""
    js = JobStore(db_path=tmp_path / "applications.db")
    js.upsert_job(_job(1))  # id 1
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    make_browser, applicator = _patch_apply_browser(monkeypatch)

    result = CliRunner().invoke(cli.app, ["apply", "--from", "1", "--no-cover-letter"])
    assert result.exit_code == 0, result.output
    make_browser.assert_called_once()  # browser IS needed to fill the form
    cli._make_scraper.assert_not_called()  # type: ignore[attr-defined]  # but no search
    assert applicator.apply.await_count == 1
    applied_job = applicator.apply.await_args_list[0].args[0]
    assert str(applied_job.url) == "https://linkedin.com/jobs/1"
    assert applicator.apply.await_args_list[0].kwargs.get("submit") is False  # dry-run default


def test_apply_saved_list_applies_each(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare `apply` with saved jobs applies (dry-run) to each, no search."""
    js = JobStore(db_path=tmp_path / "applications.db")
    js.upsert_job(_job(1))
    js.upsert_job(_job(2))
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    _, applicator = _patch_apply_browser(monkeypatch)

    result = CliRunner().invoke(cli.app, ["apply", "--no-cover-letter"])
    assert result.exit_code == 0, result.output
    cli._make_scraper.assert_not_called()  # type: ignore[attr-defined]
    assert applicator.apply.await_count == 2


# ------------------------------------------------------------- tailor --from
def test_tailor_from_unknown_errors(
    tmp_path: Path, sample_resume: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tailor --from with no matching stored job is a clean typed error (no traceback)."""
    js = JobStore(db_path=tmp_path / "applications.db")  # empty
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    loader = MagicMock()
    loader.load.return_value = sample_resume
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    monkeypatch.setattr(cli, "_run_ats_preflight", lambda r: MagicMock(score=80.0))

    result = CliRunner().invoke(cli.app, ["tailor", "--from", "999", "--resume", "/tmp/r.txt"])
    assert result.exit_code == 1, result.output
    assert "No stored job matches" in result.output
    assert "Traceback (most recent call last)" not in result.output


def _patch_tailor_stack(
    monkeypatch: pytest.MonkeyPatch, sample_resume: object, store: MagicMock
) -> None:
    """Mock the tailor command's resume/LLM/workflow stack so its cli wiring (esp. the
    post-workflow mark_tailored hook) runs offline. The interactive workflow is stubbed
    to simulate the user accepting ([A]) by setting result.output_path."""
    from job_applicator.models import TailoredResume

    monkeypatch.setattr(cli, "_get_jobs_store", lambda: store)
    loader = MagicMock()
    loader.load.return_value = sample_resume
    monkeypatch.setattr("job_applicator.documents.resume.ResumeLoader", lambda: loader)
    monkeypatch.setattr(cli, "_run_ats_preflight", lambda r: MagicMock(score=80.0))
    monkeypatch.setattr(
        cli, "_detect_tone", lambda job: MagicMock(primary="professional", confidence=0.8)
    )
    monkeypatch.setattr(cli, "_make_runtime", lambda settings, name="llm": MagicMock())
    audit = MagicMock(
        entries=[],
        warnings=[],
        staleness_issues=[],
        ordering_issues=[],
        is_stale=False,
        earliest_date="2017",
        latest_date="2024",
    )
    monkeypatch.setattr(
        "job_applicator.documents.resume_tailor.ResumeDateValidator",
        lambda: MagicMock(audit=lambda r: audit),
    )
    tailored = TailoredResume(
        original_path="/tmp/r.txt",
        tailored_text="TAILORED",
        job_title="X",
        job_company="Y",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
        changes_summary="changes",
    )
    monkeypatch.setattr(
        "job_applicator.documents.resume_tailor.ResumeTailor",
        lambda *a, **k: MagicMock(tailor=AsyncMock(return_value=tailored)),
    )

    async def _wf(
        console,
        settings,
        job,
        resume_data,
        style,
        tone,
        _eng,
        session,
        result,
        reporter,
        yes=False,
        **kwargs,
    ):  # type: ignore[no-untyped-def]
        result.output_path = "/out/tailored.txt"  # simulate the user accepting ([A])

    monkeypatch.setattr(cli, "_tailor_workflow", _wf)


def test_tailor_from_marks_tailored(
    tmp_path: Path, sample_resume: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tailor --from a stored job runs the engine and marks the job tailored in the store
    (covers the post-workflow mark_tailored hook offline — otherwise LIVE-only)."""
    from job_applicator.models import StoredJob

    store = MagicMock()
    store.get.return_value = StoredJob(
        id=1, job=_job(1), first_seen_at=datetime.now(UTC), updated_at=datetime.now(UTC)
    )
    _patch_tailor_stack(monkeypatch, sample_resume, store)

    result = CliRunner().invoke(
        cli.app, ["tailor", "--from", "1", "--resume", "/tmp/r.txt", "--yes"]
    )
    assert result.exit_code == 0, result.output
    store.get.assert_called_once_with("1")
    store.mark_tailored.assert_called_once()
    assert store.mark_tailored.call_args.kwargs["tailored_resume_path"] == "/out/tailored.txt"


def test_tailor_manual_without_url_not_persisted(
    tmp_path: Path, sample_resume: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manual `tailor -t/-c` with no --url/--from is NOT written to the store — its
    placeholder URL would otherwise collide with every other no-url manual tailor."""
    store = MagicMock()
    _patch_tailor_stack(monkeypatch, sample_resume, store)

    result = CliRunner().invoke(
        cli.app, ["tailor", "-t", "Eng", "-c", "Acme", "--resume", "/tmp/r.txt", "--yes"]
    )
    assert result.exit_code == 0, result.output
    store.mark_tailored.assert_not_called()


def test_apply_from_uses_stored_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """apply --from an Indeed job must build the Indeed browser/applicator — not the
    --site linkedin default — so the stored job's board governs which applicator runs."""
    js = JobStore(db_path=tmp_path / "applications.db")
    js.upsert_job(
        JobListing(
            title="Data Eng", company="Initech", url="https://indeed.com/j/9", board=JobBoard.INDEED
        )
    )  # id 1
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    _, applicator = _patch_apply_browser(monkeypatch)
    make_browser = MagicMock(return_value=_browser_cm())
    make_applicator = MagicMock(return_value=applicator)
    monkeypatch.setattr(cli, "_make_browser", make_browser)
    monkeypatch.setattr(cli, "_make_applicator", make_applicator)

    result = CliRunner().invoke(cli.app, ["apply", "--from", "1", "--no-cover-letter"])
    assert result.exit_code == 0, result.output
    assert make_browser.call_args.args[0] == "indeed"  # stored board, not the linkedin default
    assert make_applicator.call_args.args[0] == "indeed"


# --------------------------------------------------- store robustness / dedup
def test_status_dedups_job_in_both_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A job present in BOTH stores (tailored head + submitted tail) is counted once, at
    its furthest stage (applied) — the core no-double-count contract."""
    from job_applicator.state import ApplicationState

    shared = _job(1)
    js = JobStore(db_path=tmp_path / "applications.db")
    js.mark_tailored(shared, tailored_resume_path="/t.txt")  # head: tailored, url .../1
    st = ApplicationState(db_path=tmp_path / "applications.db")
    st.record(
        ApplicationResult(
            job=shared, status=ApplicationStatus.SUBMITTED, timestamp=datetime.now(UTC)
        )
    )  # tail: submitted, same url
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: js)
    monkeypatch.setattr("job_applicator.state.ApplicationState", lambda *a, **k: st)

    result = CliRunner().invoke(cli.app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["total"] == 1  # same URL → one row, not two
    assert data["counts"]["applied"] == 1
    assert data["counts"]["tailored"] == 0  # overridden to its furthest stage


def test_search_persistence_failure_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store-write failure warns but does NOT sink the freshly-scraped results."""
    from job_applicator.jobs_store import JobStoreError

    scraper = MagicMock(scrape=AsyncMock(return_value=[_job(1)]))
    store = MagicMock()
    store.upsert_job.side_effect = JobStoreError("database is locked")
    monkeypatch.setattr(cli, "_make_browser", lambda *a, **k: _browser_cm())
    monkeypatch.setattr(cli, "_make_scraper", lambda *a, **k: scraper)
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: store)

    result = CliRunner().invoke(cli.app, ["search", "-q", "python", "--json"])
    assert result.exit_code == 0, result.output  # scrape result survives the store hiccup
    assert len(json.loads(result.stdout)) == 1  # the scraped job is still emitted
    assert "Could not save" in result.stderr  # but the failure is surfaced


def test_status_db_error_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store read failure (e.g. a concurrent write-lock) is a clean typed error."""
    from job_applicator.jobs_store import JobStoreError

    store = MagicMock()
    store.list_jobs.side_effect = JobStoreError("database is locked")
    monkeypatch.setattr(cli, "_get_jobs_store", lambda: store)
    monkeypatch.setattr(
        "job_applicator.state.ApplicationState",
        lambda *a, **k: MagicMock(list_recent=lambda limit=0: []),
    )

    result = CliRunner().invoke(cli.app, ["status"])
    assert result.exit_code == 1, result.output
    assert "Traceback (most recent call last)" not in (result.stdout + result.stderr)
    assert "database is locked" in result.stderr
