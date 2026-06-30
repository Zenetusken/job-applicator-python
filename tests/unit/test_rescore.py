"""Unit tests for the `rescore` command — account-safe in-place re-scoring of stored jobs.

`rescore` re-scores the funnel against the current résumé WITHOUT re-scraping: it reads
stored jobs, recomputes match scores via the matcher, and writes them back in place. These
pin the guarantees that matter: scores update, the funnel stage is preserved, an empty store
is handled cleanly, the `--json` contract holds, and — the headline — NO scraper/browser is
ever constructed (the account-safety promise of refreshing without touching LinkedIn/Indeed).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from job_applicator.embeddings.matching import MatchResult
from job_applicator.models import FunnelStatus, JobBoard, JobListing, ResumeData


def _job(n: int, **over: object) -> JobListing:
    data: dict[str, object] = {
        "title": f"Analyst {n}",
        "company": f"Co{n}",
        "url": f"https://linkedin.com/jobs/{n}",
        "description": "soc monitoring",
        "location": "Montréal",
        "requirements": ["siem", "python"],
        "board": JobBoard.LINKEDIN,
        "seniority": "junior",
    }
    data.update(over)
    return JobListing(**data)  # type: ignore[arg-type]


def _match(job: JobListing, score: float) -> MatchResult:
    return MatchResult(
        job=job,
        score=score,
        semantic_score=score,
        skill_score=score,
        matched_skills=["python"],
        missing_skills=["siem"],
        summary="x",
    )


def _drive(monkeypatch, *, seed, new_scores, args=None, rank_raises=None):
    """Seed the (conftest-isolated) store, then run `rescore` with a mocked matcher + CV load.

    seed: list of (JobListing, old_score). new_scores: {url: new_score} the mock matcher returns.
    rank_raises: if set, the mock matcher raises it (to exercise the fail-closed path).
    Returns (CliRunner result, the JobStore for assertions, scraper_calls list).
    """
    import job_applicator.cli as cli

    store = cli._get_jobs_store()
    for job, old in seed:
        store.upsert_job(job)
        store.upsert_match(_match(job, old))

    async def fake_rank(resume, jobs, top_k):
        if rank_raises is not None:
            raise rank_raises
        return [_match(j, new_scores[str(j.url)]) for j in jobs]

    matcher = MagicMock()
    matcher.rank_jobs = fake_rank

    loader = MagicMock()
    loader.load.return_value = ResumeData(raw_text="Jane\njane@x.com\nPython", name="Jane")

    scraper_calls: list[str] = []

    def _no_scrape(*a, **k):  # account-safety tripwire
        scraper_calls.append("scraper")
        raise AssertionError("rescore must not construct a scraper/browser")

    with (
        patch("job_applicator.documents.resume.ResumeLoader", return_value=loader),
        patch("job_applicator.embeddings.matching.JobMatcher", return_value=matcher),
        patch.object(cli, "_make_runtime", MagicMock()),
        patch.object(cli, "_make_scraper", _no_scrape),
        patch.object(cli, "_make_browser", _no_scrape),
    ):
        result = CliRunner().invoke(cli.app, ["rescore", "--resume", "r.docx", *(args or [])])
    return result, cli._get_jobs_store(), scraper_calls


def test_rescore_updates_scores_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored job's score is refreshed to the recomputed value, in place (keyed by url)."""
    job = _job(1)
    result, store, _ = _drive(monkeypatch, seed=[(job, 0.40)], new_scores={str(job.url): 0.85})
    assert result.exit_code == 0, result.output
    refreshed = store.get(str(job.url))
    assert refreshed is not None
    assert refreshed.match_score == pytest.approx(0.85)  # 0.40 → 0.85, written back


def test_rescore_is_account_safe_no_scraper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The headline guarantee: rescore never constructs a scraper/browser (no re-scraping)."""
    job = _job(1)
    result, _store, scraper_calls = _drive(
        monkeypatch, seed=[(job, 0.40)], new_scores={str(job.url): 0.50}
    )
    assert result.exit_code == 0, result.output
    assert scraper_calls == []  # the tripwire never fired


def test_rescore_preserves_funnel_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job already advanced past 'matched' keeps its further stage after a rescore."""
    import job_applicator.cli as cli

    job = _job(1)
    # Seed + advance to 'tailored' BEFORE the rescore.
    store = cli._get_jobs_store()
    store.upsert_job(job)
    store.upsert_match(_match(job, 0.40))
    store.mark_tailored(job, tailored_resume_path="/tmp/x.txt")

    result, store2, _ = _drive(monkeypatch, seed=[], new_scores={str(job.url): 0.85})
    assert result.exit_code == 0, result.output
    refreshed = store2.get(str(job.url))
    assert refreshed is not None
    assert refreshed.match_score == pytest.approx(0.85)  # score refreshed
    assert refreshed.funnel_status == FunnelStatus.TAILORED  # ...stage NOT regressed to matched


def test_rescore_empty_store_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty funnel rescores to nothing, cleanly (exit 0, a helpful message)."""
    result, _store, _ = _drive(monkeypatch, seed=[], new_scores={})
    assert result.exit_code == 0, result.output
    assert "No stored jobs" in result.output


def test_rescore_empty_store_json_is_empty_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty funnel + `--json` emits exactly `[]` on stdout (parseable, no Rich leakage)."""
    result, _store, _ = _drive(monkeypatch, seed=[], new_scores={}, args=["--json"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "[]"


def test_rescore_fail_closed_leaves_funnel_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    """The headline safety guarantee: if matching fails (e.g. the LLM endpoint is down), NO
    scores are written — the funnel is left intact (fail-closed) and the command exits non-zero.
    `rank_jobs` raises before returning, so the write loop never runs (no partial refresh)."""
    from job_applicator.exceptions import JobApplicatorError

    j1, j2 = _job(1), _job(2)
    result, store, _ = _drive(
        monkeypatch,
        seed=[(j1, 0.40), (j2, 0.55)],
        new_scores={},
        rank_raises=JobApplicatorError("LLM endpoint unreachable"),
    )
    assert result.exit_code != 0  # failed loud, not a silent success
    assert store.get(str(j1.url)).match_score == pytest.approx(0.40)  # UNCHANGED — no write
    assert store.get(str(j2.url)).match_score == pytest.approx(0.55)  # ...and not partially written


def test_rescore_multi_job_console_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple jobs with mixed deltas exercise the console summary (rose/fell counts + the
    'Biggest score changes' table) — the non-json path the single-job/json tests don't cover —
    and confirm every job is written back."""
    j1, j2, j3 = _job(1), _job(2), _job(3)
    result, store, _ = _drive(
        monkeypatch,
        seed=[(j1, 0.40), (j2, 0.60), (j3, 0.50)],
        new_scores={str(j1.url): 0.70, str(j2.url): 0.30, str(j3.url): 0.50},  # rise · fall · same
    )
    assert result.exit_code == 0, result.output
    assert "Re-scored 3 stored jobs" in result.output
    assert "rose" in result.output and "fell" in result.output
    assert store.get(str(j1.url)).match_score == pytest.approx(0.70)
    assert store.get(str(j2.url)).match_score == pytest.approx(0.30)
    assert store.get(str(j3.url)).match_score == pytest.approx(0.50)


def test_rescore_json_emits_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    """`rescore --json` emits parseable JSON carrying before/after/delta per job (stdout only)."""
    job = _job(1)
    result, _store, _ = _drive(
        monkeypatch, seed=[(job, 0.40)], new_scores={str(job.url): 0.85}, args=["--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)  # raises if Rich output leaked onto stdout
    assert len(payload) == 1
    row = payload[0]
    assert row["score"] == pytest.approx(0.85)
    assert row["score_before"] == pytest.approx(0.40)
    assert row["delta"] == pytest.approx(0.45)
