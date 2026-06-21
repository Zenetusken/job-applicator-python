"""Unit tests for the Indeed applicator — search-only scope."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from job_applicator.models import ApplicationStatus, JobBoard, JobListing


async def test_indeed_apply_is_search_only_never_auto_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indeed is scoped to search-only: even apply(submit=True) on an 'Easily apply'
    posting must NOT auto-submit — it returns a clean SKIPPED search-only result.
    Guards the safety invariant that Indeed never sends a real application."""
    import job_applicator.applicators.indeed as ind

    page = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=page)
    cm.__aexit__ = AsyncMock(return_value=False)
    browser = MagicMock()
    browser.persistent_page.return_value = cm

    monkeypatch.setattr(ind, "navigate", AsyncMock())
    monkeypatch.setattr(ind, "random_delay", AsyncMock())
    # "Easily apply" detected → would be the auto-submit path for LinkedIn.
    monkeypatch.setattr(ind, "wait_for_selector", AsyncMock(return_value=True))

    app = ind.IndeedApplicator(browser, MagicMock())
    job = JobListing(
        title="Dev",
        company="Acme",
        url="https://www.indeed.com/viewjob?jk=1",
        board=JobBoard.INDEED,
    )

    result = await app.apply(job, submit=True)  # explicit --submit...

    assert result.status == ApplicationStatus.SKIPPED  # ...yet never auto-submits
    assert result.status is not ApplicationStatus.SUBMITTED
    assert "search-only" in (result.notes or "").lower()
