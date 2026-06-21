"""Safety tests for the LinkedIn Easy Apply dry-run gate.

The critical guarantee: an automated `apply` run must NOT submit a real
application unless the caller explicitly opts in with submit=True.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from job_applicator.applicators.linkedin import LinkedInApplicator
from job_applicator.config import AppSettings
from job_applicator.models import ApplicationStatus, JobBoard, JobListing


def _job() -> JobListing:
    return JobListing(
        title="X", company="Y", url="https://www.linkedin.com/jobs/1", board=JobBoard.LINKEDIN
    )


def _page_reaching_submit(submit_btn: AsyncMock) -> AsyncMock:
    """A page whose only matched selector is the final 'Submit application' button."""
    page = AsyncMock()

    async def query_selector(selector: str) -> object | None:
        return submit_btn if "Submit application" in selector else None

    page.query_selector = query_selector
    return page


@pytest.mark.asyncio
async def test_easy_apply_dry_run_does_not_submit(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    submit_btn = AsyncMock()
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(
        _page_reaching_submit(submit_btn), _job(), None, submit=False
    )

    assert result.status == ApplicationStatus.SKIPPED
    assert "DRY RUN" in result.notes
    submit_btn.click.assert_not_awaited()  # the critical guarantee — nothing submitted


@pytest.mark.asyncio
async def test_easy_apply_submits_only_with_opt_in(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    monkeypatch.setattr(
        "job_applicator.applicators.linkedin.wait_for_selector", AsyncMock(return_value=True)
    )
    submit_btn = AsyncMock()
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(
        _page_reaching_submit(submit_btn), _job(), None, submit=True
    )

    assert result.status == ApplicationStatus.SUBMITTED
    submit_btn.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_easy_apply_dry_run_reports_validation_details(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run captures whether the form reached the Submit button."""
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    submit_btn = AsyncMock()
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(
        _page_reaching_submit(submit_btn), _job(), None, submit=False
    )

    assert result.dry_run is not None
    assert result.dry_run.reached_submit is True
    assert result.dry_run.easy_apply_button_found is True


@pytest.mark.asyncio
async def test_easy_apply_missing_submit_button_reports_failed_validation(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the Submit button is not found, dry_run.reached_submit is False."""
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())

    page = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(page, _job(), None, submit=False)

    assert result.status == ApplicationStatus.FAILED
    assert result.dry_run is not None
    assert result.dry_run.reached_submit is False
