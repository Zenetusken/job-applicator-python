"""Unit tests for scrapers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from job_applicator.config import AppSettings
from job_applicator.exceptions import LoginRequiredError
from job_applicator.scrapers.base import SearchParams
from job_applicator.scrapers.linkedin import LinkedInScraper


@pytest.mark.asyncio
async def test_scrape_without_credentials_names_correct_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The credential error must point at the env vars that actually work."""
    # Skip retry backoff delays so the test is fast.
    monkeypatch.setattr("job_applicator.utils.retry.asyncio.sleep", AsyncMock())

    settings = AppSettings()  # empty linkedin_email / linkedin_password by default
    scraper = LinkedInScraper(MagicMock(), settings)

    with pytest.raises(LoginRequiredError) as excinfo:
        await scraper.scrape(SearchParams(query="python"))

    message = str(excinfo.value)
    assert "JOB_APPLICATOR_TARGET_LINKEDIN_EMAIL" in message
    assert "JOB_APPLICATOR_TARGET_LINKEDIN_PASSWORD" in message
    assert "config.toml" in message
    # The old, misleading guidance must be gone.
    assert "Set LINKEDIN_EMAIL and LINKEDIN_PASSWORD" not in message
