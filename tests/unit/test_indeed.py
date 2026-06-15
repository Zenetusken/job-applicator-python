"""Unit tests for the Indeed scraper (URL building + safety)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from job_applicator.config import AppSettings
from job_applicator.models import JobBoard
from job_applicator.scrapers.base import SearchParams
from job_applicator.scrapers.indeed import INDEED_JOBS, IndeedScraper


def test_indeed_board(app_settings: AppSettings) -> None:
    assert IndeedScraper(MagicMock(), app_settings).board == JobBoard.INDEED


def test_indeed_search_url(app_settings: AppSettings) -> None:
    scraper = IndeedScraper(MagicMock(), app_settings)
    url = scraper._build_search_url(SearchParams(query="python developer", location="Montreal, QC"))
    assert url.startswith(INDEED_JOBS + "?")
    assert "q=python+developer" in url
    assert "l=Montreal" in url


def test_indeed_remote_filter_applied(app_settings: AppSettings) -> None:
    scraper = IndeedScraper(MagicMock(), app_settings)
    url = scraper._build_search_url(SearchParams(query="dev", remote_only=True))
    assert "sc=" in url


@pytest.mark.asyncio
async def test_indeed_login_disabled_for_safety(app_settings: AppSettings) -> None:
    """Indeed search is public; automated login must never submit credentials."""
    scraper = IndeedScraper(MagicMock(), app_settings)
    assert await scraper.login("user@example.com", "secret") is False
