"""Unit tests for the Indeed scraper (URL building + safety)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from job_applicator.config import AppSettings
from job_applicator.models import JobBoard
from job_applicator.scrapers.base import SearchParams
from job_applicator.scrapers.indeed import IndeedScraper, _is_indeed_host


def test_indeed_board(app_settings: AppSettings) -> None:
    assert IndeedScraper(MagicMock(), app_settings).board == JobBoard.INDEED


def test_indeed_search_url(app_settings: AppSettings) -> None:
    scraper = IndeedScraper(MagicMock(), app_settings)
    url = scraper._build_search_url(SearchParams(query="python developer", location="Montreal, QC"))
    # Default region origin derives from config (target.indeed_domain), not a constant.
    assert url.startswith(f"https://{app_settings.target.indeed_domain}/jobs?")
    assert "q=python+developer" in url
    assert "l=Montreal" in url


def test_indeed_search_url_respects_region_domain(app_settings: AppSettings) -> None:
    app_settings.target.indeed_domain = "ca.indeed.com"
    scraper = IndeedScraper(MagicMock(), app_settings)
    url = scraper._build_search_url(SearchParams(query="python"))
    assert url.startswith("https://ca.indeed.com/jobs?")


def test_indeed_remote_filter_applied(app_settings: AppSettings) -> None:
    scraper = IndeedScraper(MagicMock(), app_settings)
    url = scraper._build_search_url(SearchParams(query="dev", remote_only=True))
    assert "sc=" in url


@pytest.mark.asyncio
async def test_indeed_login_disabled_for_safety(app_settings: AppSettings) -> None:
    """Indeed search is public; automated login must never submit credentials."""
    scraper = IndeedScraper(MagicMock(), app_settings)
    assert await scraper.login("user@example.com", "secret") is False


def test_is_indeed_host_rejects_lookalikes() -> None:
    assert _is_indeed_host("www.indeed.com") is True
    assert _is_indeed_host("ca.indeed.com") is True
    assert _is_indeed_host("indeed.com") is True
    assert _is_indeed_host("notindeed.com") is False
    assert _is_indeed_host("indeed.com.evil.example") is False


@pytest.mark.asyncio
async def test_extract_job_uses_legacy_fallback_selectors(app_settings: AppSettings) -> None:
    """Cards with only the legacy markup still yield company/location, not the
    'Unknown'/'' degradation that dropping the fallback selectors would cause."""
    scraper = IndeedScraper(MagicMock(), app_settings)

    title_el = AsyncMock()
    title_el.inner_text = AsyncMock(return_value="Backend Engineer")
    title_el.get_attribute = AsyncMock(return_value="/viewjob?jk=1")
    company_el = AsyncMock()
    company_el.inner_text = AsyncMock(return_value="LegacyCo")
    location_el = AsyncMock()
    location_el.inner_text = AsyncMock(return_value="Montreal, QC")

    async def query(selector: str) -> object | None:
        if "jcs-JobTitle" in selector:
            return title_el
        if "span.companyName" in selector:  # only the legacy company selector is present
            return company_el
        if "div.companyLocation" in selector:  # only the legacy location selector is present
            return location_el
        return None

    card = MagicMock()
    card.query_selector = AsyncMock(side_effect=query)

    job = await scraper._extract_job(card, JobBoard.INDEED)
    assert job is not None
    assert job.company == "LegacyCo"
    assert job.location == "Montreal, QC"


@pytest.mark.asyncio
async def test_scrape_auto_retries_on_region_redirect(
    app_settings: AppSettings, tmp_path: object
) -> None:
    """If the search bounces to a regional Indeed host with no results, the
    scraper pins that host and re-issues the search there (auto region)."""
    scraper = IndeedScraper(MagicMock(), app_settings)
    # Isolate from any real ~/.job-applicator/cookies/indeed.json so scrape()'s
    # load_cookies() is a no-op on a non-existent path (no env-dependent failure).
    scraper.COOKIE_PATH = tmp_path / "indeed.json"  # type: ignore[assignment]
    scraper._browser.persistent_context = AsyncMock(return_value=MagicMock())
    scraper._new_stealth_page = AsyncMock(return_value=AsyncMock())
    scraper._extract_job = AsyncMock(return_value=None)

    calls: list[str] = []

    async def fake_load(page: object, params: SearchParams) -> list[object]:
        calls.append(scraper._base)
        if len(calls) == 1:
            scraper._resolved_base = "https://ca.indeed.com"  # simulate landing on ca
            return []
        return [MagicMock()]

    scraper._load_results = fake_load

    await scraper.scrape(SearchParams(query="python developer"))

    assert len(calls) == 2  # retried after the region redirect
    assert scraper._resolved_base == "https://ca.indeed.com"
    assert calls[0] == "https://www.indeed.com"  # first attempt on the default
    assert calls[1] == "https://ca.indeed.com"  # retry on the detected region
