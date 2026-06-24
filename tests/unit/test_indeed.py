"""Unit tests for the Indeed scraper (URL building + safety)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from job_applicator.config import AppSettings
from job_applicator.exceptions import ScraperError
from job_applicator.models import JobBoard, JobListing
from job_applicator.scrapers.base import SearchParams
from job_applicator.scrapers.indeed import IndeedScraper, _is_indeed_host


def test_indeed_board(app_settings: AppSettings) -> None:
    assert IndeedScraper(MagicMock(), app_settings).board == JobBoard.INDEED


def test_indeed_search_url(app_settings: AppSettings) -> None:
    app_settings.target.indeed_domain = "www.indeed.com"  # pin (default is "" = auto)
    scraper = IndeedScraper(MagicMock(), app_settings)
    url = scraper._build_search_url(SearchParams(query="python developer", location="Montreal, QC"))
    assert url.startswith("https://www.indeed.com/jobs?")
    assert "q=python+developer" in url
    assert "l=Montreal" in url


def test_indeed_base_auto_detects_region_when_unpinned(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty indeed_domain → derive the regional host from the timezone."""
    app_settings.target.indeed_domain = ""  # auto
    monkeypatch.setattr(
        "job_applicator.scrapers.indeed.detect_indeed_domain", lambda: "ca.indeed.com"
    )
    scraper = IndeedScraper(MagicMock(), app_settings)
    assert scraper._base == "https://ca.indeed.com"


def test_indeed_base_caches_auto_detection(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_base computes the auto-detected origin once, not per access (no per-card I/O)."""
    app_settings.target.indeed_domain = ""
    calls = {"n": 0}

    def fake_detect() -> str:
        calls["n"] += 1
        return "ca.indeed.com"

    monkeypatch.setattr("job_applicator.scrapers.indeed.detect_indeed_domain", fake_detect)
    scraper = IndeedScraper(MagicMock(), app_settings)
    for _ in range(5):
        assert scraper._base == "https://ca.indeed.com"
    assert calls["n"] == 1  # detected once, then cached on the instance


def test_indeed_browser_policy_is_headed_ephemeral_virtual() -> None:
    """The Cloudflare requirement lives on the board, not just the CLI."""
    policy = IndeedScraper.browser_policy()
    assert policy.headed is True
    assert policy.ephemeral_profile is True
    assert policy.virtual_display is True


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
    app_settings.target.indeed_domain = "www.indeed.com"  # pin so calls[0] is deterministic
    scraper = IndeedScraper(MagicMock(), app_settings)
    # Isolate from any real ~/.job-applicator/cookies/indeed.json so scrape()'s
    # load_cookies() is a no-op on a non-existent path (no env-dependent failure).
    scraper.COOKIE_PATH = tmp_path / "indeed.json"  # type: ignore[assignment]
    scraper._browser.persistent_context = AsyncMock(return_value=MagicMock())
    scraper._new_stealth_page = AsyncMock(return_value=AsyncMock())
    # Return a real job (not None): this test is about the region retry, and a card that
    # extracts to nothing now raises (the honest-failure path), which isn't what we're testing.
    scraper._extract_job = AsyncMock(
        return_value=JobListing(
            title="E", company="Co", url="https://ca.indeed.com/jobs/1", board=JobBoard.INDEED
        )
    )

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


@pytest.mark.asyncio
async def test_scrape_emits_per_card_progress(app_settings: AppSettings, tmp_path: object) -> None:
    """scrape() ticks on_progress per CARD (1-based, /total) at the top of the loop, so a
    card whose extraction raises still advances the count to N/N (counts cards, not jobs)."""
    from job_applicator.models import JobListing

    app_settings.target.indeed_domain = "www.indeed.com"  # pin (no region detour)
    scraper = IndeedScraper(MagicMock(), app_settings)
    scraper.COOKIE_PATH = tmp_path / "indeed.json"  # type: ignore[operator,assignment]
    scraper._browser.persistent_context = AsyncMock(return_value=MagicMock())
    scraper._new_stealth_page = AsyncMock(return_value=AsyncMock())
    scraper._load_results = AsyncMock(return_value=[MagicMock() for _ in range(3)])

    def _stub(n: int) -> JobListing:
        return JobListing(
            title=f"E{n}", company="Co", url=f"https://indeed.com/jobs/{n}", board=JobBoard.INDEED
        )

    scraper._extract_job = AsyncMock(side_effect=[_stub(1), ValueError("bad card"), _stub(3)])

    msgs: list[str] = []
    jobs = await scraper.scrape(SearchParams(query="python", board=JobBoard.INDEED), msgs.append)

    assert msgs == [
        "Scraping job 1/3 on Indeed…",
        "Scraping job 2/3 on Indeed…",
        "Scraping job 3/3 on Indeed…",
    ]
    assert [j.title for j in jobs] == ["E1", "E3"]  # 2 extracted; count still reached 3/3


@pytest.mark.asyncio
async def test_scrape_streams_each_job_via_on_job(
    app_settings: AppSettings, tmp_path: object
) -> None:
    """scrape() streams each parsed listing via on_job as it lands; the streamed sequence
    equals the returned list."""
    from job_applicator.models import JobListing

    app_settings.target.indeed_domain = "www.indeed.com"
    scraper = IndeedScraper(MagicMock(), app_settings)
    scraper.COOKIE_PATH = tmp_path / "indeed.json"  # type: ignore[operator,assignment]
    scraper._browser.persistent_context = AsyncMock(return_value=MagicMock())
    scraper._new_stealth_page = AsyncMock(return_value=AsyncMock())
    scraper._load_results = AsyncMock(return_value=[MagicMock() for _ in range(2)])

    def _stub(n: int) -> JobListing:
        return JobListing(
            title=f"E{n}", company="Co", url=f"https://indeed.com/jobs/{n}", board=JobBoard.INDEED
        )

    scraper._extract_job = AsyncMock(side_effect=[_stub(1), _stub(2)])

    streamed: list[JobListing] = []
    jobs = await scraper.scrape(
        SearchParams(query="python", board=JobBoard.INDEED), on_job=streamed.append
    )
    assert [j.title for j in streamed] == ["E1", "E2"]
    assert [j.title for j in jobs] == ["E1", "E2"]


@pytest.mark.asyncio
async def test_scrape_raises_and_dumps_when_cards_present_but_none_extracted(
    app_settings: AppSettings,
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Honest failure: cards on the page but 0 extracted (stale FIELD selectors) raises
    ScraperError — so a multi-board search surfaces "Indeed failed" instead of silently
    dropping it — and dumps the live DOM for selector diagnosis. (A genuinely empty search
    returns 0 CARDS, handled separately, so this can't false-positive.)"""
    from pathlib import Path

    debug_dir = tmp_path / "debug"  # type: ignore[operator]
    monkeypatch.setattr("job_applicator.scrapers.indeed._DEBUG_DIR", debug_dir)
    app_settings.target.indeed_domain = "www.indeed.com"
    scraper = IndeedScraper(MagicMock(), app_settings)
    scraper.COOKIE_PATH = tmp_path / "indeed.json"  # type: ignore[operator,assignment]
    scraper._browser.persistent_context = AsyncMock(return_value=MagicMock())

    page = AsyncMock()
    page.url = "https://www.indeed.com/jobs?q=python"
    page.content = AsyncMock(return_value="<html><body>results here</body></html>")
    page.query_selector_all = AsyncMock(return_value=[])  # container counts for the dump
    scraper._new_stealth_page = AsyncMock(return_value=page)

    card = AsyncMock()
    card.inner_html = AsyncMock(return_value="<div data-jk='1'>live card markup</div>")
    scraper._load_results = AsyncMock(return_value=[card])  # a card IS present…
    scraper._extract_job = AsyncMock(return_value=None)  # …but extraction yields nothing

    with pytest.raises(ScraperError, match="extracted 0 jobs"):
        await scraper.scrape(SearchParams(query="python", board=JobBoard.INDEED))

    # The diagnostic landed in the isolated debug dir and captured what's needed to fix it.
    assert (Path(debug_dir) / "indeed-last-scrape.html").exists()
    summary = (Path(debug_dir) / "indeed-last-scrape.txt").read_text(encoding="utf-8")
    assert "container match counts" in summary
    assert "live card markup" in summary  # first card's inner_html, to fix field selectors
