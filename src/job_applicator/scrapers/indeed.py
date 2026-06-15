"""Indeed job scraper.

Indeed job search is public (no login required), so this scraper never submits
credentials. Indeed fronts pages with Cloudflare / anti-bot, so automated
scraping from a fresh or headless profile may be challenged; the shared
persistent profile + stealth reduce that but cannot defeat an active challenge.

NOTE: This implementation mirrors the LinkedIn scraper's architecture and safety
model but has NOT been validated against live Indeed — selectors are best-effort
and may need tuning against the current Indeed DOM.
"""

from __future__ import annotations

from urllib.parse import urlencode

from playwright.async_api import BrowserContext, ElementHandle, Page

from job_applicator.browser.actions import navigate, random_delay, wait_for_selector
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings
from job_applicator.exceptions import NavigationError, ScraperError
from job_applicator.models import JobBoard, JobListing
from job_applicator.scrapers.base import BaseScraper, SearchParams
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

logger = get_logger("scrapers.indeed")

INDEED_BASE = "https://www.indeed.com"
INDEED_JOBS = f"{INDEED_BASE}/jobs"


class IndeedScraper(BaseScraper):
    """Scrapes public job listings from Indeed."""

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config

    @property
    def board(self) -> JobBoard:
        return JobBoard.INDEED

    async def login(self, email: str, password: str) -> bool:
        """Indeed search is public — automated login is unnecessary and disabled.

        Like the LinkedIn scraper, this never submits credentials (automated
        logins trip anti-bot defenses). Returns False without touching the page.
        """
        logger.info("Indeed search is public; automated login is skipped.")
        return False

    async def _new_stealth_page(self, context: BrowserContext) -> Page:
        """Open a page in the (context-level stealthed) persistent context."""
        return await context.new_page()

    def _build_search_url(self, params: SearchParams) -> str:
        """Build an Indeed job-search URL."""
        query: dict[str, str] = {"q": params.query}
        if params.location:
            query["l"] = params.location
        if params.remote_only:
            query["sc"] = "0kf:attr(DSQF7);"  # Indeed's "Remote" filter token (best-effort)
        return f"{INDEED_JOBS}?{urlencode(query)}"

    async def _is_blocked(self, page: Page) -> bool:
        """Detect an Indeed anti-bot / Cloudflare challenge."""
        url = page.url.lower()
        if any(token in url for token in ("challenge", "captcha", "blocked", "/hcaptcha")):
            return True
        title = (await page.title()).lower()
        return "just a moment" in title or "verify you are human" in title

    @async_retry(max_attempts=3, base_delay=2.0, exceptions=(NavigationError,))
    async def scrape(self, params: SearchParams) -> list[JobListing]:
        """Scrape Indeed job listings for the given search params."""
        jobs: list[JobListing] = []
        context = await self._browser.persistent_context()
        page = await self._new_stealth_page(context)
        try:
            await navigate(page, self._build_search_url(params))
            await random_delay(2.0, 3.0)

            if await self._is_blocked(page):
                # Distinguish a block from a legitimately empty result set —
                # returning [] here would be indistinguishable from "no jobs".
                raise ScraperError(
                    "Indeed returned an anti-bot challenge; automated scraping was blocked. "
                    "Reduce frequency or seed a real browser session.",
                    context={"url": page.url},
                )

            cards: list[ElementHandle] = []
            for selector in ("div.job_seen_beacon", "[data-jk]", "div.cardOutline"):
                if await wait_for_selector(page, selector, timeout_ms=5_000):
                    cards = await page.query_selector_all(selector)
                    if cards:
                        break
            if not cards:
                logger.warning("No Indeed job cards found on page")
                return jobs

            for card in cards[: params.max_results]:
                try:
                    job = await self._extract_job(card, params.board)
                    if job:
                        jobs.append(job)
                except Exception as exc:
                    logger.warning("Failed to extract Indeed card: %s", exc)

            if not jobs:
                # Cards were present but none parsed — almost certainly stale
                # selectors against the live DOM, not a genuinely empty search.
                logger.error(
                    "Found %d Indeed card(s) but extracted 0 jobs — selectors are "
                    "likely stale against the current Indeed DOM.",
                    len(cards),
                )
            logger.info("Scraped %d jobs from Indeed", len(jobs))
            return jobs
        finally:
            await page.close()

    async def _extract_job(self, card: ElementHandle, board: JobBoard) -> JobListing | None:
        """Extract job data from an Indeed result card."""
        title_el = await card.query_selector("h2.jobTitle a, a.jcs-JobTitle, h2 a")
        if not title_el:
            return None
        title = (await title_el.inner_text()).strip()
        href = await title_el.get_attribute("href")
        if not href:
            return None
        url = href if href.startswith("http") else f"{INDEED_BASE}{href}"

        company_el = await card.query_selector('[data-testid="company-name"], span.companyName')
        company = (await company_el.inner_text()).strip() if company_el else "Unknown"

        location_el = await card.query_selector(
            '[data-testid="text-location"], div.companyLocation'
        )
        location = (await location_el.inner_text()).strip() if location_el else ""

        return JobListing(
            title=title,
            company=company,
            url=url,  # type: ignore[arg-type]
            location=location,
            board=board,
        )
