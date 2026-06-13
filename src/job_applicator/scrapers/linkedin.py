"""LinkedIn job scraper."""

from __future__ import annotations

from urllib.parse import urlencode

from playwright.async_api import ElementHandle

from job_applicator.browser.actions import (
    click,
    fill,
    navigate,
    random_delay,
    wait_for_selector,
)
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings
from job_applicator.exceptions import LoginRequiredError
from job_applicator.models import JobBoard, JobListing
from job_applicator.scrapers.base import BaseScraper, SearchParams
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

logger = get_logger("scrapers.linkedin")

LINKEDIN_BASE = "https://www.linkedin.com"
LINKEDIN_JOBS = f"{LINKEDIN_BASE}/jobs/search"


class LinkedInScraper(BaseScraper):
    """Scrapes job listings from LinkedIn."""

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config
        self._logged_in = False

    @property
    def board(self) -> JobBoard:
        return JobBoard.LINKEDIN

    async def login(self, email: str, password: str) -> bool:
        """Authenticate with LinkedIn."""
        if not email or not password:
            raise LoginRequiredError("LinkedIn credentials not configured")

        async with self._browser.new_page() as page:
            await navigate(page, f"{LINKEDIN_BASE}/login")
            await random_delay(1.0, 2.0)

            await fill(page, 'input[name="session_key"]', email)
            await fill(page, 'input[name="session_password"]', password)
            await click(page, 'button[type="submit"]')

            # Wait for feed or challenge page
            await random_delay(2.0, 4.0)

            if "feed" in page.url or "mynetwork" in page.url:
                self._logged_in = True
                logger.info("LinkedIn login successful")
                return True

            logger.warning("LinkedIn login may have failed (challenge page?)")
            return False

    @async_retry(max_attempts=3, base_delay=2.0)
    async def scrape(self, params: SearchParams) -> list[JobListing]:
        """Scrape LinkedIn job listings."""
        if not self._logged_in:
            email = self._config.target.linkedin_email
            password = self._config.target.linkedin_password
            if email and password:
                await self.login(email, password)
            else:
                raise LoginRequiredError(
                    "LinkedIn credentials required. Set LINKEDIN_EMAIL and LINKEDIN_PASSWORD.",
                )

        jobs: list[JobListing] = []
        async with self._browser.new_page() as page:
            search_url = self._build_search_url(params)
            await navigate(page, search_url)
            await random_delay(2.0, 3.0)

            # Wait for job cards to load
            found = await wait_for_selector(page, ".job-card-container", timeout_ms=15_000)
            if not found:
                logger.warning("No job cards found on page")
                return jobs

            # Extract job cards
            cards = await page.query_selector_all(".job-card-container")
            for card in cards[: params.max_results]:
                try:
                    job = await self._extract_job(card, params.board)
                    if job:
                        jobs.append(job)
                except Exception as exc:
                    logger.warning("Failed to extract job card: %s", exc)

            logger.info("Scraped %d jobs from LinkedIn", len(jobs))
        return jobs

    def _build_search_url(self, params: SearchParams) -> str:
        """Build LinkedIn job search URL."""
        query_params: dict[str, str | int] = {
            "keywords": params.query,
            "f_TPR": "r604800",  # Past week
        }
        if params.location:
            query_params["location"] = params.location
        if params.remote_only:
            query_params["f_WT"] = "2"  # Remote
        return f"{LINKEDIN_JOBS}?{urlencode(query_params)}"

    async def _extract_job(self, card: ElementHandle, board: JobBoard) -> JobListing | None:
        """Extract job data from a card element."""
        title_el = await card.query_selector(".job-card-list__title--link")
        if not title_el:
            return None

        title = (await title_el.inner_text()).strip()
        href = await title_el.get_attribute("href")
        if not href:
            return None

        company_el = await card.query_selector(".artdeco-entity-lockup__subtitle")
        company = (await company_el.inner_text()).strip() if company_el else "Unknown"

        location_el = await card.query_selector(".artdeco-entity-lockup__caption")
        location = (await location_el.inner_text()).strip() if location_el else ""

        url = href if href.startswith("http") else f"{LINKEDIN_BASE}{href}"

        return JobListing(
            title=title,
            company=company,
            url=url,  # type: ignore[arg-type]
            location=location,
            board=board,
        )
