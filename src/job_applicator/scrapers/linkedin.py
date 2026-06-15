"""LinkedIn job scraper."""

from __future__ import annotations

from urllib.parse import urlencode

from playwright.async_api import BrowserContext, ElementHandle, Page

from job_applicator.browser.actions import (
    navigate,
    random_delay,
    wait_for_selector,
)
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings
from job_applicator.exceptions import BrowserError, LoginRequiredError
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
        self._context: BrowserContext | None = None

    @property
    def board(self) -> JobBoard:
        return JobBoard.LINKEDIN

    async def _get_context(self) -> BrowserContext:
        """Get or create a persistent browser context for login + scraping."""
        if self._context is None:
            if not self._browser._browser:
                raise BrowserError("Browser not started")
            self._context = await self._browser._browser.new_context(
                viewport={
                    "width": self._config.browser.viewport_width,
                    "height": self._config.browser.viewport_height,
                },
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
            )
            self._context.set_default_timeout(self._config.browser.timeout_ms)
        return self._context

    async def login(self, email: str, password: str) -> bool:
        """Authenticate with LinkedIn."""
        if not email or not password:
            raise LoginRequiredError("LinkedIn credentials not configured")

        context = await self._get_context()
        page = await context.new_page()
        try:
            await navigate(page, f"{LINKEDIN_BASE}/login")
            await random_delay(1.0, 2.0)

            # LinkedIn uses dynamic IDs — use type-based locators with .last
            # to get the visible form fields (hidden ones exist for OAuth)
            email_loc = page.locator('input[type="email"]').last
            pwd_loc = page.locator('input[type="password"]').last
            sign_in = page.locator('button:has-text("Sign in")').last

            await email_loc.wait_for(state="visible", timeout=10_000)
            await email_loc.fill(email)
            await pwd_loc.fill(password)
            await sign_in.click()

            # Wait for feed or challenge page
            await random_delay(2.0, 4.0)

            if "feed" in page.url or "mynetwork" in page.url:
                self._logged_in = True
                logger.info("LinkedIn login successful")
                return True

            logger.warning(
                "LinkedIn login may have failed (challenge/CAPTCHA?). "
                "Consider using cookie-based session or manual credentials."
            )
            return False
        finally:
            await page.close()

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

        context = await self._get_context()
        page = await context.new_page()
        try:
            jobs: list[JobListing] = []
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
                        # Click card to load description in detail panel.
                        # LinkedIn auto-selects the first card on page load,
                        # so we need to wait for content to update.
                        prev_desc = await self._get_desc_text(page)
                        await card.click()
                        # Wait for description content to change
                        for _ in range(10):
                            await random_delay(0.3, 0.5)
                            new_desc = await self._get_desc_text(page)
                            if new_desc and new_desc != prev_desc and len(new_desc) > 100:
                                break
                        desc = await self._extract_description(page)
                        if desc:
                            job = job.model_copy(update={"description": desc})
                        jobs.append(job)
                except Exception as exc:
                    logger.warning("Failed to extract job card: %s", exc)

            logger.info("Scraped %d jobs from LinkedIn", len(jobs))
            return jobs
        finally:
            await page.close()

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

        raw_title = (await title_el.inner_text()).strip()
        title = _clean_title(raw_title)
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

    async def _get_desc_text(self, page: Page) -> str:
        """Get current description text (for change detection)."""
        el = await page.query_selector(".jobs-description__content")
        return (await el.inner_text()).strip() if el else ""

    async def _extract_description(self, page: Page) -> str:
        """Extract job description from the detail panel after clicking a card."""
        # Click "show more" button to expand truncated description.
        # LinkedIn has multiple "show more" buttons — we need the one that
        # expands the description, not the dropdown menu.
        for btn_text in ("show more", "Show more"):
            buttons = await page.query_selector_all(
                f'button:has-text("{btn_text}")[aria-expanded="false"]'
            )
            for btn in buttons:
                if not await btn.is_visible():
                    continue
                inner = (await btn.inner_text()).strip().lower()
                # Skip "Show more options" (dropdown) and "Show more filters"
                if "option" in inner or "filter" in inner:
                    continue
                try:
                    await btn.click()
                    await random_delay(0.5, 1.0)
                except Exception as exc:
                    logger.debug("Could not click show more button: %s", exc)
                break

        selectors = [
            ".jobs-description__content",
            ".jobs-description",
            "#job-details",
            ".show-more-less-html__markup",
        ]
        for sel in selectors:
            el = await page.query_selector(sel)
            if el:
                text = (await el.inner_text()).strip()
                if len(text) > 50:
                    return _clean_description(text[:5000])
        return ""


def _clean_title(raw: str) -> str:
    """Clean LinkedIn job title — remove duplicates and noise."""
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if not lines:
        return raw
    title = lines[0]
    # Remove "with verification" suffix
    if " with verification" in title.lower():
        title = title[: title.lower().index(" with verification")]
    return title.strip()


def _clean_description(raw: str) -> str:
    """Clean LinkedIn job description — remove prefixes and noise."""
    text = raw
    # Strip "About the job" prefix
    for prefix in ("About the job\n\n", "About the job\n"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    # Skip redirect-only descriptions
    if "please review our complete list" in text.lower() and len(text) < 300:
        return ""
    return text.strip()
