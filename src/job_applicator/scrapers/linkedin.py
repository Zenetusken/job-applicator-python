"""LinkedIn job scraper."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from playwright.async_api import BrowserContext, ElementHandle, Page
from playwright.async_api import Error as PlaywrightError

from job_applicator.browser.actions import (
    navigate,
    random_delay,
    wait_for_selector,
)
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings
from job_applicator.exceptions import LoginRequiredError, NavigationError
from job_applicator.models import JobBoard, JobListing
from job_applicator.scrapers.base import BaseScraper, SearchParams
from job_applicator.utils.cookies import load_cookies, save_cookies
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

logger = get_logger("scrapers.linkedin")

LINKEDIN_BASE = "https://www.linkedin.com"
LINKEDIN_JOBS = f"{LINKEDIN_BASE}/jobs/search"


def _is_authenticated_url(url: str) -> bool:
    """True only when the URL *path* is a logged-in LinkedIn page.

    Path-based, not substring: the logged-out redirect
    ``.../uas/login?session_redirect=...%2Ffeed%2F`` embeds ``feed`` in the
    query string, so a substring ``"feed" in url`` check would false-positive
    and report an authenticated session when there is none.
    """
    return urlsplit(url).path.startswith(("/feed", "/mynetwork"))


class LinkedInScraper(BaseScraper):
    """Scrapes job listings from LinkedIn."""

    COOKIE_PATH = Path.home() / ".job-applicator" / "cookies" / "linkedin.json"

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config
        self._logged_in = False

    async def _new_stealth_page(self, context: BrowserContext) -> Page:
        """Open a page in the persistent context.

        Stealth is applied once at the context level (BrowserManager.start), and
        the context auto-applies it to every page it creates, so no per-page
        stealth call is needed here (verified: navigator.webdriver is patched on
        context-created pages without a second application).
        """
        return await context.new_page()

    @property
    def _cookie_file(self) -> Path:
        return self.COOKIE_PATH

    async def _load_cookies(self, context: BrowserContext) -> bool:
        """Load saved cookies into the context (best-effort, per-cookie tolerant)."""
        added = await load_cookies(context, self._cookie_file)
        if added:
            logger.info("Loaded %d cookies from %s", added, self._cookie_file)
        return added > 0

    @classmethod
    def write_cookie_file(cls, cookies: Any) -> None:
        """Persist cookies to the on-disk session file (atomic, 0600).

        Single owner of the cookie-file path + ``{"cookies": [...]}`` envelope,
        shared by the scraper's _save_cookies and the `import-cookies` command.
        """
        save_cookies(cls.COOKIE_PATH, cookies)

    async def _save_cookies(self, context: BrowserContext) -> None:
        """Persist cookies from the browser context to disk (best-effort)."""
        try:
            cookies = await context.cookies()
            self.write_cookie_file(cookies)
            logger.info("Saved %d cookies to %s", len(cookies), self._cookie_file)
        except Exception as exc:
            logger.warning("Failed to save cookies to %s: %s", self._cookie_file, exc)

    @property
    def board(self) -> JobBoard:
        return JobBoard.LINKEDIN

    async def _get_context(self) -> BrowserContext:
        """Get the manager's shared persistent context for login + scraping.

        Using the manager's persistent context (rather than reaching into a
        private browser handle) means the login session established here is the
        same one the applicator reuses for authenticated Easy Apply.
        """
        return await self._browser.persistent_context()

    async def login(self, email: str, password: str) -> bool:
        """Automated credential login is intentionally DISABLED for account safety.

        LinkedIn blocks programmatic logins with a CAPTCHA, and repeated
        automated attempts raise the account's risk score. This method never
        submits credentials and never touches the browser — use
        :meth:`interactive_login` (the ``job-applicator login`` command) to sign
        in once in a real browser window.
        """
        logger.warning(
            "Automated LinkedIn login is disabled for account safety. "
            "Run `job-applicator login` to sign in once via a real browser window."
        )
        return False

    async def _ensure_session(self, context: BrowserContext) -> bool:
        """Return True if an authenticated LinkedIn session is already active.

        Loads any saved cookies (a portable seed) into the context, then
        verifies by loading the feed. The persistent browser profile usually
        already carries the session on its own. Never submits credentials, so it
        cannot trigger a login CAPTCHA (it is still an automated request, so
        keep overall scraping volume modest).
        """
        await self._load_cookies(context)
        page = await self._new_stealth_page(context)
        try:
            await page.goto(f"{LINKEDIN_BASE}/feed/", wait_until="domcontentloaded", timeout=15_000)
            await random_delay(1.0, 2.0)
            if _is_authenticated_url(page.url):
                self._logged_in = True
                logger.info("Reusing existing LinkedIn session")
                return True
            logger.info("No active LinkedIn session (redirected to %s)", page.url)
            return False
        except PlaywrightError as exc:
            # A transient page-load failure must NOT be misreported as "no
            # session" (which would tell the user to re-authenticate). Surface
            # it as a retryable NavigationError so scrape()'s retry can recover.
            raise NavigationError(
                f"LinkedIn session check failed to load the feed: {exc}",
                context={"url": f"{LINKEDIN_BASE}/feed/"},
            ) from exc
        finally:
            await page.close()

    async def has_active_session(self) -> bool:
        """Public check: is a usable authenticated LinkedIn session available?

        Loads any saved cookies and verifies against the feed. Submits no
        credentials, so it cannot trigger a login CAPTCHA. A transient feed-load
        failure is treated as "no session" (returns False) rather than raised —
        this is a best-effort check (used by `import-cookies --verify`).
        """
        try:
            return await self._ensure_session(await self._get_context())
        except NavigationError:
            logger.warning("Session check could not load the feed; treating as no session.")
            return False

    async def interactive_login(self, timeout_s: int = 300) -> bool:
        """Open LinkedIn's login page for a one-time, human-driven sign-in.

        Requires a headed browser (use the ``job-applicator login`` command).
        Pre-fills the configured credentials but does NOT submit — you click
        Sign in and solve any CAPTCHA/2FA yourself. Human-driven sign-in is far
        safer than a programmatic submit, though running inside an
        automation-controlled browser is never fully risk-free. Polls until a
        logged-in page is detected, then saves the session; the persistent
        profile retains it for subsequent headless runs.
        """
        context = await self._get_context()
        try:
            if await self._ensure_session(context):
                logger.info("Already signed in — existing session is active.")
                return True
        except NavigationError:
            # A transient pre-check failure must not abort the login command;
            # fall through and open the login page.
            logger.info("Could not pre-check existing session; opening the login page.")

        page = await self._new_stealth_page(context)
        try:
            await navigate(page, f"{LINKEDIN_BASE}/login")
            await random_delay(1.0, 2.0)

            # Pre-fill from config to save typing; the human reviews and submits.
            email = self._config.target.linkedin_email
            password = self._config.target.linkedin_password
            try:
                if email:
                    await page.locator('input[type="email"]').last.fill(email)
                if password:
                    await page.locator('input[type="password"]').last.fill(password)
            except Exception as exc:
                logger.debug("Could not pre-fill credentials: %s", exc)

            logger.info(
                "Waiting up to %ds for manual sign-in — click Sign in in the "
                "browser window and solve any CAPTCHA/2FA...",
                timeout_s,
            )
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if _is_authenticated_url(page.url):
                    self._logged_in = True
                    await self._save_cookies(context)
                    logger.info("Sign-in detected — session saved.")
                    return True
                await asyncio.sleep(2.0)

            logger.error("Sign-in not detected within %ds.", timeout_s)
            return False
        finally:
            await page.close()

    @async_retry(max_attempts=3, base_delay=2.0, exceptions=(NavigationError,))
    async def scrape(self, params: SearchParams) -> list[JobListing]:
        """Scrape LinkedIn job listings.

        Reuses an existing authenticated session (persistent profile / saved
        cookies). Automated credential login is NOT attempted — run
        ``job-applicator login`` first to establish a session safely. Raising
        ``LoginRequiredError`` here (rather than auto-logging-in) is deliberate:
        a programmatic login is exactly what trips LinkedIn's anti-bot CAPTCHA.

        The retry wraps both the session check and the scrape, and fires only on
        the transient :class:`NavigationError`; ``LoginRequiredError`` (genuine
        no-session) is not retried.
        """
        context = await self._get_context()
        if not self._logged_in and not await self._ensure_session(context):
            raise LoginRequiredError(
                "No authenticated LinkedIn session found. Run `job-applicator login` "
                "to sign in once in a real browser window (you solve any CAPTCHA/2FA). "
                "The session is saved to the persistent browser profile and reused "
                "automatically on subsequent runs.",
            )
        return await self._scrape_listings(params, context)

    async def _scrape_listings(
        self, params: SearchParams, context: BrowserContext
    ) -> list[JobListing]:
        """Fetch and parse job cards from the search results page."""
        jobs: list[JobListing] = []
        page = await self._new_stealth_page(context)
        try:
            search_url = self._build_search_url(params)
            await navigate(page, search_url)
            await random_delay(2.0, 3.0)

            # Wait for job cards to load (multiple selector fallbacks)
            selectors = [
                ".job-card-container",
                "[data-job-id]",
                ".jobs-search-results__list-item",
                ".job-card-list__entity",
                "li.jobs-search-results__list-item",
            ]
            found = False
            for selector in selectors:
                found = await wait_for_selector(page, selector, timeout_ms=5_000)
                if found:
                    cards = await page.query_selector_all(selector)
                    if cards:
                        break
            if not found or not cards:
                logger.warning("No job cards found on page")
                return jobs
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
