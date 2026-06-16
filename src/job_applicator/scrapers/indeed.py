"""Indeed job scraper.

Indeed job search is public (no login required), so this scraper never submits
credentials. Indeed is fronted by a Cloudflare *managed JS challenge* that blocks
headless Chrome. The fix is not a special engine: run **headed** from a **clean
(ephemeral) profile** and the existing stack clears the challenge (it even passes
cold). That requirement is declared by ``IndeedScraper.browser_policy()`` so the
browser is built correctly (see ``cli._make_browser``); an active challenge is
surfaced as a ``ScraperError``. See
``docs/compose/reports/2026-06-15-indeed-cloudflare-research.md``.

Selectors were tuned against the live Indeed DOM (2026-06-15): result cards
``div.job_seen_beacon`` / ``[data-jk]``, title link ``a.jcs-JobTitle`` (relative
href), company ``[data-testid="company-name"]``, location
``[data-testid="text-location"]``. Indeed redirects by region: the scraper
auto-detects the regional site it lands on (e.g. ca.indeed.com) and re-issues the
search there, caching the host for the session; ``target.indeed_domain`` pins a
region explicitly.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode, urlsplit

from playwright.async_api import BrowserContext, ElementHandle, Page

from job_applicator.browser.actions import navigate, random_delay, wait_for_selector
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings
from job_applicator.exceptions import NavigationError, ScraperError
from job_applicator.models import JobBoard, JobListing
from job_applicator.scrapers.base import BaseScraper, BrowserPolicy, SearchParams
from job_applicator.utils.cookies import load_cookies
from job_applicator.utils.logging import get_logger
from job_applicator.utils.region import detect_indeed_domain
from job_applicator.utils.retry import async_retry
from job_applicator.utils.url import host_matches

logger = get_logger("scrapers.indeed")


def _is_indeed_host(host: str) -> bool:
    """True only for genuine Indeed hosts (any regional ``*.indeed.com``)."""
    return host_matches(host, "indeed.com")


class IndeedScraper(BaseScraper):
    """Scrapes public job listings from Indeed."""

    COOKIE_PATH = Path.home() / ".job-applicator" / "cookies" / "indeed.json"

    @classmethod
    def browser_policy(cls) -> BrowserPolicy:
        """Indeed's Cloudflare managed challenge needs a headed browser on a clean
        profile; run it windowless via Xvfb."""
        return BrowserPolicy(headed=True, ephemeral_profile=True, virtual_display=True)

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config
        self._resolved_base: str | None = None
        self._auto_base: str | None = None  # cached region origin (computed once)

    @property
    def _base(self) -> str:
        """Region-appropriate Indeed origin.

        Order: a host pinned mid-session by a region redirect (``_resolved_base``)
        > the explicitly configured ``target.indeed_domain`` > a host auto-detected
        from the machine's timezone (e.g. ca.indeed.com in Canada). Indeed does not
        reliably redirect www→region by IP, so picking the right host up front
        matters — and the timezone is a better signal than the often-en_US locale.

        The auto-detected origin is computed once and cached on the instance, so
        repeated ``_base`` reads (e.g. per result card) don't re-scan the tz table.
        """
        if self._resolved_base:
            return self._resolved_base
        if self._auto_base is None:
            domain = self._config.target.indeed_domain or detect_indeed_domain()
            self._auto_base = f"https://{domain}"
        return self._auto_base

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
        return f"{self._base}/jobs?{urlencode(query)}"

    async def _is_blocked(self, page: Page) -> bool:
        """Detect an Indeed anti-bot / Cloudflare challenge."""
        url = page.url.lower()
        if any(token in url for token in ("challenge", "captcha", "blocked", "/hcaptcha")):
            return True
        title = (await page.title()).lower()
        return "just a moment" in title or "verify you are human" in title

    @async_retry(max_attempts=3, base_delay=2.0, exceptions=(NavigationError,))
    async def scrape(self, params: SearchParams) -> list[JobListing]:
        """Scrape Indeed job listings for the given search params.

        Indeed redirects by region; _load_results pins whatever regional host it
        lands on (e.g. ca.indeed.com), so if a region mismatch bounces us to a
        regional homepage with no results, the search is re-issued once there.
        """
        jobs: list[JobListing] = []
        # Indeed needs a headed browser (see browser_policy); warn if built headless
        # so a direct (non-CLI) caller gets a clear signal instead of a silent block.
        if getattr(self._browser, "headless", None) is True:
            logger.warning(
                "Indeed is being scraped with a HEADLESS browser; Cloudflare will "
                "likely challenge it. Build the browser per IndeedScraper.browser_policy() "
                "(headed) — the CLI does this automatically."
            )
        context = await self._browser.persistent_context()
        # Apply any imported Indeed cookies (e.g. cf_clearance) as a best-effort warm
        # start. NOT required: Indeed runs headed on a fresh profile, which clears the
        # Cloudflare challenge cold — a warm session can only help, never gate.
        await load_cookies(context, self.COOKIE_PATH)
        page = await self._new_stealth_page(context)
        try:
            searched = urlsplit(self._base).netloc
            cards = await self._load_results(page, params)
            if not cards and urlsplit(self._base).netloc != searched:
                logger.info("Indeed redirected to %s; re-issuing the search there.", self._base)
                cards = await self._load_results(page, params)
            if not cards:
                logger.warning("No Indeed job cards found (page: %s)", page.url)
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
                    "Found %d Indeed card(s) but extracted 0 jobs — selectors may be "
                    "stale against the current Indeed DOM.",
                    len(cards),
                )
            logger.info("Scraped %d jobs from Indeed", len(jobs))
            return jobs
        finally:
            await page.close()

    async def _load_results(self, page: Page, params: SearchParams) -> list[ElementHandle]:
        """Navigate to the search, fail on anti-bot, return job-card handles.

        Pins the regional Indeed host actually landed on (Indeed redirects by
        region) so job-link URLs and any retry use the same origin.
        """
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
        host = urlsplit(page.url).netloc
        if _is_indeed_host(host):
            self._resolved_base = f"https://{host}"  # pin the region we landed on
        for selector in ("div.job_seen_beacon", "[data-jk]", "div.cardOutline"):
            if await wait_for_selector(page, selector, timeout_ms=5_000):
                cards = await page.query_selector_all(selector)
                if cards:
                    return cards
        return []

    async def _extract_job(self, card: ElementHandle, board: JobBoard) -> JobListing | None:
        """Extract job data from an Indeed result card."""
        # Primary selectors verified against the live Indeed DOM (2026-06-15);
        # the legacy fallbacks (span.companyName / div.companyLocation) still
        # appear on some regional sites and A/B buckets, so keep them rather than
        # silently degrade company/location to "Unknown"/"".
        title_el = await card.query_selector("a.jcs-JobTitle, h2 a")
        if not title_el:
            return None
        title = (await title_el.inner_text()).strip()
        href = await title_el.get_attribute("href")
        if not href:
            return None
        url = href if href.startswith("http") else f"{self._base}{href}"

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
