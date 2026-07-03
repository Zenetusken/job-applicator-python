"""LinkedIn job scraper."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
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
from job_applicator.exceptions import (
    LoginRequiredError,
    NavigationError,
    RateLimitError,
    ScraperError,
)
from job_applicator.models import JobBoard, JobListing, SessionHealth
from job_applicator.scrapers.base import BaseScraper, SearchParams
from job_applicator.scrapers.text_repair import repair_glued_text
from job_applicator.utils.cookies import load_cookies, save_cookies
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger("scrapers.linkedin")

LINKEDIN_BASE = "https://www.linkedin.com"
LINKEDIN_JOBS = f"{LINKEDIN_BASE}/jobs/search"
LINKEDIN_TYPEAHEAD = f"{LINKEDIN_BASE}/jobs-guest/api/typeaheadHits"


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

    async def _raise_if_blocked(self, page: Page) -> None:
        """Raise a precise typed error if the page is a LinkedIn anti-bot wall.

        A security checkpoint/challenge served to a flagged account otherwise fails
        ``_is_authenticated_url`` and is mis-diagnosed as "no session" — prescribing re-login, the
        wrong remedy for a flagged account (re-running login automation raises the account's risk).
        A rate-limit interstitial otherwise surfaces as the generic "stale or blocked" 0-cards
        error. Detect both via URL + title tokens (mirrors ``IndeedScraper._is_blocked``) and raise
        the right error so the caller STOPS / backs off rather than retrying or re-authenticating.
        A title that can't be read is treated as "not blocked".
        """
        url = page.url.lower()
        try:
            title = (await page.title()).lower()
        except PlaywrightError:
            title = ""
        if (
            "/checkpoint" in url
            or "/challenge" in url
            or "captcha" in url
            or "security verification" in title
            or "unusual activity" in title
            or "verify you are" in title  # CAPTCHA wording; NOT the benign "verify your email"
        ):
            raise ScraperError(
                "LinkedIn served a security checkpoint/challenge — the account may be under "
                "review. Sign in manually in a real browser to clear it; do NOT re-run login "
                "automation (it raises the account's risk). Stop automated runs until resolved."
            )
        if (
            "too many requests" in title
            or "commercial use limit" in title  # LinkedIn's canonical (monthly) search cap
            or "weekly limit" in title
            or "monthly limit" in title
        ):
            raise RateLimitError(
                "LinkedIn returned a rate-limit / usage-limit interstitial. Back off and retry "
                "later, and reduce search/apply volume."
            )

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
            # A checkpoint/challenge here must be surfaced as itself, not mis-read as "no session".
            await self._raise_if_blocked(page)
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
        except (NavigationError, ScraperError):
            # A transient failure OR an anti-bot block both mean "no usable session right now".
            # This is a best-effort check (import-cookies --verify), so degrade to False rather
            # than raise; the block itself is already logged.
            logger.warning("Session check could not confirm a usable session; treating as none.")
            return False

    async def check_session(self) -> SessionHealth:
        """Best-effort health check for the LinkedIn session.

        Returns healthy when the feed loads while authenticated. Transient
        network failures are captured in ``details`` rather than raised.
        """
        try:
            healthy = await self._ensure_session(await self._get_context())
        except NavigationError as exc:
            return SessionHealth(
                board=JobBoard.LINKEDIN,
                healthy=False,
                details=f"Could not reach LinkedIn to verify the session: {exc}",
            )

        if healthy:
            return SessionHealth(
                board=JobBoard.LINKEDIN,
                healthy=True,
                details="Authenticated LinkedIn session is active.",
            )
        return SessionHealth(
            board=JobBoard.LINKEDIN,
            healthy=False,
            details=(
                "No authenticated LinkedIn session found. "
                "Run `job-applicator login` to sign in once in a real browser window."
            ),
        )

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
        except (NavigationError, ScraperError):
            # A transient pre-check failure OR an anti-bot checkpoint/rate-limit must not abort the
            # login command — opening the login page IS how the user manually clears a checkpoint.
            # Fall through and open it.
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
    async def scrape(
        self,
        params: SearchParams,
        on_progress: Callable[[str], None] | None = None,
        on_job: Callable[[JobListing], None] | None = None,
    ) -> list[JobListing]:
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
        return await self._scrape_listings(params, context, on_progress, on_job)

    async def _scrape_listings(
        self,
        params: SearchParams,
        context: BrowserContext,
        on_progress: Callable[[str], None] | None = None,
        on_job: Callable[[JobListing], None] | None = None,
    ) -> list[JobListing]:
        """Fetch and parse job cards from the search results page."""
        jobs: list[JobListing] = []
        page = await self._new_stealth_page(context)
        try:
            geo_id = await self._resolve_geo_id(page, params.location) if params.location else None
            search_url = self._build_search_url(params, geo_id)
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
            cards: list[ElementHandle] = []
            for selector in selectors:
                found = await wait_for_selector(page, selector, timeout_ms=5_000)
                if found:
                    cards = await page.query_selector_all(selector)
                    if cards:
                        break
            if not found or not cards:
                # 0 cards is ambiguous (genuinely-empty search vs stale container selectors /
                # unauthenticated / anti-bot block) — FAIL LOUDLY rather than report a silent
                # empty result that masks a probable failure. Classify a checkpoint / rate-limit
                # first (a precise typed error the caller can act on) before the generic message.
                await self._raise_if_blocked(page)
                raise ScraperError(
                    "No LinkedIn job cards found on the results page — the container selectors "
                    "are stale, the session is unauthenticated, or the search was blocked."
                )
            # Two passes, because clicking a card re-renders LinkedIn's virtualized list and
            # DETACHES the other captured handles ("element not attached to the DOM" — a measured,
            # deterministic loss of a fixed card batch per scrape). Pass 1 snapshots EVERY card's
            # metadata from the still-fresh handles with NO mutation; pass 2 loads each description
            # by RE-RESOLVING the card fresh (by job id), so a description miss degrades to a
            # metadata-only listing instead of dropping the whole job.
            selected = cards[: params.max_results]
            total = len(selected)
            meta_failures = 0
            snapshots: list[JobListing] = []
            for card in selected:
                try:
                    job = await self._extract_job(card, params.board)
                except Exception as exc:  # one unreadable card must not sink the whole scrape —
                    # the 0-snapshots guard below still fails loud if ALL cards fail.
                    meta_failures += 1
                    logger.warning("Failed to read a job card's metadata: %s", exc)
                    continue
                if job:
                    snapshots.append(job)

            if not snapshots and total:
                # Cards were present (total > 0) but NOT ONE yielded metadata. Key on `total`, NOT
                # failures: a stale TITLE/href selector makes _extract_job RETURN None (it does not
                # raise), so a failures-keyed guard would silently return [] — the exact masking
                # this prevents (R1). FAIL LOUDLY, consistent with the 0-cards guard above;
                # max_results=0 leaves total=0 and is a legitimate empty, excluded here.
                raise ScraperError(
                    f"Found {total} LinkedIn job card(s) but extracted 0 titles "
                    f"({meta_failures} threw, {total - meta_failures} returned no title/href). "
                    "The title/field selectors are likely stale against the live LinkedIn DOM — "
                    "or this search genuinely has no matches and rendered only non-card elements. "
                    "Re-run a broader, known-populated search to disambiguate."
                )

            desc_misses = 0
            n = len(snapshots)
            for i, job in enumerate(snapshots, start=1):
                # Tick per listing at the top so the count shows WHILE the slow click+description
                # wait runs and never stalls on a description miss.
                if on_progress is not None:
                    on_progress(f"Scraping job {i}/{n} on LinkedIn…")
                desc = await self._load_job_description(page, str(job.url))
                if desc:
                    job = job.model_copy(update={"description": desc})
                else:
                    desc_misses += 1
                jobs.append(job)  # keep the listing even without a description — metadata matches
                if on_job is not None:  # stream the listing as soon as it's complete
                    on_job(job)

            if meta_failures or desc_misses:
                logger.warning(
                    "Scraped %d/%d LinkedIn cards (%d metadata failures, %d without a description)",
                    len(jobs),
                    total,
                    meta_failures,
                    desc_misses,
                )
            logger.info("Scraped %d job(s) from %d LinkedIn card(s)", len(jobs), total)
            return jobs
        finally:
            await page.close()

    def _build_search_url(self, params: SearchParams, geo_id: str | None = None) -> str:
        """Build LinkedIn job search URL.

        LinkedIn geo-filters on the numeric ``geoId``, NOT the human ``location=`` string (a bare
        location string is loosely matched and commonly ignored, returning a global/remote feed).
        ``geo_id`` is resolved up front by :meth:`_resolve_geo_id`; ``location`` is kept alongside
        it for display / as a soft fallback when resolution failed.
        """
        query_params: dict[str, str | int] = {
            "keywords": params.query,
            "f_TPR": "r604800",  # Past week
        }
        if params.location:
            query_params["location"] = params.location
        if geo_id:
            query_params["geoId"] = geo_id
        if params.remote_only:
            query_params["f_WT"] = "2"  # Remote
        return f"{LINKEDIN_JOBS}?{urlencode(query_params)}"

    async def _resolve_geo_id(self, page: Page, location: str) -> str | None:
        """Resolve a location string to LinkedIn's numeric ``geoId`` via the guest typeahead API.

        Why: LinkedIn's job search geo-filters on ``geoId``; a bare ``location=`` string alone is
        loosely matched and often dropped (measured 2026-07-01: a ``location=Montréal, QC`` search
        returned 89% France/EMEA/EU-remote). Resolving the id fixes the filter.

        Robust to the common ``"City, ST"`` input: the typeahead rejects a 2-letter province/state
        abbreviation (``Montreal, QC`` → ``[]``) but accepts the bare city, so on an empty result we
        retry with the part before the first comma. To keep the dropped region as a disambiguator
        for same-name cities, ``_pick_geo_hit`` prefers a hit whose displayName reflects the typed
        region hint (falling back to the region-biased first hit, whose displayName IS logged for
        visibility). Returns ``None`` on ANY failure or no match — the caller falls back to the raw
        ``location=`` string (degraded geo, logged), never a fabricated id.
        """
        region_hint = location.split(",", 1)[1].strip().lower() if "," in location else ""
        candidates = [location]
        city = location.split(",")[0].strip()
        if city and city != location:
            candidates.append(city)
        for candidate in candidates:
            url = f"{LINKEDIN_TYPEAHEAD}?" + urlencode(
                {
                    "typeaheadType": "GEO",
                    "geoTypes": "POPULATED_PLACE,ADMIN_DIVISION_2",
                    "query": candidate,
                }
            )
            try:
                resp = await page.request.get(url, timeout=10_000)
            except PlaywrightError as exc:
                logger.warning("geoId typeahead request failed for %r (%s)", candidate, exc)
                continue
            if not resp.ok:
                logger.warning("geoId typeahead HTTP %d for %r", resp.status, candidate)
                continue
            try:
                hits = await resp.json()
            except (ValueError, PlaywrightError) as exc:
                logger.warning("geoId typeahead returned non-JSON for %r (%s)", candidate, exc)
                continue
            chosen = _pick_geo_hit(hits, region_hint)
            if chosen is not None:
                geo_id = str(chosen.get("id", ""))
                logger.info(
                    "Resolved location %r → geoId %s (%s)",
                    location,
                    geo_id,
                    chosen.get("displayName", "?"),
                )
                return geo_id
        logger.warning(
            "Could not resolve a geoId for %r — falling back to the raw location filter "
            "(results may not be geo-constrained).",
            location,
        )
        return None

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

        # Salary on the card (best-effort; LinkedIn shows it on a minority of postings). These
        # selectors are UNVERIFIED against the live LinkedIn DOM — LinkedIn is the user's real
        # account, so it is never auto-searched to capture markup; a wrong selector just yields
        # None (no salary), never a crash. Confirm/refine on the next real LinkedIn search.
        salary_el = await card.query_selector(
            ".job-card-container__metadata-wrapper [class*='salary'], "
            "[class*='job-card-container__salary'], [class*='compensation']"
        )
        salary = (await salary_el.inner_text()).strip() if salary_el else None

        url = _canonical_job_url(href if href.startswith("http") else f"{LINKEDIN_BASE}{href}")

        return JobListing(
            title=title,
            company=company,
            url=url,  # type: ignore[arg-type]
            location=location,
            salary=salary or None,
            board=board,
        )

    async def _load_job_description(self, page: Page, url: str) -> str:
        """Load a job's description by re-resolving its card fresh (by EXACT job id) and clicking.

        Re-resolves via a live selector rather than reusing a captured ElementHandle: LinkedIn's
        virtualized list detaches handles when an earlier click re-renders it (the measured
        stale-handle loss). Returns ``""`` — the caller keeps the metadata-only listing rather than
        attaching a wrong/empty description — when the id can't be parsed, the card can't be
        re-resolved (scrolled out of the virtualized window), or the panel never confirms THIS job.
        """
        job_id = _job_id_from_url(url)
        if not job_id:
            return ""
        # Match the card by EXACT job id, not an ``href*="/jobs/view/123"`` substring — that would
        # also bind a longer-id card (``/jobs/view/1234``) and attach the wrong description.
        link = None
        for candidate in await page.query_selector_all('a[href*="/jobs/view/"]'):
            href = await candidate.get_attribute("href") or ""
            if _job_id_from_url(href) == job_id:
                link = candidate
                break
        if link is None:
            return ""
        prev_desc = await self._get_desc_text(page)
        try:
            await link.scroll_into_view_if_needed(timeout=5_000)
            await link.click(timeout=5_000)
        except PlaywrightError as exc:
            logger.warning("Could not open LinkedIn job %s: %s", job_id, exc)
            return ""
        # Trust the panel's text ONLY once it demonstrably shows THIS job: either the description
        # CHANGED (a new job loaded) or the URL's currentJobId matches (covers the auto-selected
        # first card, whose text doesn't change on click). On a genuine timeout return "" — NEVER
        # the previously-selected job's still-displayed description (that would silently attach the
        # WRONG description and be miscounted as a successful load).
        for _ in range(10):
            await random_delay(0.3, 0.5)
            new_desc = await self._get_desc_text(page)
            if (
                new_desc
                and len(new_desc) > 100
                and (new_desc != prev_desc or _job_id_from_url(page.url) == job_id)
            ):
                return await self._extract_description(page)
        return ""

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


def _pick_geo_hit(hits: Any, region_hint: str) -> dict[str, Any] | None:
    """Choose the best GEO typeahead hit for a possibly-ambiguous same-name city.

    Prefers a hit whose displayName reflects the region hint typed after the city (a province/state
    name like "Quebec"/"Illinois", or a code that appears within the full name), so a same-name city
    is not silently resolved to the region-biased first hit. Falls back to the first numeric hit
    (LinkedIn ranks by the request's own locale/IP) when nothing matches the hint. Returns ``None``
    when ``hits`` is not a non-empty list of dicts carrying a numeric id.
    """
    if not isinstance(hits, list):
        return None
    numeric = [h for h in hits if isinstance(h, dict) and str(h.get("id", "")).isdigit()]
    if not numeric:
        return None
    if region_hint:
        for hit in numeric:
            if region_hint in str(hit.get("displayName", "")).lower():
                return hit
    return numeric[0]


def _job_id_from_url(url: str) -> str:
    """Extract the numeric LinkedIn job id from a job URL.

    Handles both ``/jobs/view/<id>`` (the card title-link href) and ``?currentJobId=<id>``
    (the selected-card query form). Returns ``""`` when neither is present.
    """
    match = re.search(r"/jobs/view/(\d+)", url) or re.search(r"[?&]currentJobId=(\d+)", url)
    return match.group(1) if match else ""


def _canonical_job_url(url: str) -> str:
    """Reduce a LinkedIn job URL to its stable ``/jobs/view/<id>`` identity.

    LinkedIn serves the SAME job under many tracking-decorated URLs (``?eBP=…&trackingId=…``) that
    differ per search — so storing the full URL makes one job dedup as several rows (measured
    2026-07-01: 53% of a 92-job funnel was tracking phantoms). Canonicalizing to the numeric job id
    collapses those while keeping genuinely-distinct jobs (different ids) apart. A URL with NO
    parseable id is returned UNCHANGED — we can't know its identity, so stripping its query could
    collapse two genuinely-distinct id-less URLs into one funnel key (a lossy masking default).
    Latent today — live card hrefs are always ``/jobs/view/<id>``.
    """
    job_id = _job_id_from_url(url)
    if job_id:
        return f"{LINKEDIN_BASE}/jobs/view/{job_id}/"
    return url


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
    # Corruption-gated glued-word repair FIRST (upstream rich-text mash — see text_repair);
    # clean descriptions pass through byte-identical.
    text = repair_glued_text(raw)
    # Strip "About the job" prefix
    for prefix in ("About the job\n\n", "About the job\n"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    # Skip redirect-only descriptions
    if "please review our complete list" in text.lower() and len(text) < 300:
        return ""
    return text.strip()
