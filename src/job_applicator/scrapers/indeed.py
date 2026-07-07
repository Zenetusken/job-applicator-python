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
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlsplit

from playwright.async_api import BrowserContext, ElementHandle, Page

from job_applicator.browser.actions import navigate, random_delay, wait_for_selector
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings
from job_applicator.exceptions import NavigationError, ScraperError
from job_applicator.models import JobBoard, JobListing, SessionHealth
from job_applicator.scrapers.base import BaseScraper, BrowserPolicy, SearchParams
from job_applicator.scrapers.text_repair import repair_glued_text
from job_applicator.utils.cookies import load_cookies
from job_applicator.utils.logging import get_logger
from job_applicator.utils.path import set_owner_only
from job_applicator.utils.region import detect_indeed_domain
from job_applicator.utils.retry import async_retry
from job_applicator.utils.url import host_matches

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger("scrapers.indeed")

# Where a 0-result scrape saves the live DOM for selector diagnosis. The TUI swallows
# stderr logs, so this file is the only record of *why* a scrape came up empty. A module
# constant so tests can redirect it away from the real ~/.job-applicator.
_DEBUG_DIR = Path.home() / ".job-applicator" / "debug"
INDEED_CARD_SELECTORS = ("div.job_seen_beacon", "[data-jk]", "div.cardOutline")
INDEED_TITLE_SELECTOR = "a.jcs-JobTitle, h2 a"
INDEED_COMPANY_SELECTOR = '[data-testid="company-name"], span.companyName'
INDEED_LOCATION_SELECTOR = '[data-testid="text-location"], div.companyLocation'
INDEED_SALARY_SELECTOR = '[class*="salary-snippet"], [data-testid*="salary-snippet"]'
INDEED_JK_SELECTOR = "[data-jk]"
INDEED_SNIPPET_SELECTORS = (
    '[data-testid="jobsnippet_footer"]',
    "div.job-snippet",
    "div[class*='job-snippet']",
)
INDEED_DESC_SELECTORS = (
    "#jobDescriptionText",
    ".jobsearch-JobComponent-description",
    "[id^='jobDescriptionText']",
)


def _is_indeed_host(host: str) -> bool:
    """True only for genuine Indeed hosts (any regional ``*.indeed.com``)."""
    return host_matches(host, "indeed.com")


def _clean_description(raw: str) -> str:
    """Tidy an Indeed description/snippet: collapse runs of blank lines and trim. Kept light
    on purpose — the matcher/ATS want the text, not heavy reformatting."""
    # Corruption-gated glued-word repair FIRST (upstream rich-text mash — see text_repair);
    # clean descriptions pass through byte-identical.
    raw = repair_glued_text(raw)
    lines = [line.rstrip() for line in raw.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks == 1:  # keep a single separating blank, drop the rest
                out.append("")
    return "\n".join(out).strip()


class IndeedScraper(BaseScraper):
    """Scrapes public job listings from Indeed."""

    COOKIE_PATH = Path.home() / ".job-applicator" / "cookies" / "indeed.json"

    # Result-card containers, tried in order (Indeed varies by region / A/B bucket). Shared
    # with the diagnostic dump so it reports a match count per selector.
    _CARD_SELECTORS = INDEED_CARD_SELECTORS

    # Short description teaser shown ON the result card (no click needed) — best-effort, used
    # as the baseline description so every Indeed job carries SOME text even when the full
    # description panel can't be loaded.
    _SNIPPET_SELECTORS = INDEED_SNIPPET_SELECTORS

    # Full description in Indeed's right-hand detail PANE after a card is clicked.
    # ``#jobDescriptionText`` is Indeed's long-stable description id (both the standalone
    # viewjob page and the search split-pane); the rest are best-effort fallbacks.
    _DESC_SELECTORS = INDEED_DESC_SELECTORS

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
        self._diag_dumped = False  # one failing-card diagnostic per scrape (see _dump_failed_card)

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

    async def check_session(self) -> SessionHealth:
        """Indeed job search is public and requires no login.

        The real gate is whether the headed browser can clear Cloudflare; that
        is verified per-scrape, so this check simply reports that no session is
        required.
        """
        return SessionHealth(
            board=JobBoard.INDEED,
            healthy=True,
            details="Indeed search is public; no login session required.",
        )

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
    async def scrape(
        self,
        params: SearchParams,
        on_progress: Callable[[str], None] | None = None,
        on_job: Callable[[JobListing], None] | None = None,
    ) -> list[JobListing]:
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
                # 0 cards is ambiguous (a genuinely empty search OR stale CONTAINER selectors /
                # anti-bot block); we can't tell from here. Per the no-masking rule, dump the live
                # DOM for diagnosis and FAIL LOUDLY rather than report a silent empty result.
                logger.warning("No Indeed job cards found (page: %s)", page.url)
                await self._dump_debug(page, [])
                raise ScraperError(
                    "No Indeed job cards found on the results page — the container selectors are "
                    "stale or the search was blocked (DOM dumped for diagnosis)."
                )

            selected = cards[: params.max_results]
            total = len(selected)
            # Pass 1 — extract every card's listing fields (incl. the on-card snippet) WITHOUT
            # clicking. Doing this up front means the working card-level scrape can never be
            # lost to the description enrichment below, which clicks cards and could in
            # principle navigate the page (detaching the remaining card handles).
            parsed: list[tuple[int, ElementHandle, JobListing]] = []
            seen_urls: set[str] = set()
            for i, card in enumerate(selected, start=1):
                # Tick per CARD at the top of the loop (see LinkedIn) so a failed
                # extraction never stalls the count.
                if on_progress is not None:
                    on_progress(f"Scraping job {i}/{total} on Indeed…")
                try:
                    job = await self._extract_job(card, params.board)
                except Exception as exc:
                    logger.warning("Failed to extract Indeed card: %s", exc)
                    continue
                if not job:
                    continue
                # Indeed lists the SAME job as both a sponsored ad and an organic result; the
                # canonical (data-jk) URL collapses those — skip the repeat so it isn't stored
                # twice or counted twice.
                if str(job.url) in seen_urls:
                    continue
                seen_urls.add(str(job.url))
                parsed.append((i, card, job))

            if not parsed:
                # Cards were present but none parsed → stale FIELD selectors against the live
                # DOM (a genuinely empty search returns 0 CARDS above, so this is never a real
                # empty result). Dump the DOM and FAIL LOUDLY so the caller surfaces "Indeed
                # extraction failed" instead of silently reporting 0 jobs (the bug that let a
                # multi-board search quietly drop Indeed).
                dump = await self._dump_debug(page, cards)
                raise ScraperError(
                    f"Found {len(cards)} Indeed card(s) but extracted 0 jobs — the field "
                    "selectors are stale against the current Indeed DOM."
                    + (f" Live DOM saved to {dump}." if dump else ""),
                    context={"url": page.url, "cards": len(cards)},
                )

            # Pass 2 — best-effort full-description enrichment via Indeed's split pane (mirrors
            # the LinkedIn click-to-load flow). Non-fatal per card: a job whose detail panel
            # won't load keeps its card snippet, so this can never drop a job.
            self._diag_dumped = False
            enriched = 0
            for i, card, job in parsed:
                if on_progress is not None:
                    on_progress(f"Reading description {i}/{total} on Indeed…")
                # The canonical URL is /viewjob?jk=<jk>; that key ties the loaded pane to this
                # card (see _load_description). Jobs that fell back to an href have no jk.
                jk = parse_qs(urlsplit(str(job.url)).query).get("jk", [None])[0]
                desc = ""
                try:
                    desc = await self._load_description(page, card, jk)
                except Exception as exc:  # enrichment is best-effort; keep the snippet
                    logger.debug("Indeed description enrichment failed (card %d): %s", i, exc)
                if desc and len(desc) > len(job.description):
                    job = job.model_copy(update={"description": desc})
                    enriched += 1
                elif not self._diag_dumped:
                    # First card with NO full description → capture the live DOM that tells us
                    # WHY (navigated away? challenged? empty pane? stale snippet/desc selectors?)
                    # so ONE user run pins the fix. The selectors are still unverified against a
                    # real card DOM, which is why this diagnostic, not a blind retune, ships now.
                    self._diag_dumped = True
                    await self._dump_failed_card(page, card)
                jobs.append(job)
                if on_job is not None:  # stream the (possibly enriched) listing
                    on_job(job)

            if not enriched:
                logger.warning(
                    "Indeed: no full descriptions loaded for any of %d card(s); using the card "
                    "snippet. Diagnostic dump in %s.",
                    len(parsed),
                    _DEBUG_DIR,
                )
            logger.info("Scraped %d jobs from Indeed", len(jobs))
            return jobs
        finally:
            await page.close()

    async def _dump_debug(self, page: Page, cards: list[ElementHandle]) -> Path | None:
        """Save the live DOM + per-container match counts (+ the first card's HTML) when a
        scrape comes up empty, so stale selectors can be fixed against the real page. The TUI
        swallows stderr logs, so this file is the record. Best-effort — never breaks a scrape.
        """
        try:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            set_owner_only(_DEBUG_DIR, 0o700)
            (_DEBUG_DIR / "indeed-last-scrape.html").write_text(
                await page.content(), encoding="utf-8"
            )
            lines = [f"url: {page.url}", "", "container match counts:"]
            for selector in self._CARD_SELECTORS:
                try:
                    count = len(await page.query_selector_all(selector))
                except Exception:  # a bad selector must not break the diagnostic
                    count = -1
                lines.append(f"  {selector}: {count}")
            if cards:
                lines += ["", "first card inner_html:", await cards[0].inner_html()]
            (_DEBUG_DIR / "indeed-last-scrape.txt").write_text("\n".join(lines), encoding="utf-8")
            logger.warning("Indeed debug dump written to %s", _DEBUG_DIR)
            return _DEBUG_DIR
        except Exception as exc:  # diagnostics never break the scrape
            logger.warning("Could not write Indeed debug dump: %s", exc)
            return None

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
        for selector in self._CARD_SELECTORS:
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
        title_el = await card.query_selector(INDEED_TITLE_SELECTOR)
        if not title_el:
            return None
        title = (await title_el.inner_text()).strip()
        href = await title_el.get_attribute("href")
        # Prefer the stable job key (data-jk) for a CANONICAL URL: it dedupes the same job
        # listed as both a sponsored ad (/pagead/clk… tracking redirect) and an organic
        # result (/viewjob), and avoids persisting an expiring redirect as the job URL. Fall
        # back to the card href only when the live DOM exposes no jk.
        jk = await self._card_jk(card, title_el)
        if jk:
            url = f"{self._base}/viewjob?jk={jk}"
        elif href:
            url = href if href.startswith("http") else f"{self._base}{href}"
        else:
            return None

        company_el = await card.query_selector(INDEED_COMPANY_SELECTOR)
        company = (await company_el.inner_text()).strip() if company_el else "Unknown"

        location_el = await card.query_selector(INDEED_LOCATION_SELECTOR)
        location = (await location_el.inner_text()).strip() if location_el else ""

        # Salary teaser shown on the card (best-effort; most postings omit it). Selector
        # verified against the live DOM (2026-06-24): the figure sits in a
        # ``li.salary-snippet-container`` (its inner_text is just the "$86,000-$112,000/yr" text).
        salary_el = await card.query_selector(INDEED_SALARY_SELECTOR)
        salary = (await salary_el.inner_text()).strip() if salary_el else None

        # The on-card snippet (best-effort) is the baseline description: it needs no click, so
        # every job carries SOME text even when the full-description pane can't be loaded.
        snippet = ""
        for selector in self._SNIPPET_SELECTORS:
            snippet_el = await card.query_selector(selector)
            if snippet_el:
                snippet = _clean_description((await snippet_el.inner_text()).strip())
                if snippet:
                    break

        return JobListing(
            title=title,
            company=company,
            url=url,  # type: ignore[arg-type]
            location=location,
            salary=salary or None,
            board=board,
            description=snippet,
        )

    async def _card_jk(self, card: ElementHandle, title_el: ElementHandle) -> str | None:
        """Indeed's stable job key, for a canonical URL + dedup. Tried on the card, then the
        title link, then any descendant carrying it. None when the live DOM doesn't expose it
        (the caller then falls back to the card href, so this can't regress)."""
        for el in (card, title_el):
            jk = await el.get_attribute("data-jk")
            if jk:
                return jk
        holder = await card.query_selector(INDEED_JK_SELECTOR)
        return await holder.get_attribute("data-jk") if holder else None

    async def _get_desc_text(self, page: Page) -> str:
        """Current right-pane description text (for change detection after a card click)."""
        for selector in self._DESC_SELECTORS:
            el = await page.query_selector(selector)
            if el:
                return (await el.inner_text()).strip()
        return ""

    async def _load_description(self, page: Page, card: ElementHandle, jk: str | None) -> str:
        """Click the card and return the cleaned full detail-pane description (≤5k chars).

        Indeed loads the description IN-PAGE and reflects the viewed job in the URL as
        ``&vjk=<jk>`` (verified against the live DOM). Waiting for that key to match the
        clicked card is what makes this reliable: it reads the right description even when the
        pane was *pre-opened* on the first result, where change-detection alone fails (the pane
        never "changes", so the old ``cur != prev`` check rejected a description that was right
        there — the measured cause of the first-card miss). Change-detection stays as a fallback
        for layouts/regions that don't put ``vjk`` in the URL. Best-effort: '' if nothing loads.
        """
        prev = await self._get_desc_text(page)
        await card.click(timeout=5_000)
        for _ in range(12):
            await random_delay(0.3, 0.5)
            text = await self._get_desc_text(page)
            if len(text) <= 100:
                continue
            # The pane belongs to THIS card once the viewed-job key in the URL matches it…
            if jk and f"vjk={jk}" in page.url:
                return _clean_description(text[:5000])
            # …or, where the URL carries no vjk, once it has visibly changed from the prior card
            # (guards against reading the previous card's still-displayed description).
            if not jk and text != prev:
                return _clean_description(text[:5000])
        return ""

    async def _dump_failed_card(self, page: Page, card: ElementHandle) -> None:
        """Capture everything that discriminates WHY a card yielded no full description, so a
        single user run pins the cause instead of another blind guess. Records the page
        URL/title (navigated? challenged?), whether each ``_DESC_SELECTOR`` is present vs
        present-but-empty, the live match counts for the snippet / data-jk / title selectors on
        the failing card, and the card + page HTML. Best-effort — never raises."""
        try:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            set_owner_only(_DEBUG_DIR, 0o700)
            (_DEBUG_DIR / "indeed-failed-card.html").write_text(
                await page.content(), encoding="utf-8"
            )
            lines = [
                f"page.url:   {page.url}",
                f"page.title: {await page.title()}",
                "",
                "description selectors (pane state):",
            ]
            for selector in self._DESC_SELECTORS:
                el = await page.query_selector(selector)
                if el is None:
                    lines.append(f"  {selector}: ABSENT")
                else:
                    text = (await el.inner_text()).strip()
                    lines.append(f"  {selector}: present, text len={len(text)}")
            lines += ["", "card selector match counts (on the failing card):"]
            for selector in (*self._SNIPPET_SELECTORS, INDEED_JK_SELECTOR, INDEED_TITLE_SELECTOR):
                try:
                    count = len(await card.query_selector_all(selector))
                except Exception:  # a bad selector must not break the diagnostic
                    count = -1
                lines.append(f"  {selector}: {count}")
            lines.append(f"  card data-jk attr: {await card.get_attribute('data-jk')!r}")
            lines += ["", "failing card inner_html:", await card.inner_html()]
            (_DEBUG_DIR / "indeed-failed-card.txt").write_text("\n".join(lines), encoding="utf-8")
            logger.warning("Indeed failing-card diagnostic written to %s", _DEBUG_DIR)
        except Exception as exc:  # diagnostics never break the scrape
            logger.warning("Could not write Indeed failing-card diagnostic: %s", exc)
