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
    # arg-aware: an href but no data-jk on the link
    title_el.get_attribute = AsyncMock(
        side_effect=lambda name: "/viewjob?jk=1" if name == "href" else None
    )
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
    card.get_attribute = AsyncMock(return_value=None)  # no data-jk on the card

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
    scraper._load_description = AsyncMock(return_value="")  # not under test here

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
    scraper._load_description = AsyncMock(return_value="")  # phase-2 not under test
    scraper._dump_failed_card = AsyncMock()  # no real card to dump from a mock

    def _stub(n: int) -> JobListing:
        return JobListing(
            title=f"E{n}", company="Co", url=f"https://indeed.com/jobs/{n}", board=JobBoard.INDEED
        )

    scraper._extract_job = AsyncMock(side_effect=[_stub(1), ValueError("bad card"), _stub(3)])

    msgs: list[str] = []
    jobs = await scraper.scrape(SearchParams(query="python", board=JobBoard.INDEED), msgs.append)

    # Phase 1 (card extraction) ticks per CARD — including the one whose extraction raises —
    # so the count never stalls (counts cards, not jobs).
    assert [m for m in msgs if m.startswith("Scraping job")] == [
        "Scraping job 1/3 on Indeed…",
        "Scraping job 2/3 on Indeed…",
        "Scraping job 3/3 on Indeed…",
    ]
    # Phase 2 (description enrichment) ticks only for the cards that actually parsed (1 and 3).
    assert [m for m in msgs if m.startswith("Reading description")] == [
        "Reading description 1/3 on Indeed…",
        "Reading description 3/3 on Indeed…",
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
    scraper._load_description = AsyncMock(return_value="")  # phase-2 not under test
    scraper._dump_failed_card = AsyncMock()

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


@pytest.mark.asyncio
async def test_extract_job_captures_card_snippet(app_settings: AppSettings) -> None:
    """Every card carries the on-card snippet as a baseline description (no click needed), so
    an Indeed job is never description-less even if the full pane can't be loaded."""
    scraper = IndeedScraper(MagicMock(), app_settings)

    title_el = AsyncMock()
    title_el.inner_text = AsyncMock(return_value="Backend Engineer")
    title_el.get_attribute = AsyncMock(
        side_effect=lambda name: "/viewjob?jk=1" if name == "href" else None
    )
    snippet_el = AsyncMock()
    snippet_el.inner_text = AsyncMock(return_value="Build Python APIs.\n\n\n\nAsync, Pydantic.")

    async def query(selector: str) -> object | None:
        if "jcs-JobTitle" in selector:
            return title_el
        if "jobsnippet" in selector or "job-snippet" in selector:
            return snippet_el
        return None

    card = MagicMock()
    card.query_selector = AsyncMock(side_effect=query)
    card.get_attribute = AsyncMock(return_value=None)

    job = await scraper._extract_job(card, JobBoard.INDEED)
    assert job is not None
    assert job.description.startswith("Build Python APIs.")
    assert "Async, Pydantic." in job.description
    assert "\n\n\n" not in job.description  # _clean_description collapsed the blank run


@pytest.mark.asyncio
async def test_load_description_change_detection_fallback_without_vjk(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no jk (href-fallback job), _load_description accepts the pane once it CHANGES from
    the prior card — guarding against reading the previous card's still-displayed text."""
    monkeypatch.setattr("job_applicator.scrapers.indeed.random_delay", AsyncMock())
    scraper = IndeedScraper(MagicMock(), app_settings)
    full = "Full job posting. " * 20  # > 100 chars
    reads = iter(["previous card text", full])  # pre-click read, then the loaded pane

    async def fake_get(_page: object) -> str:
        return next(reads)

    monkeypatch.setattr(scraper, "_get_desc_text", fake_get)
    card = AsyncMock()

    desc = await scraper._load_description(AsyncMock(), card, jk=None)
    assert desc.startswith("Full job posting.")
    card.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_description_reads_preopened_first_card_via_vjk(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured first-card bug: Indeed pre-opens the first result, so the pane never
    CHANGES after the click — but the URL carries vjk=<jk>. _load_description must read that
    description (matching vjk) even though text == prev, instead of returning ''."""
    monkeypatch.setattr("job_applicator.scrapers.indeed.random_delay", AsyncMock())
    scraper = IndeedScraper(MagicMock(), app_settings)
    pane = "Pre-opened first job description. " * 10  # > 100, and UNCHANGED across the click
    monkeypatch.setattr(scraper, "_get_desc_text", AsyncMock(return_value=pane))

    page = AsyncMock()
    page.url = "https://ca.indeed.com/jobs?q=python&vjk=abc123"  # viewed job == the clicked card
    card = AsyncMock()

    desc = await scraper._load_description(page, card, jk="abc123")
    assert desc.startswith("Pre-opened first job description.")  # not '' despite no change


@pytest.mark.asyncio
async def test_scrape_enriches_snippet_with_full_description(
    app_settings: AppSettings, tmp_path: object
) -> None:
    """The card snippet is upgraded to the full pane description when it loads (longer wins)."""
    app_settings.target.indeed_domain = "www.indeed.com"
    scraper = IndeedScraper(MagicMock(), app_settings)
    scraper.COOKIE_PATH = tmp_path / "indeed.json"  # type: ignore[operator,assignment]
    scraper._browser.persistent_context = AsyncMock(return_value=MagicMock())
    scraper._new_stealth_page = AsyncMock(return_value=AsyncMock())
    scraper._load_results = AsyncMock(return_value=[MagicMock()])
    scraper._extract_job = AsyncMock(
        return_value=JobListing(
            title="E",
            company="Co",
            url="https://indeed.com/1",
            board=JobBoard.INDEED,
            description="short snippet",
        )
    )
    full = "A much longer full description with the real posting body. " * 5
    scraper._load_description = AsyncMock(return_value=full)

    jobs = await scraper.scrape(SearchParams(query="x", board=JobBoard.INDEED))
    assert jobs[0].description == full  # upgraded from snippet → full


@pytest.mark.asyncio
async def test_extract_job_captures_salary(app_settings: AppSettings) -> None:
    """The on-card salary teaser is captured into JobListing.salary (best-effort)."""
    scraper = IndeedScraper(MagicMock(), app_settings)

    title_el = AsyncMock()
    title_el.inner_text = AsyncMock(return_value="Dev")
    title_el.get_attribute = AsyncMock(
        side_effect=lambda name: "/viewjob?jk=1" if name == "href" else None
    )
    salary_el = AsyncMock()
    salary_el.inner_text = AsyncMock(return_value="$86,000–$112,000 a year")  # noqa: RUF001

    async def query(selector: str) -> object | None:
        if "jcs-JobTitle" in selector:
            return title_el
        if "salary" in selector:
            return salary_el
        return None

    card = MagicMock()
    card.query_selector = AsyncMock(side_effect=query)
    card.get_attribute = AsyncMock(return_value=None)

    job = await scraper._extract_job(card, JobBoard.INDEED)
    assert job is not None
    assert job.salary == "$86,000–$112,000 a year"  # noqa: RUF001


@pytest.mark.asyncio
async def test_extract_job_uses_data_jk_for_canonical_url(app_settings: AppSettings) -> None:
    """The job key (data-jk) yields a canonical /viewjob URL — NOT the sponsored /pagead/clk
    tracking redirect — so the ad and organic copies of one job dedupe to the same URL."""
    app_settings.target.indeed_domain = "ca.indeed.com"
    scraper = IndeedScraper(MagicMock(), app_settings)

    title_el = AsyncMock()
    title_el.inner_text = AsyncMock(return_value="Dev")
    title_el.get_attribute = AsyncMock(  # an AD card: href is a tracking redirect
        side_effect=lambda name: "/pagead/clk?ad=xyz" if name == "href" else None
    )

    async def query(selector: str) -> object | None:
        return title_el if ("jcs-JobTitle" in selector or "h2 a" in selector) else None

    card = MagicMock()
    card.query_selector = AsyncMock(side_effect=query)
    card.get_attribute = AsyncMock(return_value="abc123")  # data-jk on the card

    job = await scraper._extract_job(card, JobBoard.INDEED)
    assert job is not None
    assert "viewjob?jk=abc123" in str(job.url)  # canonical, not the /pagead/clk redirect
    assert "pagead" not in str(job.url)


@pytest.mark.asyncio
async def test_scrape_dedupes_a_job_listed_as_ad_and_organic(
    app_settings: AppSettings, tmp_path: object
) -> None:
    """Indeed shows the same job twice (sponsored + organic); the canonical-URL dedup in pass 1
    collapses the two cards into one job rather than storing/counting it twice."""
    app_settings.target.indeed_domain = "www.indeed.com"
    scraper = IndeedScraper(MagicMock(), app_settings)
    scraper.COOKIE_PATH = tmp_path / "indeed.json"  # type: ignore[operator,assignment]
    scraper._browser.persistent_context = AsyncMock(return_value=MagicMock())
    scraper._new_stealth_page = AsyncMock(return_value=AsyncMock())
    scraper._load_results = AsyncMock(return_value=[MagicMock(), MagicMock()])  # two cards…
    scraper._extract_job = AsyncMock(  # …both resolving to the same canonical URL
        return_value=JobListing(
            title="E",
            company="Co",
            url="https://www.indeed.com/viewjob?jk=dup",
            board=JobBoard.INDEED,
        )
    )
    scraper._load_description = AsyncMock(return_value="")

    jobs = await scraper.scrape(SearchParams(query="x", board=JobBoard.INDEED))
    assert len(jobs) == 1  # the duplicate was collapsed, not stored twice


@pytest.mark.asyncio
async def test_scrape_keeps_snippet_and_dumps_failing_card(
    app_settings: AppSettings, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-effort: if a card's full description doesn't load, the snippet is kept (job NOT
    dropped) and a rich failing-card diagnostic is written so ONE live run pins the cause."""
    from pathlib import Path

    debug_dir = tmp_path / "debug"  # type: ignore[operator]
    monkeypatch.setattr("job_applicator.scrapers.indeed._DEBUG_DIR", debug_dir)
    app_settings.target.indeed_domain = "www.indeed.com"
    scraper = IndeedScraper(MagicMock(), app_settings)
    scraper.COOKIE_PATH = tmp_path / "indeed.json"  # type: ignore[operator,assignment]
    scraper._browser.persistent_context = AsyncMock(return_value=MagicMock())

    page = AsyncMock()
    page.url = "https://www.indeed.com/jobs?q=x"
    page.title = AsyncMock(return_value="Jobs")
    page.content = AsyncMock(return_value="<html><body>pane</body></html>")
    page.query_selector = AsyncMock(return_value=None)  # description pane absent
    scraper._new_stealth_page = AsyncMock(return_value=page)

    card = AsyncMock()
    card.query_selector_all = AsyncMock(return_value=[])
    card.get_attribute = AsyncMock(return_value=None)
    card.inner_html = AsyncMock(return_value="<div data-jk='1'>card</div>")
    scraper._load_results = AsyncMock(return_value=[card])
    scraper._extract_job = AsyncMock(
        return_value=JobListing(
            title="E",
            company="Co",
            url="https://indeed.com/1",
            board=JobBoard.INDEED,
            description="snippet text",
        )
    )
    scraper._load_description = AsyncMock(return_value="")  # full description fails

    jobs = await scraper.scrape(SearchParams(query="x", board=JobBoard.INDEED))
    assert jobs[0].description == "snippet text"  # snippet kept, job NOT dropped
    assert (Path(debug_dir) / "indeed-failed-card.html").exists()
    diag = (Path(debug_dir) / "indeed-failed-card.txt").read_text(encoding="utf-8")
    assert "page.url:" in diag  # navigated/challenged discriminator
    assert "card data-jk attr:" in diag  # confirms whether data-jk exists for the dedup fix
