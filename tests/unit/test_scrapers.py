"""Unit tests for scrapers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import Error as PlaywrightError

from job_applicator.config import AppSettings
from job_applicator.exceptions import LoginRequiredError, RateLimitError, ScraperError
from job_applicator.scrapers.base import SearchParams
from job_applicator.scrapers.linkedin import LinkedInScraper, _is_authenticated_url


@pytest.mark.asyncio
async def test_scrape_without_session_directs_to_login(app_settings: AppSettings) -> None:
    """With no active session, scrape() must raise a LoginRequiredError that
    directs the user to the safe `job-applicator login` flow — and must never
    attempt an automated credential login (which would trip LinkedIn's CAPTCHA
    and raise the account's risk score)."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._get_context = AsyncMock(return_value=MagicMock())
    scraper._ensure_session = AsyncMock(return_value=False)
    scraper.login = AsyncMock()

    with pytest.raises(LoginRequiredError) as excinfo:
        await scraper.scrape(SearchParams(query="python"))

    assert "job-applicator login" in str(excinfo.value)
    scraper.login.assert_not_called()  # the risky automated path is never taken


@pytest.mark.asyncio
async def test_login_is_disabled_for_account_safety(app_settings: AppSettings) -> None:
    """Automated credential login is disabled: it returns False without touching
    the browser (no context/page, no form submit)."""
    mock_browser = MagicMock()
    scraper = LinkedInScraper(mock_browser, app_settings)

    result = await scraper.login("user@example.com", "secret")

    assert result is False
    mock_browser.persistent_context.assert_not_called()


@pytest.mark.asyncio
async def test_load_cookies_success(
    app_settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use a temp cookie path so the test never touches the user's real session.
    monkeypatch.setattr(LinkedInScraper, "COOKIE_PATH", tmp_path / "linkedin.json")
    scraper = LinkedInScraper(MagicMock(), app_settings)
    mock_context = AsyncMock()
    mock_context.add_cookies = AsyncMock()

    cookie_data = {"cookies": [{"name": "li_at", "value": "test", "domain": ".linkedin.com"}]}
    scraper._cookie_file.parent.mkdir(parents=True, exist_ok=True)
    scraper._cookie_file.write_text(json.dumps(cookie_data))

    result = await scraper._load_cookies(mock_context)
    assert result is True
    mock_context.add_cookies.assert_called_once()


@pytest.mark.asyncio
async def test_save_cookies(
    app_settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(LinkedInScraper, "COOKIE_PATH", tmp_path / "linkedin.json")
    scraper = LinkedInScraper(MagicMock(), app_settings)
    mock_context = AsyncMock()
    mock_context.cookies = AsyncMock(return_value=[{"name": "li_at", "value": "test"}])

    await scraper._save_cookies(mock_context)

    assert scraper._cookie_file.exists()
    data = json.loads(scraper._cookie_file.read_text())
    assert data["cookies"] == [{"name": "li_at", "value": "test"}]
    # The session cookie file must not be group/world-readable.
    assert (scraper._cookie_file.stat().st_mode & 0o077) == 0


@pytest.mark.asyncio
async def test_load_cookies_missing_file(
    app_settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(LinkedInScraper, "COOKIE_PATH", tmp_path / "missing.json")
    scraper = LinkedInScraper(MagicMock(), app_settings)
    mock_context = AsyncMock()

    result = await scraper._load_cookies(mock_context)
    assert result is False


@pytest.mark.asyncio
async def test_ensure_session_true_when_logged_in(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ensure_session reports True (and sets the flag) when /feed loads."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._load_cookies = AsyncMock(return_value=False)
    page = AsyncMock()
    page.url = "https://www.linkedin.com/feed/"
    page.title = AsyncMock(return_value="Feed | LinkedIn")
    scraper._new_stealth_page = AsyncMock(return_value=page)

    assert await scraper._ensure_session(MagicMock()) is True
    assert scraper._logged_in is True
    page.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_session_false_when_logged_out(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ensure_session reports False when redirected to the auth wall (no submit)."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._load_cookies = AsyncMock(return_value=False)
    page = AsyncMock()
    page.url = "https://www.linkedin.com/uas/login"
    page.title = AsyncMock(return_value="Sign In | LinkedIn")
    scraper._new_stealth_page = AsyncMock(return_value=page)

    assert await scraper._ensure_session(MagicMock()) is False
    assert scraper._logged_in is False


@pytest.mark.asyncio
async def test_ensure_session_raises_scraper_error_on_security_checkpoint(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A LinkedIn security checkpoint must raise a distinct ScraperError, NOT be mis-diagnosed as
    'no session' (False -> LoginRequiredError -> 'run login'), which is the wrong remedy for a
    possibly-flagged account (re-running login automation raises the account's risk)."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._load_cookies = AsyncMock(return_value=False)
    page = AsyncMock()
    page.url = "https://www.linkedin.com/checkpoint/challenge/"
    page.title = AsyncMock(return_value="Security Verification | LinkedIn")
    scraper._new_stealth_page = AsyncMock(return_value=page)
    with pytest.raises(ScraperError):
        await scraper._ensure_session(MagicMock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "title",
    [
        "You've reached the weekly limit",
        "Monthly limit reached",
        "You've reached the commercial use limit",
        "Too many requests",
    ],
)
async def test_raise_if_blocked_rate_limit_raises_rate_limit_error(
    app_settings: AppSettings, title: str
) -> None:
    """A LinkedIn rate-limit / usage-limit interstitial (weekly/monthly/commercial-use/too-many)
    raises the typed RateLimitError (so a caller can back off), not the generic 'blocked' error."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.title = AsyncMock(return_value=title)
    with pytest.raises(RateLimitError):
        await scraper._raise_if_blocked(page)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "title",
    ["Feed | LinkedIn", "Verify your email | LinkedIn"],  # the 2nd is a benign near-miss
)
async def test_raise_if_blocked_passes_benign_pages(app_settings: AppSettings, title: str) -> None:
    """A normal page — including a benign 'verify your email' page — is NOT a block: only the
    CAPTCHA wording 'verify you are ...' counts, so a legitimate run is never aborted."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.title = AsyncMock(return_value=title)
    await scraper._raise_if_blocked(page)  # no raise


@pytest.mark.asyncio
async def test_interactive_login_saves_session_on_detect(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """interactive_login detects the logged-in feed and persists the session."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    monkeypatch.setattr("job_applicator.scrapers.linkedin.navigate", AsyncMock())
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._get_context = AsyncMock(return_value=MagicMock())
    scraper._ensure_session = AsyncMock(return_value=False)
    scraper._save_cookies = AsyncMock()
    page = AsyncMock()
    page.url = "https://www.linkedin.com/feed/"
    scraper._new_stealth_page = AsyncMock(return_value=page)

    assert await scraper.interactive_login(timeout_s=5) is True
    assert scraper._logged_in is True
    scraper._save_cookies.assert_awaited_once()


@pytest.mark.asyncio
async def test_interactive_login_opens_login_page_on_checkpoint(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint during the pre-check must NOT abort `login` — opening the login page is exactly
    how the user manually clears a checkpoint. interactive_login falls through to the manual
    flow."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    navigated = AsyncMock()
    monkeypatch.setattr("job_applicator.scrapers.linkedin.navigate", navigated)
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._get_context = AsyncMock(return_value=MagicMock())
    scraper._ensure_session = AsyncMock(side_effect=ScraperError("checkpoint"))  # pre-check blocks
    scraper._save_cookies = AsyncMock()
    page = AsyncMock()
    page.url = "https://www.linkedin.com/feed/"  # poll detects sign-in once the manual flow opens
    scraper._new_stealth_page = AsyncMock(return_value=page)

    result = await scraper.interactive_login(timeout_s=5)

    assert result is True  # did NOT abort — fell through, opened login, detected sign-in
    navigated.assert_awaited()  # the login page was opened (the manual remedy for a checkpoint)


def test_is_authenticated_url_rejects_logged_out_redirect() -> None:
    """The logged-out /feed redirect embeds 'feed' in the query string, so a
    substring check would false-positive; the path-based check must reject it."""
    logged_out = (
        "https://www.linkedin.com/uas/login"
        "?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2F"
    )
    assert _is_authenticated_url(logged_out) is False
    assert _is_authenticated_url("https://www.linkedin.com/checkpoint/challenge/") is False
    assert _is_authenticated_url("https://www.linkedin.com/feed/") is True
    assert _is_authenticated_url("https://www.linkedin.com/mynetwork/") is True


@pytest.mark.asyncio
async def test_has_active_session_false_on_transient_error(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient feed-load failure degrades to False (not a raised traceback),
    so `import-cookies --verify` reports cleanly instead of crashing."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._get_context = AsyncMock(return_value=MagicMock())
    scraper._load_cookies = AsyncMock(return_value=False)
    page = AsyncMock()
    page.goto = AsyncMock(side_effect=PlaywrightError("net::ERR_TIMED_OUT"))
    scraper._new_stealth_page = AsyncMock(return_value=page)

    assert await scraper.has_active_session() is False  # NavigationError swallowed
    page.close.assert_awaited_once()  # page still cleaned up


@pytest.mark.asyncio
async def test_has_active_session_false_on_anti_bot_block(app_settings: AppSettings) -> None:
    """A best-effort session check degrades an anti-bot block (ScraperError) to False, like a
    transient failure — so `import-cookies --verify` reports cleanly, not a raw traceback."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._get_context = AsyncMock(return_value=MagicMock())
    scraper._ensure_session = AsyncMock(side_effect=ScraperError("checkpoint"))
    assert await scraper.has_active_session() is False


@pytest.mark.asyncio
async def test_load_cookies_skips_invalid_cookie(
    app_settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single invalid cookie must not void an otherwise-valid session
    (context.add_cookies is all-or-nothing, so we fall back to per-cookie)."""
    monkeypatch.setattr(LinkedInScraper, "COOKIE_PATH", tmp_path / "linkedin.json")
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._cookie_file.write_text(
        json.dumps({"cookies": [{"name": "li_at", "value": "good"}, {"name": "bad", "value": "x"}]})
    )

    async def add_cookies(cks: list[dict[str, str]]) -> None:
        if len(cks) > 1:  # whole batch rejected because of the bad cookie
            raise PlaywrightError("batch rejected")
        if cks[0]["name"] == "bad":
            raise PlaywrightError("bad cookie")

    context = AsyncMock()
    context.add_cookies = add_cookies

    assert await scraper._load_cookies(context) is True  # the good cookie still loaded


@pytest.mark.asyncio
async def test_check_session_healthy(app_settings: AppSettings) -> None:
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._get_context = AsyncMock(return_value=MagicMock())
    scraper._ensure_session = AsyncMock(return_value=True)

    health = await scraper.check_session()

    assert health.healthy
    assert health.board.value == "linkedin"
    assert "active" in health.details.lower()


@pytest.mark.asyncio
async def test_check_session_unhealthy_directs_to_login(app_settings: AppSettings) -> None:
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._get_context = AsyncMock(return_value=MagicMock())
    scraper._ensure_session = AsyncMock(return_value=False)

    health = await scraper.check_session()

    assert not health.healthy
    assert "job-applicator login" in health.details


@pytest.mark.asyncio
async def test_scrape_emits_per_job_progress(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scrape() ticks on_progress once per EXTRACTED job (1-based, /N) during the description
    pass — a card that yields no metadata (returns None or raises) is dropped in the metadata
    snapshot pass and never reaches the counter, so N is the count of jobs found, not raw cards
    seen. (The two-pass split exists so an early card's click can't detach later card handles.)"""
    from job_applicator.models import JobBoard, JobListing

    monkeypatch.setattr("job_applicator.scrapers.linkedin.navigate", AsyncMock())
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    monkeypatch.setattr(
        "job_applicator.scrapers.linkedin.wait_for_selector", AsyncMock(return_value=True)
    )
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._logged_in = True  # skip the session check; we exercise the scrape loop
    scraper._get_context = AsyncMock(return_value=MagicMock())
    page = AsyncMock()
    page.query_selector_all = AsyncMock(
        return_value=[MagicMock(click=AsyncMock()) for _ in range(4)]
    )
    scraper._new_stealth_page = AsyncMock(return_value=page)
    scraper._get_desc_text = AsyncMock(return_value="")  # no description-change → no early break
    scraper._extract_description = AsyncMock(return_value="")

    def _stub(n: int) -> JobListing:
        return JobListing(
            title=f"E{n}",
            company="Co",
            url=f"https://linkedin.com/jobs/{n}",
            board=JobBoard.LINKEDIN,
        )

    # card 2 → None, card 3 → raises: both drop out in the metadata pass and are NOT counted;
    # only the 2 cards that yield metadata (E1, E4) reach the per-job description counter.
    scraper._extract_job = AsyncMock(side_effect=[_stub(1), None, ValueError("bad card"), _stub(4)])

    msgs: list[str] = []
    jobs = await scraper.scrape(SearchParams(query="python", board=JobBoard.LINKEDIN), msgs.append)

    assert msgs == [
        "Scraping job 1/2 on LinkedIn…",
        "Scraping job 2/2 on LinkedIn…",
    ]
    assert [j.title for j in jobs] == ["E1", "E4"]  # 2 of 4 cards yielded metadata → progress /2


@pytest.mark.asyncio
async def test_scrape_streams_each_job_via_on_job(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scrape() calls on_job with each fully-parsed listing as it lands (streaming) — and
    only with real jobs (a card whose extraction returns None is ticked for progress but
    NOT streamed). The streamed sequence equals the returned list."""
    from job_applicator.models import JobBoard, JobListing

    monkeypatch.setattr("job_applicator.scrapers.linkedin.navigate", AsyncMock())
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    monkeypatch.setattr(
        "job_applicator.scrapers.linkedin.wait_for_selector", AsyncMock(return_value=True)
    )
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._logged_in = True
    scraper._get_context = AsyncMock(return_value=MagicMock())
    page = AsyncMock()
    page.query_selector_all = AsyncMock(
        return_value=[MagicMock(click=AsyncMock()) for _ in range(3)]
    )
    scraper._new_stealth_page = AsyncMock(return_value=page)
    scraper._get_desc_text = AsyncMock(return_value="")
    scraper._extract_description = AsyncMock(return_value="")

    def _stub(n: int) -> JobListing:
        return JobListing(
            title=f"E{n}",
            company="Co",
            url=f"https://linkedin.com/jobs/{n}",
            board=JobBoard.LINKEDIN,
        )

    scraper._extract_job = AsyncMock(side_effect=[_stub(1), None, _stub(3)])  # card 2 fails

    streamed: list[JobListing] = []
    jobs = await scraper.scrape(
        SearchParams(query="python", board=JobBoard.LINKEDIN), on_job=streamed.append
    )
    assert [j.title for j in streamed] == ["E1", "E3"]  # only real jobs streamed
    assert [j.title for j in jobs] == ["E1", "E3"]  # streamed sequence == returned list


@pytest.mark.asyncio
async def test_scrape_without_on_progress_is_safe(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_progress is optional — the existing CLI/batch callers pass none and scrape() must
    not blow up (the per-card hook is guarded)."""
    from job_applicator.models import JobBoard, JobListing

    monkeypatch.setattr("job_applicator.scrapers.linkedin.navigate", AsyncMock())
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    monkeypatch.setattr(
        "job_applicator.scrapers.linkedin.wait_for_selector", AsyncMock(return_value=True)
    )
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._logged_in = True
    scraper._get_context = AsyncMock(return_value=MagicMock())
    page = AsyncMock()
    page.query_selector_all = AsyncMock(return_value=[MagicMock(click=AsyncMock())])
    scraper._new_stealth_page = AsyncMock(return_value=page)
    scraper._get_desc_text = AsyncMock(return_value="")
    scraper._extract_description = AsyncMock(return_value="")
    scraper._extract_job = AsyncMock(
        return_value=JobListing(
            title="E1", company="Co", url="https://linkedin.com/jobs/1", board=JobBoard.LINKEDIN
        )
    )

    jobs = await scraper.scrape(SearchParams(query="python", board=JobBoard.LINKEDIN))
    assert [j.title for j in jobs] == ["E1"]


@pytest.mark.asyncio
async def test_check_session_graceful_on_navigation_error(
    app_settings: AppSettings,
) -> None:
    from job_applicator.exceptions import NavigationError

    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._get_context = AsyncMock(return_value=MagicMock())
    scraper._ensure_session = AsyncMock(side_effect=NavigationError("timeout"))

    health = await scraper.check_session()

    assert not health.healthy
    assert "Could not reach LinkedIn" in health.details


@pytest.mark.asyncio
async def test_linkedin_extract_job_captures_salary_when_present(
    app_settings: AppSettings,
) -> None:
    """Best-effort LinkedIn salary capture. Selectors are unverified against the live DOM
    (LinkedIn is never auto-searched), so this asserts the wiring: when a salary element is
    present its text lands in JobListing.salary; when absent, salary stays None (no crash)."""
    from job_applicator.models import JobBoard

    scraper = LinkedInScraper(MagicMock(), app_settings)
    title_el = AsyncMock()
    title_el.inner_text = AsyncMock(return_value="Backend Engineer")
    title_el.get_attribute = AsyncMock(return_value="/jobs/view/1")
    salary_el = AsyncMock()
    salary_el.inner_text = AsyncMock(return_value="$120,000/yr - $150,000/yr")

    async def query(selector: str) -> object | None:
        if "title" in selector:
            return title_el
        if "salary" in selector or "compensation" in selector:
            return salary_el
        return None

    card = MagicMock()
    card.query_selector = AsyncMock(side_effect=query)

    job = await scraper._extract_job(card, JobBoard.LINKEDIN)
    assert job is not None
    assert job.salary == "$120,000/yr - $150,000/yr"


@pytest.mark.asyncio
async def test_linkedin_extract_job_salary_none_when_absent(app_settings: AppSettings) -> None:
    from job_applicator.models import JobBoard

    scraper = LinkedInScraper(MagicMock(), app_settings)
    title_el = AsyncMock()
    title_el.inner_text = AsyncMock(return_value="Backend Engineer")
    title_el.get_attribute = AsyncMock(return_value="/jobs/view/1")

    async def query(selector: str) -> object | None:
        return title_el if "title" in selector else None

    card = MagicMock()
    card.query_selector = AsyncMock(side_effect=query)

    job = await scraper._extract_job(card, JobBoard.LINKEDIN)
    assert job is not None
    assert job.salary is None


async def _extract_raises(*_a: object, **_k: object) -> None:
    """_extract_job stub: extraction throws → the loop's `failures` counter increments."""
    raise RuntimeError("stale field selector")


async def _extract_returns_none(*_a: object, **_k: object) -> None:
    """_extract_job stub: stale TITLE/href selector → returns None with NO exception, so the
    loop's `failures` counter stays 0 (the R1 path a `failures`-keyed guard silently missed)."""
    return None


@pytest.mark.parametrize(
    "extract_stub",
    [_extract_raises, _extract_returns_none],
    ids=["cards-raise", "cards-return-none-R1"],
)
async def test_scrape_listings_cards_present_but_zero_jobs_raises(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch, extract_stub: object
) -> None:
    """LinkedIn: cards present but EVERY extraction yields no job → ScraperError, never a silent
    empty list (the honest-failure twin of the Indeed guard).

    Both failure sub-modes must fail loud and are parametrized: extraction RAISING (`failures` > 0)
    and extraction RETURNING None (a stale TITLE/href selector — `failures` stays 0; the R1
    regression the old `failures`-keyed guard returned [] on). The guard keys on cards being
    present (`total`), not on `failures`."""
    import job_applicator.scrapers.linkedin as lk
    from job_applicator.exceptions import ScraperError
    from job_applicator.models import JobBoard
    from job_applicator.scrapers.base import SearchParams

    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = MagicMock()
    page.query_selector_all = AsyncMock(return_value=[MagicMock(), MagicMock()])
    page.close = AsyncMock()

    async def _page(_ctx: object) -> object:
        return page

    async def _noop(*_a: object, **_k: object) -> None:
        return None

    async def _wait(*_a: object, **_k: object) -> bool:
        return True

    monkeypatch.setattr(scraper, "_new_stealth_page", _page)
    monkeypatch.setattr(lk, "navigate", _noop)
    monkeypatch.setattr(lk, "random_delay", _noop)
    monkeypatch.setattr(lk, "wait_for_selector", _wait)
    monkeypatch.setattr(scraper, "_extract_job", extract_stub)

    params = SearchParams(query="python", max_results=5, board=JobBoard.LINKEDIN)
    with pytest.raises(ScraperError):
        await scraper._scrape_listings(params, MagicMock())


async def test_scrape_listings_no_cards_raises_scraper_error(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LinkedIn: 0 job cards found → ScraperError (0 cards is ambiguous empty-vs-blocked; fail
    loudly), never a silent empty list."""
    import job_applicator.scrapers.linkedin as lk
    from job_applicator.exceptions import ScraperError
    from job_applicator.models import JobBoard
    from job_applicator.scrapers.base import SearchParams

    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = MagicMock()
    page.query_selector_all = AsyncMock(return_value=[])  # no cards
    page.close = AsyncMock()
    page.url = "https://www.linkedin.com/jobs/search"
    page.title = AsyncMock(return_value="Jobs | LinkedIn")  # not a block; falls to the guard

    async def _page(_ctx: object) -> object:
        return page

    async def _noop(*_a: object, **_k: object) -> None:
        return None

    async def _nowait(*_a: object, **_k: object) -> bool:
        return False  # no container selector ever resolves

    monkeypatch.setattr(scraper, "_new_stealth_page", _page)
    monkeypatch.setattr(lk, "navigate", _noop)
    monkeypatch.setattr(lk, "random_delay", _noop)
    monkeypatch.setattr(lk, "wait_for_selector", _nowait)

    params = SearchParams(query="python", max_results=5, board=JobBoard.LINKEDIN)
    with pytest.raises(ScraperError):
        await scraper._scrape_listings(params, MagicMock())


# --------------------------------------------------- geoId resolution (#4 fix)
def _page_typeahead(*payloads: object) -> MagicMock:
    """Mock Page whose context ``.request.get`` returns each payload in turn.

    Each payload is a ``list`` (typeahead hits, HTTP 200) or ``None`` (an HTTP error response).
    """
    page = MagicMock()
    resps = []
    for payload in payloads:
        r = MagicMock()
        r.ok = payload is not None
        r.status = 200 if payload is not None else 500
        r.json = AsyncMock(return_value=payload)
        resps.append(r)
    page.request.get = AsyncMock(side_effect=resps)
    return page


def test_build_search_url_threads_geoid(app_settings: AppSettings) -> None:
    """A resolved geoId is threaded into the search URL — LinkedIn geo-filters on it, not on the
    human location string."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    url = scraper._build_search_url(
        SearchParams(query="SOC analyst", location="Montréal, QC"), geo_id="101330853"
    )
    assert "geoId=101330853" in url
    assert "keywords=SOC+analyst" in url


def test_build_search_url_omits_geoid_when_unresolved(app_settings: AppSettings) -> None:
    """No geoId (resolution failed) → the URL omits it and keeps the raw location (a degraded but
    honest filter, not a fabricated id)."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    url = scraper._build_search_url(SearchParams(query="x", location="Nowhere"), geo_id=None)
    assert "geoId" not in url
    assert "location=Nowhere" in url


@pytest.mark.asyncio
async def test_resolve_geo_id_returns_first_hit(app_settings: AppSettings) -> None:
    """The first typeahead hit's numeric id is returned; region bias ranks the right city first."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = _page_typeahead([{"id": "101330853", "displayName": "Montreal, Quebec, Canada"}])
    assert await scraper._resolve_geo_id(page, "Montréal, QC") == "101330853"
    page.request.get.assert_awaited_once()  # raw query hit; no city retry needed


@pytest.mark.asyncio
async def test_resolve_geo_id_retries_city_on_empty_abbreviation(
    app_settings: AppSettings,
) -> None:
    """LinkedIn's typeahead rejects the 'City, ST' abbreviation (→ []); the resolver retries with
    the bare city (before the first comma) before giving up. Regression for the measured
    ``Montreal, QC`` → ``[]`` gap that left searches geo-unconstrained (89% off-geo)."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = _page_typeahead([], [{"id": "101330853", "displayName": "Montreal, Quebec, Canada"}])
    assert await scraper._resolve_geo_id(page, "Montreal, QC") == "101330853"
    assert page.request.get.await_count == 2  # raw [] → city retry


@pytest.mark.asyncio
async def test_resolve_geo_id_none_when_unresolvable(app_settings: AppSettings) -> None:
    """All candidates empty → None → caller falls back to the raw location (never a fake id)."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = _page_typeahead([], [])
    assert await scraper._resolve_geo_id(page, "Atlantis, XX") is None


@pytest.mark.asyncio
async def test_resolve_geo_id_none_on_request_error(app_settings: AppSettings) -> None:
    """A typeahead transport error is a failure, not a fabricated id — return None (degraded)."""
    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = MagicMock()
    page.request.get = AsyncMock(side_effect=PlaywrightError("network down"))
    assert await scraper._resolve_geo_id(page, "Montreal") is None


# ------------------------------------------- card-iteration robustness (#3 fix)
def test_job_id_from_url_parses_view_and_current_job_id() -> None:
    """The job id is parsed from both the ``/jobs/view/<id>`` href and the ``currentJobId=<id>``
    query form; an id-less URL yields ``""`` (→ the caller keeps the metadata-only listing)."""
    from job_applicator.scrapers.linkedin import _job_id_from_url

    assert _job_id_from_url("https://www.linkedin.com/jobs/view/4123456789/?trk=x") == "4123456789"
    assert (
        _job_id_from_url("https://www.linkedin.com/jobs/search?currentJobId=987654321&x=1")
        == "987654321"
    )
    assert _job_id_from_url("https://www.linkedin.com/jobs/collections/recommended") == ""


@pytest.mark.asyncio
async def test_scrape_keeps_metadata_when_description_unavailable(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3: a card whose description can't be re-resolved (its handle was detached by an earlier
    card's click re-rendering the virtualized list) is kept as a metadata-only listing, NOT
    dropped. The whole point of the two-pass split — a description miss costs the description,
    never the whole job (the measured 'lost a fixed ~11-card batch' regression)."""
    from job_applicator.models import JobBoard, JobListing

    monkeypatch.setattr("job_applicator.scrapers.linkedin.navigate", AsyncMock())
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    monkeypatch.setattr(
        "job_applicator.scrapers.linkedin.wait_for_selector", AsyncMock(return_value=True)
    )
    scraper = LinkedInScraper(MagicMock(), app_settings)
    scraper._logged_in = True
    scraper._get_context = AsyncMock(return_value=MagicMock())
    page = AsyncMock()
    page.query_selector_all = AsyncMock(return_value=[MagicMock() for _ in range(3)])
    scraper._new_stealth_page = AsyncMock(return_value=page)

    def _stub(n: int) -> JobListing:
        return JobListing(
            title=f"E{n}",
            company="Co",
            url=f"https://linkedin.com/jobs/view/{n}",
            board=JobBoard.LINKEDIN,
        )

    scraper._extract_job = AsyncMock(side_effect=[_stub(1), _stub(2), _stub(3)])
    # Every description load fails (re-resolution finds nothing) — the jobs must survive anyway.
    scraper._load_job_description = AsyncMock(return_value="")

    jobs = await scraper.scrape(SearchParams(query="python", board=JobBoard.LINKEDIN))

    assert [j.title for j in jobs] == ["E1", "E2", "E3"]  # all 3 kept despite 0 descriptions
    assert all(j.description == "" for j in jobs)


# ------------------------------------------ review fixes (#140 code-review)
def _desc_link(href: str) -> MagicMock:
    link = MagicMock()
    link.get_attribute = AsyncMock(return_value=href)
    link.scroll_into_view_if_needed = AsyncMock()
    link.click = AsyncMock()
    return link


@pytest.mark.asyncio
async def test_load_job_description_returns_empty_on_timeout(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONFIRMED review defect: when the detail panel never confirms THIS job (its text never
    changes AND the URL's currentJobId doesn't match), return '' — NEVER the previously-selected
    job's still-displayed text (which would silently attach the WRONG description)."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = MagicMock()
    page.query_selector_all = AsyncMock(return_value=[_desc_link("/jobs/view/999/")])
    page.url = "https://www.linkedin.com/jobs/search?currentJobId=111"  # a DIFFERENT job selected
    stale = "PREVIOUS job's description text that never changes. " * 5
    scraper._get_desc_text = AsyncMock(return_value=stale)  # panel stuck on the prior job
    scraper._extract_description = AsyncMock(return_value=stale)

    result = await scraper._load_job_description(page, "https://www.linkedin.com/jobs/view/999/")
    assert result == ""  # a miss, not the stale prior text
    scraper._extract_description.assert_not_called()  # the panel was never trusted


@pytest.mark.asyncio
async def test_load_job_description_accepts_autoselected_via_url(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auto-selected first card — whose panel text doesn't CHANGE on click — is still accepted
    when the URL's currentJobId matches this job, so it isn't lost as a false timeout."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    scraper = LinkedInScraper(MagicMock(), app_settings)
    page = MagicMock()
    page.query_selector_all = AsyncMock(return_value=[_desc_link("/jobs/view/777/")])
    page.url = "https://www.linkedin.com/jobs/search?currentJobId=777"  # panel shows THIS job
    desc = "This job's full description text, well over one hundred characters long. " * 3
    scraper._get_desc_text = AsyncMock(return_value=desc)  # unchanged (auto-selected)
    scraper._extract_description = AsyncMock(return_value=desc)

    result = await scraper._load_job_description(page, "/jobs/view/777/")
    assert result == desc  # accepted via the URL match despite no text change


@pytest.mark.asyncio
async def test_load_job_description_matches_exact_id_not_substring(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review defect: id 123 must NOT bind to a longer-id card (/jobs/view/1234). The card is
    re-resolved by EXACT job id, even when the colliding card appears first in the DOM."""
    monkeypatch.setattr("job_applicator.scrapers.linkedin.random_delay", AsyncMock())
    scraper = LinkedInScraper(MagicMock(), app_settings)
    wrong = _desc_link("/jobs/view/1234/")  # substring-collides with 123, appears FIRST
    right = _desc_link("/jobs/view/123/")
    page = MagicMock()
    page.query_selector_all = AsyncMock(return_value=[wrong, right])
    page.url = "https://www.linkedin.com/jobs/search?currentJobId=123"
    desc = "job 123's description text, comfortably longer than one hundred chars. " * 3
    scraper._get_desc_text = AsyncMock(return_value=desc)
    scraper._extract_description = AsyncMock(return_value=desc)

    await scraper._load_job_description(page, "/jobs/view/123/")
    right.click.assert_awaited_once()  # the exact-id card
    wrong.click.assert_not_called()  # NOT the substring-colliding 1234 card


def test_pick_geo_hit_prefers_region_match() -> None:
    """Review defect: with a region hint, prefer the hit whose displayName carries it — so an
    ambiguous same-name city isn't silently resolved to the region-biased first hit."""
    from job_applicator.scrapers.linkedin import _pick_geo_hit

    hits = [
        {"id": "1", "displayName": "Springfield, Missouri, United States"},
        {"id": "2", "displayName": "Springfield, Illinois, United States"},
    ]
    chosen = _pick_geo_hit(hits, "illinois")
    assert chosen is not None and chosen["id"] == "2"  # region hint wins over document order


def test_pick_geo_hit_falls_back_to_first_numeric() -> None:
    """No hint (or no match) → the region-biased first NUMERIC hit; non-numeric ids and non-lists
    are rejected (None), never a fabricated pick."""
    from job_applicator.scrapers.linkedin import _pick_geo_hit

    hits = [
        {"id": "not-numeric", "displayName": "x"},  # skipped
        {"id": "42", "displayName": "Montreal, Quebec, Canada"},
    ]
    first = _pick_geo_hit(hits, "")
    assert first is not None and first["id"] == "42"
    assert _pick_geo_hit([], "qc") is None
    assert _pick_geo_hit("not a list", "qc") is None


def test_canonical_job_url_strips_tracking_dedups_reposts() -> None:
    """The SAME LinkedIn job served under different tracking params canonicalizes to ONE
    /jobs/view/<id> URL, so URL-dedup collapses tracking-param reposts (measured: 53% of a funnel);
    a genuinely-different job id stays distinct."""
    from job_applicator.scrapers.linkedin import _canonical_job_url

    a = _canonical_job_url(
        "https://www.linkedin.com/jobs/view/4430471726/?eBP=CwEAAAA&trackingId=x"
    )
    b = _canonical_job_url("https://www.linkedin.com/jobs/view/4430471726/?eBP=DIFFERENT&refId=y")
    assert a == b == "https://www.linkedin.com/jobs/view/4430471726/"  # same job → same URL
    assert _canonical_job_url("https://www.linkedin.com/jobs/view/9999999999/?eBP=z") != a
    assert (
        _canonical_job_url("https://www.linkedin.com/jobs/search?currentJobId=123&x=1")
        == "https://www.linkedin.com/jobs/view/123/"
    )
    assert (
        _canonical_job_url("https://www.linkedin.com/jobs/collections/?trk=z")
        == "https://www.linkedin.com/jobs/collections/"
    )
