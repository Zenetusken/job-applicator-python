"""Unit tests for scrapers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import Error as PlaywrightError

from job_applicator.config import AppSettings
from job_applicator.exceptions import LoginRequiredError
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
    scraper._new_stealth_page = AsyncMock(return_value=page)

    assert await scraper._ensure_session(MagicMock()) is False
    assert scraper._logged_in is False


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
async def test_scrape_emits_per_card_progress(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scrape() ticks on_progress once per CARD (1-based, /total) at the TOP of the loop,
    so a card whose extraction returns None or raises still advances the count to N/N —
    the count tracks cards processed, not jobs extracted (else it stalls on a bad card)."""
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

    # card 2 → None (no job), card 3 → raises; both must still tick the counter.
    scraper._extract_job = AsyncMock(side_effect=[_stub(1), None, ValueError("bad card"), _stub(4)])

    msgs: list[str] = []
    jobs = await scraper.scrape(SearchParams(query="python", board=JobBoard.LINKEDIN), msgs.append)

    assert msgs == [
        "Scraping job 1/4 on LinkedIn…",
        "Scraping job 2/4 on LinkedIn…",
        "Scraping job 3/4 on LinkedIn…",
        "Scraping job 4/4 on LinkedIn…",
    ]
    assert [j.title for j in jobs] == ["E1", "E4"]  # only 2 extracted; count still reached 4/4


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


async def test_scrape_listings_all_cards_fail_raises_scraper_error(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LinkedIn: cards present but EVERY extraction fails → ScraperError (stale field selectors),
    never a silent empty list — the honest-failure twin of the Indeed guard."""
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

    async def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("stale field selector")

    monkeypatch.setattr(scraper, "_new_stealth_page", _page)
    monkeypatch.setattr(lk, "navigate", _noop)
    monkeypatch.setattr(lk, "random_delay", _noop)
    monkeypatch.setattr(lk, "wait_for_selector", _wait)
    monkeypatch.setattr(scraper, "_extract_job", _boom)

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
