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
