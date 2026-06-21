"""Tests for the per-board browser factory (_make_browser).

Indeed needs headed + ephemeral profile + virtual display (Cloudflare managed
challenge); LinkedIn stays headless on the shared persistent profile.
"""

from __future__ import annotations

import pytest
import typer

from job_applicator.config import AppSettings
from job_applicator.factories import _make_browser, _scraper_class
from job_applicator.scrapers.linkedin import LinkedInScraper


def test_make_browser_indeed_headed_ephemeral_windowless() -> None:
    settings = AppSettings()
    settings.browser.headless = True  # default (no --headed)
    browser = _make_browser("indeed", settings)
    assert browser._config.headless is False  # forced headed (challenge needs it)
    assert browser._ephemeral_profile is True  # fresh profile each run
    assert browser._virtual_display is True  # windowless via Xvfb by default


def test_make_browser_indeed_headed_flag_shows_window() -> None:
    settings = AppSettings()
    settings.browser.headless = False  # user passed --headed → wants to watch
    browser = _make_browser("indeed", settings)
    assert browser._config.headless is False
    assert browser._virtual_display is False  # real window, not a virtual display


def test_make_browser_linkedin_unchanged() -> None:
    settings = AppSettings()
    settings.browser.headless = True
    browser = _make_browser("linkedin", settings)
    assert browser._config.headless is True  # stays headless
    assert browser._ephemeral_profile is False  # shared persistent profile
    assert browser._virtual_display is False


def test_linkedin_browser_policy_is_default() -> None:
    policy = LinkedInScraper.browser_policy()
    assert policy.headed is False
    assert policy.ephemeral_profile is False
    assert policy.virtual_display is False


def test_make_browser_rejects_unknown_site_without_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown board is rejected BEFORE any BrowserManager is constructed."""
    import job_applicator.browser.manager as mgr

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("BrowserManager must not be constructed for an unknown site")

    monkeypatch.setattr(mgr, "BrowserManager", _boom)
    with pytest.raises(typer.Exit):
        _make_browser("glassdoor", AppSettings())


def test_scraper_class_resolves_known_boards() -> None:
    from job_applicator.scrapers.indeed import IndeedScraper

    assert _scraper_class("linkedin") is LinkedInScraper
    assert _scraper_class("indeed") is IndeedScraper
