"""Integration smoke tests: board → BrowserPolicy → BrowserManager wiring.

Construction-only (no real browser launch): verifies the CLAUDE.md invariant that a
board declares its browser needs via ``browser_policy()`` and the CLI factory builds a
``BrowserManager`` honoring them — so anti-bot requirements can't drift from the CLI.
"""

from __future__ import annotations

from job_applicator.cli import _make_browser, _make_scraper
from job_applicator.config import AppSettings, BrowserConfig
from job_applicator.scrapers.base import BaseScraper


def _settings(headless: bool = True) -> AppSettings:
    return AppSettings(browser=BrowserConfig(headless=headless))


def test_indeed_declares_headed_ephemeral_virtual() -> None:
    """Indeed's Cloudflare managed challenge needs headed + fresh profile + Xvfb."""
    from job_applicator.scrapers.indeed import IndeedScraper

    policy = IndeedScraper.browser_policy()
    assert (policy.headed, policy.ephemeral_profile, policy.virtual_display) == (True, True, True)


def test_linkedin_uses_default_policy() -> None:
    """LinkedIn keeps the default headless shared profile."""
    from job_applicator.scrapers.linkedin import LinkedInScraper

    policy = LinkedInScraper.browser_policy()
    assert (policy.headed, policy.ephemeral_profile, policy.virtual_display) == (
        False,
        False,
        False,
    )


def test_make_browser_forces_headed_for_indeed() -> None:
    """The CLI factory honors Indeed's policy: a headed browser even though config
    defaults to headless."""
    assert _make_browser("indeed", _settings(headless=True)).headless is False


def test_make_browser_keeps_default_for_linkedin() -> None:
    """LinkedIn gets the configured (default headless) browser."""
    assert _make_browser("linkedin", _settings(headless=True)).headless is True


def test_make_scraper_builds_the_site_class() -> None:
    """The scraper factory wires settings + browser into the right board class."""
    settings = _settings()
    browser = _make_browser("linkedin", settings)
    scraper = _make_scraper("linkedin", browser, settings)
    assert isinstance(scraper, BaseScraper)
    assert type(scraper).__name__ == "LinkedInScraper"
