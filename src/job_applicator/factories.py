"""Board/browser/runtime factories — extracted from cli.py to slim orchestration.

A board declares its browser needs via ``BrowserPolicy`` on its scraper class; these
factories read that policy so anti-bot requirements can't drift from the CLI, and they
validate the site before constructing anything (so an unknown board never launches a
browser).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from job_applicator.utils.console import console
from job_applicator.utils.llm import LLMRuntime

if TYPE_CHECKING:
    from job_applicator.applicators.base import BaseApplicator
    from job_applicator.browser.manager import BrowserManager
    from job_applicator.config import AppSettings
    from job_applicator.scrapers.base import BaseScraper


def _scraper_class(site: str) -> type[BaseScraper]:
    """Resolve a board's scraper class, or exit if unsupported.

    Site validation lives here so it happens BEFORE any browser is launched.
    """
    if site == "linkedin":
        from job_applicator.scrapers.linkedin import LinkedInScraper

        return LinkedInScraper
    if site == "indeed":
        from job_applicator.scrapers.indeed import IndeedScraper

        return IndeedScraper
    console.print(f"[yellow]{site} scraper not yet implemented[/yellow]")
    raise typer.Exit(1)


def _make_browser(site: str, settings: AppSettings) -> BrowserManager:
    """Build a browser per the board's declared ``BrowserPolicy``.

    The policy lives on the scraper class (not here), so a board's anti-bot needs
    can't drift from the CLI and every caller building a browser gets them right.
    Indeed declares headed + ephemeral-profile + virtual-display (its Cloudflare
    managed challenge fails headless); LinkedIn keeps the default headless shared
    profile. ``--headed`` (config headless=False) shows a real window instead of a
    virtual one. Validates the site before constructing anything (so an unknown
    board never launches a browser).
    """
    from job_applicator.browser.manager import BrowserManager

    policy = _scraper_class(site).browser_policy()
    cfg = settings.browser
    if policy.headed:
        cfg = cfg.model_copy(update={"headless": False})
    # Use a virtual display only when forcing headed AND the user didn't ask to
    # watch (--headed leaves config headless=False → show a real window).
    use_virtual = policy.virtual_display and settings.browser.headless
    return BrowserManager(
        cfg, ephemeral_profile=policy.ephemeral_profile, virtual_display=use_virtual
    )


def _make_scraper(site: str, browser: BrowserManager, settings: AppSettings) -> BaseScraper:
    """Construct the scraper for a job board, or exit if unsupported."""
    return _scraper_class(site)(browser, settings)


def _make_applicator(site: str, browser: BrowserManager, settings: AppSettings) -> BaseApplicator:
    """Construct the applicator for a job board, or exit if unsupported."""
    if site == "linkedin":
        from job_applicator.applicators.linkedin import LinkedInApplicator

        return LinkedInApplicator(browser, settings)
    if site == "indeed":
        from job_applicator.applicators.indeed import IndeedApplicator

        return IndeedApplicator(browser, settings)
    console.print(f"[yellow]{site} applicator not yet implemented[/yellow]")
    raise typer.Exit(1)


def _make_runtime(settings: AppSettings, name: str = "llm") -> LLMRuntime:
    """Build the per-command LLM resilience runtime (shared circuit breaker +
    validation-retry policy) from settings — constructed once per command and shared
    across its LLM consumers (cover-letter + résumé tailoring), so the breaker spans
    the whole run. Named "llm" (neutral) since one breaker now guards both."""
    return LLMRuntime.from_config(settings.llm_resilience, name=name)
