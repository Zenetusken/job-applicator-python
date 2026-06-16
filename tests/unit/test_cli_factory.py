"""Tests for the per-board browser factory (_make_browser).

Indeed needs headed + ephemeral profile + virtual display (Cloudflare managed
challenge); LinkedIn stays headless on the shared persistent profile.
"""

from __future__ import annotations

from job_applicator.cli import _make_browser
from job_applicator.config import AppSettings


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
