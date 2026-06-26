"""Unit tests for low-level browser actions (browser/actions.py) — no real browser."""

from __future__ import annotations

import pytest
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeout

from job_applicator.browser.actions import navigate
from job_applicator.exceptions import NavigationError


class _FakePage:
    """A Page whose goto() raises the given exception (the only method navigate() touches)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def goto(self, *_args: object, **_kwargs: object) -> None:
        raise self._exc


@pytest.mark.asyncio
async def test_navigate_wraps_timeout_as_navigation_error() -> None:
    page = _FakePage(PlaywrightTimeout("Timeout 30000ms exceeded"))
    with pytest.raises(NavigationError):
        await navigate(page, "https://example.com")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_navigate_wraps_non_timeout_playwright_error() -> None:
    """A non-timeout Playwright error (DNS failure, connection refused, ERR_ABORTED) must ALSO
    become a typed NavigationError — so it's caught by the NavigationError-only retry and never
    reaches the user as a raw traceback."""
    page = _FakePage(PlaywrightError("net::ERR_NAME_NOT_RESOLVED"))
    with pytest.raises(NavigationError):
        await navigate(page, "https://nope.invalid")  # type: ignore[arg-type]
