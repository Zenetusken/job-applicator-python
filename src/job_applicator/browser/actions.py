"""Low-level browser actions — click, fill, wait, screenshot."""

from __future__ import annotations

from pathlib import Path
from secrets import SystemRandom

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeout

from job_applicator.exceptions import ElementNotFoundError, NavigationError
from job_applicator.utils.logging import get_logger

logger = get_logger("browser.actions")
_random = SystemRandom()


async def navigate(page: Page, url: str, timeout_ms: int = 30_000) -> None:
    """Navigate to a URL with error handling."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        logger.info("Navigated to %s", url)
    except PlaywrightTimeout as exc:
        raise NavigationError(
            f"Navigation timed out: {url}",
            context={"url": url, "timeout_ms": timeout_ms},
        ) from exc
    except PlaywrightError as exc:
        # Non-timeout navigation failures (DNS, connection refused, ERR_ABORTED, …) are a base
        # playwright Error, NOT a TimeoutError — wrap them too so they're retryable via the
        # NavigationError-only retry and never surface as a raw traceback (CLAUDE.md: typed errors).
        raise NavigationError(
            f"Navigation failed: {url} ({exc})",
            context={"url": url},
        ) from exc


async def click(page: Page, selector: str, timeout_ms: int = 10_000) -> None:
    """Click an element with retry."""
    try:
        await page.click(selector, timeout=timeout_ms)
        logger.debug("Clicked %s", selector)
    except PlaywrightTimeout as exc:
        raise ElementNotFoundError(
            f"Element not found for click: {selector}",
            context={"selector": selector},
        ) from exc


async def fill(page: Page, selector: str, value: str, timeout_ms: int = 10_000) -> None:
    """Fill a form field with human-like typing."""
    try:
        await page.fill(selector, value, timeout=timeout_ms)
        logger.debug("Filled %s", selector)
    except PlaywrightTimeout as exc:
        raise ElementNotFoundError(
            f"Element not found for fill: {selector}",
            context={"selector": selector},
        ) from exc


async def type_human(page: Page, selector: str, value: str, timeout_ms: int = 10_000) -> None:
    """Type into a field with realistic delays between keystrokes."""
    try:
        await page.click(selector, timeout=timeout_ms)
        for char in value:
            await page.keyboard.type(char, delay=_random.randint(30, 100))
        logger.debug("Typed into %s", selector)
    except PlaywrightTimeout as exc:
        raise ElementNotFoundError(
            f"Element not found for typing: {selector}",
            context={"selector": selector},
        ) from exc


async def select_option(page: Page, selector: str, value: str, timeout_ms: int = 10_000) -> None:
    """Select a dropdown option."""
    try:
        await page.select_option(selector, value, timeout=timeout_ms)
        logger.debug("Selected %s in %s", value, selector)
    except PlaywrightTimeout as exc:
        raise ElementNotFoundError(
            f"Element not found for select: {selector}",
            context={"selector": selector},
        ) from exc


async def wait_for_selector(page: Page, selector: str, timeout_ms: int = 10_000) -> bool:
    """Wait for an element to appear. Returns True if found."""
    try:
        await page.wait_for_selector(selector, timeout=timeout_ms)
        return True
    except PlaywrightTimeout:
        return False


async def screenshot(page: Page, path: Path) -> None:
    """Take a screenshot for debugging."""
    await page.screenshot(path=str(path), full_page=True)
    logger.info("Screenshot saved to %s", path)


async def random_delay(min_s: float = 0.5, max_s: float = 2.0) -> None:
    """Wait a random duration to simulate human behavior."""
    import asyncio

    delay = _random.uniform(min_s, max_s)
    await asyncio.sleep(delay)
