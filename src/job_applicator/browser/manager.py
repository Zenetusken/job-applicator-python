"""Playwright browser lifecycle manager."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from job_applicator.config import BrowserConfig
from job_applicator.exceptions import BrowserError
from job_applicator.utils.logging import get_logger

logger = get_logger("browser.manager")


class BrowserManager:
    """Manages Playwright browser lifecycle and contexts."""

    def __init__(self, config: BrowserConfig) -> None:
        self._config = config
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def start(self) -> None:
        """Launch the Playwright browser."""
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self._config.headless,
                slow_mo=self._config.slow_mo,
            )
            logger.info(
                "Browser launched (headless=%s)",
                self._config.headless,
            )
        except Exception as exc:
            raise BrowserError(
                f"Failed to launch browser: {exc}",
                context={"headless": self._config.headless},
            ) from exc

    async def stop(self) -> None:
        """Close the Playwright browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Browser closed")

    @asynccontextmanager
    async def new_context(self) -> AsyncIterator[BrowserContext]:
        """Create a new browser context with configured settings."""
        if not self._browser:
            raise BrowserError("Browser not started. Call start() first.")

        context = await self._browser.new_context(
            viewport={
                "width": self._config.viewport_width,
                "height": self._config.viewport_height,
            },
            user_agent=self._config.user_agent,
            locale="en-US",
            timezone_id="America/New_York",
        )
        context.set_default_timeout(self._config.timeout_ms)
        try:
            yield context
        finally:
            await context.close()

    @asynccontextmanager
    async def new_page(self) -> AsyncIterator[Page]:
        """Create a new page in an isolated context."""
        async with self.new_context() as context:
            page = await context.new_page()
            yield page

    async def __aenter__(self) -> BrowserManager:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()
