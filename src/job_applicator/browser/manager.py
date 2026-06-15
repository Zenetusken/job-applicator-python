"""Playwright browser lifecycle manager."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from job_applicator.config import BrowserConfig
from job_applicator.exceptions import BrowserError
from job_applicator.utils.logging import get_logger

logger = get_logger("browser.manager")

# A realistic desktop UA. Playwright's default advertises "HeadlessChrome",
# which sites like LinkedIn flag; use this when no UA is configured.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)


class BrowserManager:
    """Manages Playwright browser lifecycle and contexts."""

    def __init__(self, config: BrowserConfig) -> None:
        self._config = config
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._persistent_context: BrowserContext | None = None

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
        if self._persistent_context:
            await self._persistent_context.close()
            self._persistent_context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Browser closed")

    async def _create_context(self, browser: Browser) -> BrowserContext:
        """Build a context with the configured viewport/UA/locale settings."""
        context = await browser.new_context(
            viewport={
                "width": self._config.viewport_width,
                "height": self._config.viewport_height,
            },
            user_agent=self._config.user_agent or DEFAULT_USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
        )
        context.set_default_timeout(self._config.timeout_ms)
        return context

    @asynccontextmanager
    async def new_context(self) -> AsyncIterator[BrowserContext]:
        """Create a new (isolated) browser context with configured settings."""
        if not self._browser:
            raise BrowserError("Browser not started. Call start() first.")

        context = await self._create_context(self._browser)
        try:
            yield context
        finally:
            await context.close()

    async def persistent_context(self) -> BrowserContext:
        """Return a single shared context that lives for the manager's lifetime.

        Unlike :meth:`new_context`, this context is created once and reused, and
        is NOT closed when an individual page finishes. Cookies/auth set by one
        component (e.g. the scraper logging in) are therefore visible to another
        (e.g. the applicator submitting Easy Apply) using the same manager. It
        is closed in :meth:`stop`.
        """
        if not self._browser:
            raise BrowserError("Browser not started. Call start() first.")
        if self._persistent_context is None:
            self._persistent_context = await self._create_context(self._browser)
        return self._persistent_context

    @asynccontextmanager
    async def new_page(self) -> AsyncIterator[Page]:
        """Create a new page in an isolated context."""
        async with self.new_context() as context:
            page = await context.new_page()
            yield page

    @asynccontextmanager
    async def persistent_page(self) -> AsyncIterator[Page]:
        """Open a page in the shared persistent context (auth/cookies preserved).

        Only the page is closed on exit; the shared context stays alive.
        """
        context = await self.persistent_context()
        page = await context.new_page()
        try:
            yield page
        finally:
            await page.close()

    async def __aenter__(self) -> BrowserManager:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()
