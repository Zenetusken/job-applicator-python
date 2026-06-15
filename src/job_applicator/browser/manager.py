"""Playwright browser lifecycle manager."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright
from playwright_stealth import Stealth

from job_applicator.config import BrowserConfig
from job_applicator.exceptions import BrowserError
from job_applicator.utils.logging import get_logger
from job_applicator.utils.region import detect_chrome_user_agent, detect_locale, detect_timezone

logger = get_logger("browser.manager")

# Persistent Chrome profile directory — preserves ALL browser state (cookies,
# localStorage, IndexedDB, service workers, history) between runs. LinkedIn
# fingerprints this state; a fresh context every time looks like a bot.
PROFILE_DIR = Path.home() / ".job-applicator" / "browser-profile"


class BrowserManager:
    """Manages Playwright browser lifecycle and contexts."""

    def __init__(self, config: BrowserConfig) -> None:
        self._config = config
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._persistent_context: BrowserContext | None = None
        self._stealth = Stealth()

    async def start(self) -> None:
        """Launch the Playwright browser."""
        try:
            self._playwright = await async_playwright().start()
            # Use a persistent Chrome profile so browser state (cookies,
            # localStorage, history, service workers) accumulates over time.
            # This is indistinguishable from a real user's browser.
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            PROFILE_DIR.chmod(0o700)  # profile holds the live authenticated session
            # Advertise the host's real locale/timezone (auto-detected unless
            # configured) so geo-aware sites serve the correct region.
            resolved_locale = self._config.locale or detect_locale()
            resolved_tz = self._config.timezone or detect_timezone()
            resolved_ua = self._config.user_agent or detect_chrome_user_agent()
            self._persistent_context = await self._playwright.chromium.launch_persistent_context(
                str(PROFILE_DIR),
                headless=self._config.headless,
                slow_mo=self._config.slow_mo,
                args=[
                    "--disable-blink-features=AutomationControlled",
                ],
                viewport={
                    "width": self._config.viewport_width,
                    "height": self._config.viewport_height,
                },
                user_agent=resolved_ua,
                locale=resolved_locale,
                timezone_id=resolved_tz,
            )
            self._persistent_context.set_default_timeout(self._config.timeout_ms)
            # Apply stealth to the persistent context
            await self._stealth.apply_stealth_async(self._persistent_context)
            logger.info(
                "Browser launched (headless=%s, locale=%s, tz=%s)",
                self._config.headless,
                resolved_locale,
                resolved_tz,
            )
        except Exception as exc:
            # Clean up a partially-initialised launch (e.g. the stealth/timeout
            # step raised after the context was created) so we don't leak a
            # Chrome process or hold the profile's SingletonLock. start() runs
            # inside __aenter__, so stop() would otherwise never run.
            with suppress(Exception):
                await self.stop()
            raise BrowserError(
                f"Failed to launch browser: {exc}. Another job-applicator instance "
                f"may be using the browser profile, or a previous run left a lock "
                f"(remove {PROFILE_DIR / 'SingletonLock'} and retry if so).",
                context={"headless": self._config.headless, "profile": str(PROFILE_DIR)},
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

    async def persistent_context(self) -> BrowserContext:
        """Return the persistent browser context.

        With launch_persistent_context, this IS the browser — there's no
        separate Browser object. All state (cookies, localStorage, etc.)
        persists on disk between runs.
        """
        if self._persistent_context is None:
            raise BrowserError("Browser not started. Call start() first.")
        return self._persistent_context

    @asynccontextmanager
    async def persistent_page(self) -> AsyncIterator[Page]:
        """Open a page in the persistent context (auth/cookies preserved).

        Only the page is closed on exit; the persistent context stays alive.
        """
        context = await self.persistent_context()
        # Stealth is applied once to the context in start(); the context
        # auto-applies it to every page it creates, so no per-page call here.
        page = await context.new_page()
        try:
            yield page
        finally:
            await page.close()

    @asynccontextmanager
    async def new_page(self) -> AsyncIterator[Page]:
        """Open a page in the persistent context (alias for persistent_page)."""
        async with self.persistent_page() as page:
            yield page

    async def __aenter__(self) -> BrowserManager:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()
