"""Playwright browser lifecycle manager."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright
from playwright_stealth import Stealth

from job_applicator.config import BrowserConfig
from job_applicator.exceptions import BrowserError
from job_applicator.utils.logging import get_logger
from job_applicator.utils.region import (
    chrome_user_agent_for_binary,
    detect_chrome_user_agent,
    detect_locale,
    detect_timezone,
    host_chrome_path,
    navigator_languages,
    navigator_platform_for_ua,
)

if TYPE_CHECKING:
    from pyvirtualdisplay import Display

logger = get_logger("browser.manager")

# Persistent Chrome profile directory — preserves ALL browser state (cookies,
# localStorage, IndexedDB, service workers, history) between runs. LinkedIn
# fingerprints this state; a fresh context every time looks like a bot.
PROFILE_DIR = Path.home() / ".job-applicator" / "browser-profile"


def _aligned_stealth(user_agent: str, locale: str) -> Stealth:
    """Build a Stealth whose navigator overrides match the advertised UA + locale.

    playwright_stealth defaults to navigator.platform='Win32' and navigator.languages=('en-US',
    'en'), which contradict a Linux UA or a non-US locale — a detectable fingerprint
    inconsistency. Aligning them keeps platform/languages consistent with what the context
    already advertises (the resolved user_agent / locale).
    """
    return Stealth(
        navigator_platform_override=navigator_platform_for_ua(user_agent),
        navigator_languages_override=navigator_languages(locale),
        # Disable the WebGL vendor/renderer spoof: its default is a macOS "Intel Iris" string, an
        # obvious cross-OS tell under a Linux UA. With channel="chrome" the REAL Chrome GPU renderer
        # is already platform-consistent (and honest), so let it through rather than lie about it.
        webgl_vendor=False,
    )


class BrowserManager:
    """Manages Playwright browser lifecycle and contexts."""

    def __init__(
        self,
        config: BrowserConfig,
        *,
        profile_dir: Path | None = None,
        ephemeral_profile: bool = False,
        virtual_display: bool = False,
    ) -> None:
        """Manage a Playwright browser.

        ``profile_dir`` overrides the shared persistent profile (use a dedicated
        one per board to avoid cross-contamination). ``ephemeral_profile`` uses a
        fresh throwaway profile per run — the empirically reliable choice for the
        Cloudflare-fronted Indeed path, which passes from a clean profile.
        ``virtual_display`` runs a headed browser windowless on an X virtual
        display (Xvfb) — Indeed's managed challenge fails headless, so it must run
        headed, and a virtual display keeps that off-screen and server-capable.
        """
        if ephemeral_profile and profile_dir is not None:
            raise BrowserError("ephemeral_profile and profile_dir are mutually exclusive")
        self._config = config
        self._profile_dir = profile_dir
        self._ephemeral_profile = ephemeral_profile
        self._virtual_display = virtual_display
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._persistent_context: BrowserContext | None = None
        self._display: Display | None = None
        self._tmp_profile: tempfile.TemporaryDirectory[str] | None = None
        self._active_profile_dir: Path | None = None  # the dir actually launched with

    @property
    def headless(self) -> bool:
        """Whether this manager launches Chrome headless (board policy checks this)."""
        return self._config.headless

    def _enter_virtual_display(self) -> Display | None:
        """Start an Xvfb virtual display for a headed browser; graceful fallback.

        Prefers pyvirtualdisplay (the optional ``[indeed]`` extra). If it's
        unavailable or Xvfb won't start, fall back to the ambient ``$DISPLAY``
        (a window may appear); if there's no display at all, raise with guidance.
        """
        try:
            from pyvirtualdisplay import Display
        except ImportError:
            if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                logger.info(
                    "pyvirtualdisplay not installed; using ambient display (window may show)."
                )
                return None
            raise BrowserError(
                "Headed mode needs a display. Install the optional extra "
                '(pip install "job-applicator[indeed]") to auto-manage a virtual '
                "display, or run the command under `xvfb-run`."
            ) from None
        try:
            disp = Display(
                visible=False,
                size=(self._config.viewport_width, self._config.viewport_height),
            )
            disp.start()
        except Exception as exc:
            if os.environ.get("DISPLAY"):
                logger.warning("Virtual display failed (%s); using ambient display.", exc)
                return None
            raise BrowserError(
                f"Could not start a virtual display (Xvfb): {exc}. Install Xvfb "
                "(e.g. `apt install xvfb`) or run under `xvfb-run`."
            ) from exc
        logger.info("Started virtual display for headed browser")
        return disp

    def _resolve_profile_dir(self) -> Path:
        """Return the user-data dir to launch with (ephemeral temp dir if requested)."""
        if self._ephemeral_profile:
            self._tmp_profile = tempfile.TemporaryDirectory(prefix="ja-profile-")
            self._active_profile_dir = Path(self._tmp_profile.name)  # mkdtemp is 0700
            return self._active_profile_dir
        profile = self._profile_dir or PROFILE_DIR
        profile.mkdir(parents=True, exist_ok=True)
        profile.chmod(0o700)  # profile holds the live authenticated session
        self._active_profile_dir = profile
        return profile

    async def start(self) -> None:
        """Launch the Playwright browser."""
        try:
            self._playwright = await async_playwright().start()
            # A headed browser (required to clear Cloudflare's managed challenge on
            # Indeed) needs a display; run it windowless on a virtual one.
            if self._virtual_display:
                self._display = self._enter_virtual_display()
            # Use a persistent Chrome profile so browser state (cookies,
            # localStorage, history, service workers) accumulates over time.
            # This is indistinguishable from a real user's browser.
            profile_dir = self._resolve_profile_dir()
            # Advertise the host's real locale/timezone (auto-detected unless
            # configured) so geo-aware sites serve the correct region.
            resolved_locale = self._config.locale or detect_locale()
            resolved_tz = self._config.timezone or detect_timezone()
            # Prefer the host's REAL Chrome (channel="chrome") — no HeadlessChrome client-hint leak,
            # real-GPU WebGL, UA == Sec-CH-UA. Fall back to bundled Chromium (with a warning) if no
            # host Chrome is installed. The UA reflects whichever engine actually launches, so the
            # advertised version matches the engine's own client-hints (no detectable version skew).
            resolved_channel = self._config.channel or None
            if resolved_channel == "chrome" and host_chrome_path() is None:
                logger.warning(
                    "channel='chrome' requested but no host Chrome found; using bundled Chromium"
                )
                resolved_channel = None
            if self._config.user_agent:
                resolved_ua = self._config.user_agent
            elif resolved_channel == "chrome":
                resolved_ua = detect_chrome_user_agent()  # the host Chrome we actually launch
            else:
                bundled = self._playwright.chromium.executable_path
                resolved_ua = chrome_user_agent_for_binary(bundled)
            self._persistent_context = await self._playwright.chromium.launch_persistent_context(
                str(profile_dir),
                channel=resolved_channel,
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
            # Apply stealth, with navigator overrides aligned to the advertised UA + locale — the
            # library defaults (Win32 platform + en-US/en languages) otherwise contradict a Linux
            # UA or a non-US locale, a detectable fingerprint inconsistency.
            stealth = _aligned_stealth(resolved_ua, resolved_locale)
            await stealth.apply_stealth_async(self._persistent_context)
            logger.info(
                "Browser launched (engine=%s, headless=%s, locale=%s, tz=%s)",
                resolved_channel or "chromium (bundled)",
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
            # A BrowserError from a helper (e.g. _enter_virtual_display's
            # missing-display guidance) is already actionable — don't bury it in a
            # generic "another instance may be using the profile" message.
            if isinstance(exc, BrowserError):
                raise
            # Reference the profile actually launched with (Indeed uses a temp dir),
            # and only mention SingletonLock for a real (persistent) profile — a
            # fresh ephemeral dir can never have a stale lock.
            active = self._active_profile_dir or self._profile_dir or PROFILE_DIR
            lock_hint = (
                ""
                if self._ephemeral_profile
                else (
                    " Another job-applicator instance may be using the profile, or a "
                    f"previous run left a lock (remove {active / 'SingletonLock'} and retry)."
                )
            )
            raise BrowserError(
                f"Failed to launch browser: {exc}.{lock_hint}",
                context={"headless": self._config.headless, "profile": str(active)},
            ) from exc

    async def stop(self) -> None:
        """Close the Playwright browser.

        Best-effort: each teardown step is isolated so one failure (e.g. a context
        that won't close) can't skip the rest — otherwise a hung close() would leak
        the Xvfb virtual display and the ephemeral temp profile.
        """
        if self._persistent_context:
            with suppress(Exception):
                await self._persistent_context.close()
            self._persistent_context = None
        if self._browser:
            with suppress(Exception):
                await self._browser.close()
            self._browser = None
        if self._playwright:
            with suppress(Exception):
                await self._playwright.stop()
            self._playwright = None
        if self._display is not None:
            with suppress(Exception):
                self._display.stop()
            self._display = None
        if self._tmp_profile is not None:
            with suppress(Exception):
                self._tmp_profile.cleanup()
            self._tmp_profile = None
        self._active_profile_dir = None
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
