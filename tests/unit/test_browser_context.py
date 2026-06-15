"""Tests for shared persistent browser context (H-4 / L-2).

These pin the wiring that makes the LinkedIn applicator reuse the
authenticated session the scraper established, without touching a real
browser. A small fake Playwright surface stands in for chromium.
"""

from __future__ import annotations

import pytest

from job_applicator.applicators.linkedin import LinkedInApplicator
from job_applicator.browser.manager import DEFAULT_USER_AGENT, BrowserManager
from job_applicator.config import AppSettings, BrowserConfig
from job_applicator.scrapers.linkedin import LinkedInScraper


class FakePage:
    def __init__(self, context: FakeContext) -> None:
        self.context = context
        self.closed = False

    async def query_selector(self, _selector: str) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class FakeContext:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.timeout_ms: int | None = None
        self.closed = False
        self.pages: list[FakePage] = []

    def set_default_timeout(self, ms: int) -> None:
        self.timeout_ms = ms

    async def new_page(self) -> FakePage:
        page = FakePage(self)
        self.pages.append(page)
        return page

    async def close(self) -> None:
        self.closed = True


class FakeBrowser:
    def __init__(self) -> None:
        self.contexts: list[FakeContext] = []
        self.closed = False

    async def new_context(self, **kwargs: object) -> FakeContext:
        ctx = FakeContext(**kwargs)
        self.contexts.append(ctx)
        return ctx

    async def close(self) -> None:
        self.closed = True


def _manager_with_fake_browser() -> tuple[BrowserManager, FakeBrowser]:
    manager = BrowserManager(BrowserConfig(headless=True, timeout_ms=5000))
    fake = FakeBrowser()
    manager._browser = fake  # type: ignore[assignment]
    return manager, fake


@pytest.mark.asyncio
async def test_persistent_context_is_created_once_and_reused() -> None:
    manager, fake = _manager_with_fake_browser()

    first = await manager.persistent_context()
    second = await manager.persistent_context()

    assert first is second
    assert len(fake.contexts) == 1
    assert first.timeout_ms == 5000


@pytest.mark.asyncio
async def test_persistent_context_uses_default_user_agent() -> None:
    manager, _ = _manager_with_fake_browser()

    ctx = await manager.persistent_context()

    # BrowserConfig.user_agent defaults to None; manager must fall back to a
    # realistic UA so sites don't see "HeadlessChrome".
    assert ctx.kwargs["user_agent"] == DEFAULT_USER_AGENT


@pytest.mark.asyncio
async def test_persistent_page_keeps_context_open() -> None:
    manager, _ = _manager_with_fake_browser()

    async with manager.persistent_page() as page:
        pass

    assert page.closed is True
    assert page.context.closed is False  # context survives the page


@pytest.mark.asyncio
async def test_new_context_is_isolated_and_closed() -> None:
    manager, _ = _manager_with_fake_browser()

    async with manager.new_context() as ctx:
        pass

    assert ctx.closed is True
    # An isolated context is distinct from the shared persistent one.
    persistent = await manager.persistent_context()
    assert persistent is not ctx


@pytest.mark.asyncio
async def test_stop_closes_persistent_context() -> None:
    manager, fake = _manager_with_fake_browser()
    ctx = await manager.persistent_context()

    await manager.stop()

    assert ctx.closed is True
    assert manager._persistent_context is None
    assert fake.closed is True


@pytest.mark.asyncio
async def test_scraper_and_applicator_share_one_context(
    app_settings: AppSettings, sample_job: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The applicator must reuse the scraper's authenticated context (H-4)."""
    manager, fake = _manager_with_fake_browser()
    scraper = LinkedInScraper(manager, app_settings)
    applicator = LinkedInApplicator(manager, app_settings)

    async def _noop(*args: object, **kwargs: object) -> None:
        return None

    # No Easy Apply button -> external-apply path; no real navigation.
    monkeypatch.setattr("job_applicator.applicators.linkedin.navigate", _noop)
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", _noop)
    monkeypatch.setattr("job_applicator.applicators.linkedin.wait_for_selector", _noop)

    # Scraper acquires the shared context (as it would during login()).
    scraper_ctx = await scraper._get_context()
    await applicator.apply(sample_job)  # type: ignore[arg-type]

    # Only ONE context ever created, and the applicator's page lives in it.
    assert len(fake.contexts) == 1
    assert scraper_ctx is fake.contexts[0]
    assert fake.contexts[0].pages, "applicator should open a page in the shared context"


@pytest.mark.asyncio
async def test_apply_screenshots_the_failed_page(
    app_settings: AppSettings, sample_job: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On failure the screenshot is taken on the same page, not a fresh one."""
    manager, fake = _manager_with_fake_browser()
    assert app_settings.screenshot_on_error is True  # default; drives this path
    applicator = LinkedInApplicator(manager, app_settings)

    async def _noop(*args: object, **kwargs: object) -> None:
        return None

    async def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("navigation blew up")

    screenshotted: dict[str, object] = {}

    async def _capture(page: object, path: object) -> None:
        screenshotted["page"] = page

    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", _noop)
    monkeypatch.setattr("job_applicator.applicators.linkedin.navigate", _boom)
    monkeypatch.setattr("job_applicator.applicators.linkedin.screenshot", _capture)

    result = await applicator.apply(sample_job)  # type: ignore[arg-type]

    assert result.status.value == "failed"
    # Exactly one page was opened, and that's the page we screenshotted.
    assert len(fake.contexts) == 1
    assert len(fake.contexts[0].pages) == 1
    assert screenshotted["page"] is fake.contexts[0].pages[0]
