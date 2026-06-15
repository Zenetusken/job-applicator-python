"""Tests for shared persistent browser context (H-4 / L-2).

These pin the wiring that makes the LinkedIn applicator reuse the
authenticated session the scraper established, without touching a real
browser. A small fake Playwright surface stands in for chromium.
"""

from __future__ import annotations

import pytest

from job_applicator.applicators.linkedin import LinkedInApplicator
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings, BrowserConfig
from job_applicator.scrapers.linkedin import LinkedInScraper


class FakePage:
    def __init__(self, context: FakeContext) -> None:
        self.context = context
        self.closed = False

    async def query_selector(self, _selector: str) -> None:
        return None

    async def add_init_script(self, _script: object) -> None:
        """No-op for stealth patches in tests."""

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


def _manager_with_fake_context() -> tuple[BrowserManager, FakeContext]:
    """Create a manager with a pre-injected fake persistent context.

    With launch_persistent_context, the manager stores the context directly
    on _persistent_context (no separate Browser object). Tests bypass start()
    and inject a fake context.
    """
    manager = BrowserManager(BrowserConfig(headless=True, timeout_ms=5000))
    fake_ctx = FakeContext()
    manager._persistent_context = fake_ctx  # type: ignore[assignment]
    return manager, fake_ctx


@pytest.mark.asyncio
async def test_persistent_context_is_reused() -> None:
    manager, fake_ctx = _manager_with_fake_context()

    first = await manager.persistent_context()
    second = await manager.persistent_context()

    assert first is second
    assert first is fake_ctx


@pytest.mark.asyncio
async def test_persistent_page_keeps_context_open() -> None:
    manager, fake_ctx = _manager_with_fake_context()

    async with manager.persistent_page() as page:
        pass

    assert page.closed is True
    assert fake_ctx.closed is False  # context survives the page


@pytest.mark.asyncio
async def test_stop_closes_persistent_context() -> None:
    manager, fake_ctx = _manager_with_fake_context()
    _ = await manager.persistent_context()

    await manager.stop()

    assert fake_ctx.closed is True
    assert manager._persistent_context is None


@pytest.mark.asyncio
async def test_scraper_and_applicator_share_one_context(
    app_settings: AppSettings, sample_job: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The applicator must reuse the scraper's authenticated context (H-4)."""
    manager, fake_ctx = _manager_with_fake_context()
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

    # Both scraper and applicator use the same persistent context.
    assert scraper_ctx is fake_ctx
    assert fake_ctx.pages, "applicator should open a page in the shared context"


@pytest.mark.asyncio
async def test_apply_screenshots_the_failed_page(
    app_settings: AppSettings, sample_job: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On failure the screenshot is taken on the same page, not a fresh one."""
    manager, fake_ctx = _manager_with_fake_context()
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
    assert len(fake_ctx.pages) == 1
    assert screenshotted["page"] is fake_ctx.pages[0]


@pytest.mark.asyncio
async def test_apply_returns_failed_when_context_entry_raises(
    app_settings: AppSettings, sample_job: object
) -> None:
    """If persistent_page() entry raises (e.g. browser not started), apply() must
    return ApplicationResult(FAILED) — not propagate and crash the apply loop."""
    # Browser never started -> persistent_context() raises BrowserError on entry.
    manager = BrowserManager(BrowserConfig(headless=True, timeout_ms=5000))
    applicator = LinkedInApplicator(manager, app_settings)

    result = await applicator.apply(sample_job)  # type: ignore[arg-type]

    assert result.status.value == "failed"
    assert "Browser not started" in (result.error_message or "")
