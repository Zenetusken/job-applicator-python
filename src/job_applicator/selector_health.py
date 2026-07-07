"""Live selector health probes for job-board DOM drift diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from playwright.async_api import ElementHandle, Page

from job_applicator.browser.actions import navigate, random_delay, screenshot, wait_for_selector
from job_applicator.config import AppSettings
from job_applicator.exceptions import LoginRequiredError, ScraperError, SelectorHealthError
from job_applicator.models import (
    BoardSelectorHealth,
    JobBoard,
    SelectorHealthReport,
    SelectorProbe,
    SelectorProbeResult,
    SelectorProbeStatus,
)
from job_applicator.scrapers.base import SearchParams
from job_applicator.selector_registry import APPLY_SURFACE, SEARCH_SURFACE, selector_probes
from job_applicator.utils.cookies import load_cookies
from job_applicator.utils.logging import get_logger
from job_applicator.utils.path import safe_filename_slug, set_owner_only
from job_applicator.utils.url import host_matches

if TYPE_CHECKING:
    from job_applicator.browser.manager import BrowserManager

logger = get_logger("selector_health")

_DEBUG_DIR = Path.home() / ".job-applicator" / "debug" / "selector-health"


class SelectorHealthService:
    """Probe known board selectors against live pages without scraping or submitting."""

    def __init__(self, browser: BrowserManager, settings: AppSettings) -> None:
        self._browser = browser
        self._settings = settings

    async def probe_search(
        self, site: str, query: str, location: str = "", *, max_cards: int = 3
    ) -> SelectorHealthReport:
        """Navigate to a search page and validate registered search selectors."""
        board = _board(site)
        probes = selector_probes(board, SEARCH_SURFACE)
        if not probes:
            return _skipped_report(
                board, SEARCH_SURFACE, f"No selector probes registered for {site}"
            )

        if board == JobBoard.LINKEDIN:
            return await self._probe_linkedin_search(query, location, max_cards=max_cards)
        if board == JobBoard.INDEED:
            return await self._probe_indeed_search(query, location, max_cards=max_cards)
        return _skipped_report(board, SEARCH_SURFACE, f"No search probe implemented for {site}")

    async def probe_apply(self, site: str, job_url: str) -> SelectorHealthReport:
        """Open a job page and validate Easy Apply selectors without submitting."""
        board = _board(site)
        probes = selector_probes(board, APPLY_SURFACE)
        if not probes:
            return _skipped_report(
                board,
                APPLY_SURFACE,
                f"No apply selector probes registered for {site}",
                url=job_url,
            )
        if board != JobBoard.LINKEDIN:
            return _skipped_report(
                board,
                APPLY_SURFACE,
                f"{board.display_name} apply selector health is not implemented.",
                url=job_url,
            )
        return await self._probe_linkedin_apply(job_url)

    async def _probe_linkedin_search(
        self, query: str, location: str, *, max_cards: int
    ) -> SelectorHealthReport:
        from job_applicator.scrapers.linkedin import LinkedInScraper

        scraper = LinkedInScraper(self._browser, self._settings)
        context = await scraper._get_context()
        if not await scraper._ensure_session(context):
            raise LoginRequiredError(
                "No authenticated LinkedIn session found. Run `job-applicator login` first."
            )

        page = await scraper._new_stealth_page(context)
        try:
            params = SearchParams(
                query=query,
                location=location,
                max_results=max_cards,
                board=JobBoard.LINKEDIN,
            )
            geo_id = (
                await scraper._resolve_geo_id(page, params.location) if params.location else None
            )
            search_url = scraper._build_search_url(params, geo_id)
            await navigate(page, search_url)
            await random_delay(2.0, 3.0)
            await scraper._raise_if_blocked(page)
            report = await self._probe_search_page(JobBoard.LINKEDIN, page, max_cards=max_cards)
            return await self._attach_failure_artifacts(report, page)
        finally:
            await page.close()

    async def _probe_indeed_search(
        self, query: str, location: str, *, max_cards: int
    ) -> SelectorHealthReport:
        from job_applicator.scrapers.indeed import IndeedScraper

        scraper = IndeedScraper(self._browser, self._settings)
        context = await self._browser.persistent_context()
        await load_cookies(context, scraper.COOKIE_PATH)
        page = await scraper._new_stealth_page(context)
        try:
            params = SearchParams(
                query=query,
                location=location,
                max_results=max_cards,
                board=JobBoard.INDEED,
            )
            await navigate(page, scraper._build_search_url(params))
            await random_delay(2.0, 3.0)
            if await scraper._is_blocked(page):
                raise ScraperError(
                    "Indeed returned an anti-bot challenge during selector health probing.",
                    context={"url": page.url},
                )
            host = urlsplit(page.url).netloc
            if host_matches(host, "indeed.com"):
                scraper._resolved_base = f"https://{host}"
            report = await self._probe_search_page(JobBoard.INDEED, page, max_cards=max_cards)
            return await self._attach_failure_artifacts(report, page)
        finally:
            await page.close()

    async def _probe_search_page(
        self, board: JobBoard, page: Page, *, max_cards: int
    ) -> SelectorHealthReport:
        probes = selector_probes(board, SEARCH_SURFACE)
        card_probe = probes[0] if probes else None
        if card_probe is not None:
            await _wait_for_any(page, card_probe.selectors)
        sampled_cards = (
            await _matching_elements(page, card_probe.selectors, max_items=max_cards)
            if card_probe is not None
            else []
        )
        results = await _evaluate_probes(probes, page, sampled_cards)
        return _report(board, SEARCH_SURFACE, results, url=page.url)

    async def _probe_linkedin_apply(self, job_url: str) -> SelectorHealthReport:
        from job_applicator.scrapers.linkedin import LinkedInScraper

        scraper = LinkedInScraper(self._browser, self._settings)
        context = await scraper._get_context()
        if not await scraper._ensure_session(context):
            raise LoginRequiredError(
                "No authenticated LinkedIn session found. Run `job-applicator login` first."
            )

        page = await scraper._new_stealth_page(context)
        try:
            await navigate(page, job_url)
            await random_delay(2.0, 3.0)
            await scraper._raise_if_blocked(page)

            probes = selector_probes(JobBoard.LINKEDIN, APPLY_SURFACE)
            entry_probe = probes[0]
            await _wait_for_any(page, entry_probe.selectors)
            results = [await _evaluate_probe(entry_probe, page, url=page.url)]
            if results[0].status == SelectorProbeStatus.PASSED:
                opened, details = await _click_first(page, entry_probe.selectors)
                if opened:
                    await random_delay(1.0, 2.0)
                    results.append(await _linkedin_form_controls_result(page, url=page.url))
                    results.extend(await _evaluate_probes(probes[1:], page, None))
                else:
                    results.append(
                        SelectorProbeResult(
                            board=JobBoard.LINKEDIN,
                            surface=APPLY_SURFACE,
                            name="Easy Apply modal open",
                            selector=entry_probe.selector,
                            required=True,
                            matched_count=results[0].matched_count,
                            status=SelectorProbeStatus.FAIL,
                            details=details,
                            url=page.url,
                        )
                    )
                    results.extend(await _evaluate_probes(probes[1:], page, None))
            else:
                external_skip = await _linkedin_external_apply_skip_report(page, url=page.url)
                if external_skip is not None:
                    return external_skip
                results.extend(await _evaluate_probes(probes[1:], page, None))

            report = _report(JobBoard.LINKEDIN, APPLY_SURFACE, results, url=page.url)
            return await self._attach_failure_artifacts(report, page)
        finally:
            await page.close()

    async def _attach_failure_artifacts(
        self, report: SelectorHealthReport, page: Page | None
    ) -> SelectorHealthReport:
        if report.status != SelectorProbeStatus.FAIL:
            return report
        artifacts = await write_failure_diagnostics(report, page)
        if not artifacts:
            return report
        boards: list[BoardSelectorHealth] = []
        for board_health in report.boards:
            results = [
                result.model_copy(update={"artifacts": [*result.artifacts, *artifacts]})
                if result.status == SelectorProbeStatus.FAIL
                else result
                for result in board_health.results
            ]
            boards.append(
                board_health.model_copy(
                    update={"results": results, "artifacts": [*board_health.artifacts, *artifacts]}
                )
            )
        return report.model_copy(
            update={"boards": boards, "artifacts": [*report.artifacts, *artifacts]}
        )


async def _wait_for_any(page: Page, selectors: list[str], timeout_ms: int = 5_000) -> None:
    for selector in selectors:
        if await wait_for_selector(page, selector, timeout_ms=timeout_ms):
            return


async def _matching_elements(
    page: Page, selectors: list[str], *, max_items: int
) -> list[ElementHandle]:
    for selector in selectors:
        try:
            matches = await page.query_selector_all(selector)
        except Exception as exc:
            logger.debug("Selector probe could not query %s: %s", selector, exc)
            continue
        if matches:
            return matches[:max_items]
    return []


async def _click_first(page: Page, selectors: list[str]) -> tuple[bool, str]:
    for selector in selectors:
        try:
            element = await page.query_selector(selector)
        except Exception as exc:
            return False, f"{selector}: {type(exc).__name__}: {exc}"
        if element is None:
            continue
        try:
            await element.click(timeout=5_000)
        except Exception as exc:
            return False, f"{selector}: could not click ({type(exc).__name__}: {exc})"
        return True, f"opened with {selector}"
    return False, "Easy Apply button was not found when opening the modal."


async def _linkedin_form_controls_result(page: Page, *, url: str) -> SelectorProbeResult:
    from job_applicator.applicators.linkedin import (
        _ADVANCE_BUTTON_SELECTORS,
        _SUBMIT_BUTTON_SELECTORS,
    )

    selectors = [*_ADVANCE_BUTTON_SELECTORS, *_SUBMIT_BUTTON_SELECTORS]
    matched, details = await _count_selectors(page, selectors)
    return SelectorProbeResult(
        board=JobBoard.LINKEDIN,
        surface=APPLY_SURFACE,
        name="Easy Apply form controls",
        selector=", ".join(selectors),
        required=True,
        matched_count=matched,
        status=SelectorProbeStatus.PASSED if matched else SelectorProbeStatus.FAIL,
        details=details,
        url=url,
    )


async def _linkedin_external_apply_skip_report(
    page: Page, *, url: str
) -> SelectorHealthReport | None:
    from job_applicator.applicators.linkedin import EXTERNAL_APPLY_BUTTON_SELECTORS

    selectors = list(EXTERNAL_APPLY_BUTTON_SELECTORS)
    matched, details = await _count_selectors(page, selectors)
    if matched <= 0:
        return None

    skipped: list[SelectorProbeResult] = [
        SelectorProbeResult(
            board=JobBoard.LINKEDIN,
            surface=APPLY_SURFACE,
            name="Easy Apply button",
            selector='button:has-text("Easy Apply")',
            required=True,
            matched_count=0,
            status=SelectorProbeStatus.SKIPPED,
            details="No Easy Apply button; external apply button detected.",
            url=url,
        ),
        SelectorProbeResult(
            board=JobBoard.LINKEDIN,
            surface=APPLY_SURFACE,
            name="external apply button",
            selector=", ".join(selectors),
            required=False,
            matched_count=matched,
            status=SelectorProbeStatus.SKIPPED,
            details=details,
            url=url,
        ),
    ]
    for probe in selector_probes(JobBoard.LINKEDIN, APPLY_SURFACE)[1:]:
        skipped.append(
            SelectorProbeResult(
                board=probe.board,
                surface=probe.surface,
                name=probe.name,
                selector=probe.selector,
                required=probe.required,
                matched_count=0,
                status=SelectorProbeStatus.SKIPPED,
                details="External apply job; Easy Apply form selectors are not applicable.",
                url=url,
            )
        )
    return _report(JobBoard.LINKEDIN, APPLY_SURFACE, skipped, url=url)


async def _evaluate_probes(
    probes: tuple[SelectorProbe, ...], page: Page, sampled_cards: list[ElementHandle] | None
) -> list[SelectorProbeResult]:
    results: list[SelectorProbeResult] = []
    for probe in probes:
        target: Any = page if probe.scope == "page" else sampled_cards
        results.append(await _evaluate_probe(probe, target, url=page.url))
    return results


async def _evaluate_probe(probe: SelectorProbe, target: Any, *, url: str) -> SelectorProbeResult:
    if target is None or target == []:
        status = SelectorProbeStatus.FAIL if probe.required else SelectorProbeStatus.WARN
        return SelectorProbeResult(
            board=probe.board,
            surface=probe.surface,
            name=probe.name,
            selector=probe.selector,
            required=probe.required,
            matched_count=0,
            status=status,
            details="no sampled card was available for scoped selector probe",
            url=url,
        )

    matched, details = await _count_selectors(target, probe.selectors)
    if matched > 0:
        status = SelectorProbeStatus.PASSED
    elif probe.required:
        status = SelectorProbeStatus.FAIL
    else:
        status = SelectorProbeStatus.WARN
    return SelectorProbeResult(
        board=probe.board,
        surface=probe.surface,
        name=probe.name,
        selector=probe.selector,
        required=probe.required,
        matched_count=matched,
        status=status,
        details=details,
        url=url,
    )


async def _count_selectors(target: Any, selectors: list[str]) -> tuple[int, str]:
    if isinstance(target, list):
        total = 0
        card_details: list[str] = []
        for index, item in enumerate(target, start=1):
            matched, item_details = await _count_selectors(item, selectors)
            total += matched
            card_details.append(f"card {index}: {item_details}")
        return total, " | ".join(card_details)

    total = 0
    details: list[str] = []
    for selector in selectors:
        try:
            matches = await target.query_selector_all(selector)
        except Exception as exc:
            details.append(f"{selector}=error:{type(exc).__name__}")
            continue
        count = len(matches)
        total += count
        details.append(f"{selector}={count}")
    return total, "; ".join(details)


def _report(
    board: JobBoard, surface: str, results: list[SelectorProbeResult], *, url: str = ""
) -> SelectorHealthReport:
    board_health = BoardSelectorHealth(
        board=board,
        surface=surface,
        status=aggregate_status([result.status for result in results]),
        url=url,
        results=results,
    )
    return SelectorHealthReport(status=board_health.status, boards=[board_health])


def aggregate_status(statuses: list[SelectorProbeStatus]) -> SelectorProbeStatus:
    """Aggregate probe statuses with required misses taking priority."""
    if not statuses:
        return SelectorProbeStatus.SKIPPED
    if SelectorProbeStatus.FAIL in statuses:
        return SelectorProbeStatus.FAIL
    if SelectorProbeStatus.WARN in statuses:
        return SelectorProbeStatus.WARN
    if all(status == SelectorProbeStatus.SKIPPED for status in statuses):
        return SelectorProbeStatus.SKIPPED
    return SelectorProbeStatus.PASSED


async def write_failure_diagnostics(
    report: SelectorHealthReport, page: Page | None, debug_dir: Path | None = None
) -> list[str]:
    """Write selector-health diagnostics. Best-effort; never masks the probe failure."""
    try:
        output_dir = debug_dir or _DEBUG_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        set_owner_only(output_dir, 0o700)
        board = report.boards[0].board.value if report.boards else "unknown"
        surface = report.boards[0].surface if report.boards else "unknown"
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        prefix = output_dir / f"{stamp}-{safe_filename_slug(board)}-{safe_filename_slug(surface)}"
        summary_path = prefix.with_suffix(".txt")
        summary_path.write_text(_diagnostic_summary(report), encoding="utf-8")
        artifacts = [str(summary_path)]

        if page is not None:
            try:
                html = await page.content()
                html_path = prefix.with_suffix(".html")
                html_path.write_text(html, encoding="utf-8")
                artifacts.append(str(html_path))
            except Exception as exc:
                logger.debug("Could not write selector-health HTML dump: %s", exc)
            try:
                screenshot_path = prefix.with_suffix(".png")
                await screenshot(page, screenshot_path)
                artifacts.append(str(screenshot_path))
            except Exception as exc:
                logger.debug("Could not write selector-health screenshot: %s", exc)
        logger.warning("Selector health diagnostics written to %s", output_dir)
        return artifacts
    except Exception as exc:
        logger.warning("Could not write selector-health diagnostics: %s", exc)
        return []


def _diagnostic_summary(report: SelectorHealthReport) -> str:
    lines = [f"status: {report.status.value}", f"generated_at: {report.generated_at.isoformat()}"]
    for board in report.boards:
        lines += [
            "",
            f"board: {board.board.value}",
            f"surface: {board.surface}",
            f"url: {board.url}",
        ]
        for result in board.results:
            lines.append(
                f"- {result.status.value} {result.name} required={result.required} "
                f"matched={result.matched_count} selector={result.selector} "
                f"details={result.details}"
            )
    return "\n".join(lines)


def _skipped_report(
    board: JobBoard, surface: str, details: str, *, url: str = ""
) -> SelectorHealthReport:
    result = SelectorProbeResult(
        board=board,
        surface=surface,
        name="selector registry",
        selector="",
        required=False,
        matched_count=0,
        status=SelectorProbeStatus.SKIPPED,
        details=details,
        url=url,
    )
    return _report(board, surface, [result], url=url)


def _board(site: str) -> JobBoard:
    try:
        return JobBoard(site)
    except ValueError as exc:
        raise SelectorHealthError(f"Unsupported site for selector health: {site}") from exc
