"""Indeed applicator.

Mirrors the LinkedIn applicator's safety model: dry-run by default (never submits
without explicit opt-in). Many Indeed postings redirect to an external ATS, which
is reported as SKIPPED. The actual on-site "Easily apply" submission flow is NOT
implemented/validated — it refuses rather than guessing at a real submission.
"""

from __future__ import annotations

from job_applicator.applicators.base import BaseApplicator
from job_applicator.browser.actions import navigate, random_delay, wait_for_selector
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings
from job_applicator.models import ApplicationResult, ApplicationStatus, JobListing
from job_applicator.utils.logging import get_logger

logger = get_logger("applicators.indeed")


class IndeedApplicator(BaseApplicator):
    """Submits applications on Indeed (best-effort; not validated against live Indeed)."""

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config

    async def apply(
        self, job: JobListing, cover_letter: str | None = None, submit: bool = False
    ) -> ApplicationResult:
        """Apply to an Indeed job.

        Only on-site "Easily apply" postings are automatable; many Indeed jobs
        redirect to an external ATS and are reported as SKIPPED. As with
        LinkedIn, when ``submit`` is False (default) nothing is submitted.
        """
        try:
            async with self._browser.persistent_page() as page:
                await navigate(page, str(job.url))
                await random_delay(2.0, 3.0)

                easily_apply = await wait_for_selector(
                    page,
                    'button:has-text("Easily apply"), #indeedApplyButton',
                    timeout_ms=5_000,
                )
                if not easily_apply:
                    return ApplicationResult(
                        job=job,
                        status=ApplicationStatus.SKIPPED,
                        notes="External application required — manual follow-up needed",
                    )

                if not submit:
                    logger.info(
                        "DRY RUN — Indeed 'Easily apply' detected for %s at %s; NOT submitted.",
                        job.title,
                        job.company,
                    )

                async def _do_submit() -> ApplicationResult:
                    # Indeed is scoped to search/match only — automated apply is
                    # intentionally unsupported (Cloudflare anti-bot + ToS risk), not a
                    # pending feature. Return a clean SKIPPED result; never auto-submit.
                    return ApplicationResult(
                        job=job,
                        status=ApplicationStatus.SKIPPED,
                        notes=(
                            "Indeed automated apply is unsupported (search-only by design) — "
                            "apply manually on indeed.com."
                        ),
                    )

                # Route through the same base-class dry-run gate as LinkedIn.
                return await self._gated_submit(
                    submit=submit,
                    job=job,
                    cover_letter=cover_letter,
                    do_submit=_do_submit,
                    dry_run_note="Indeed is search-only — automated apply is unsupported; "
                    "apply manually on indeed.com.",
                )
        except Exception as exc:
            logger.error("Failed to apply to %s at %s: %s", job.title, job.company, exc)
            return ApplicationResult(
                job=job, status=ApplicationStatus.FAILED, error_message=str(exc)
            )

    async def check_already_applied(self, job: JobListing) -> bool:
        """Best-effort check for an existing Indeed application."""
        async with self._browser.persistent_page() as page:
            await navigate(page, str(job.url))
            await random_delay(1.0, 2.0)
            return bool(await wait_for_selector(page, 'text="Applied"', timeout_ms=3_000))
