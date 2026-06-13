"""LinkedIn job applicator — fills and submits applications."""

from __future__ import annotations

from pathlib import Path

from playwright.async_api import Page

from job_applicator.applicators.base import BaseApplicator
from job_applicator.browser.actions import (
    click,
    navigate,
    random_delay,
    screenshot,
    wait_for_selector,
)
from job_applicator.browser.manager import BrowserManager
from job_applicator.config import AppSettings
from job_applicator.models import ApplicationResult, ApplicationStatus, JobListing
from job_applicator.utils.logging import get_logger

logger = get_logger("applicators.linkedin")


class LinkedInApplicator(BaseApplicator):
    """Submits job applications on LinkedIn."""

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config

    async def apply(self, job: JobListing, cover_letter: str | None = None) -> ApplicationResult:
        """Apply to a LinkedIn job."""
        try:
            async with self._browser.new_page() as page:
                await navigate(page, str(job.url))
                await random_delay(2.0, 3.0)

                # Check for "Easy Apply" button
                easy_apply = await wait_for_selector(
                    page, 'button:has-text("Easy Apply")', timeout_ms=5_000
                )

                if easy_apply:
                    return await self._easy_apply(page, job, cover_letter)
                else:
                    return await self._external_apply(page, job)

        except Exception as exc:
            logger.error("Failed to apply to %s at %s: %s", job.title, job.company, exc)
            if self._config.screenshot_on_error:
                try:
                    async with self._browser.new_page() as page:
                        await navigate(page, str(job.url))
                        await screenshot(page, Path(f"error_{job.company}_{job.title}.png"))
                except Exception as e:
                    logger.debug("Screenshot failed: %s", e)
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.FAILED,
                error_message=str(exc),
            )

    async def _easy_apply(
        self, page: Page, job: JobListing, cover_letter: str | None
    ) -> ApplicationResult:
        """Handle LinkedIn Easy Apply flow."""
        await click(page, 'button:has-text("Easy Apply")')
        await random_delay(1.0, 2.0)

        # Fill contact info if present
        await self._fill_form_fields(page)

        # Upload resume if file input exists
        resume_input = await page.query_selector('input[type="file"]')
        if resume_input and self._config.resume_path:
            await resume_input.set_input_files(self._config.resume_path)
            await random_delay(1.0, 2.0)

        # Fill cover letter if provided and field exists
        if cover_letter:
            cl_field = await page.query_selector('textarea[aria-label*="cover" i]')
            if cl_field:
                await cl_field.fill(cover_letter)

        # Click through multi-step forms
        for _ in range(5):
            next_btn = await page.query_selector('button:has-text("Next")')
            if next_btn:
                await next_btn.click()
                await random_delay(0.5, 1.0)
                await self._fill_form_fields(page)
            else:
                break

        # Submit
        submit_btn = await page.query_selector('button:has-text("Submit")')
        if submit_btn:
            await submit_btn.click()
            await random_delay(2.0, 3.0)

            # Check for confirmation
            confirmed = await wait_for_selector(
                page, 'div:has-text("Application sent")', timeout_ms=5_000
            )
            if confirmed:
                logger.info("Successfully applied to %s at %s", job.title, job.company)
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.SUBMITTED,
                    cover_letter=cover_letter,
                )

        return ApplicationResult(
            job=job,
            status=ApplicationStatus.FAILED,
            error_message="Could not complete Easy Apply flow",
        )

    async def _external_apply(self, page: Page, job: JobListing) -> ApplicationResult:
        """Handle external application redirect."""
        # Find and click the apply link
        apply_link = await page.query_selector('a:has-text("Apply")')
        if apply_link:
            href = await apply_link.get_attribute("href")
            logger.info("External application redirect to: %s", href)
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.SKIPPED,
                notes="External application required — manual follow-up needed",
            )

        return ApplicationResult(
            job=job,
            status=ApplicationStatus.SKIPPED,
            notes="No apply button found",
        )

    async def _fill_form_fields(self, page: Page) -> None:
        """Auto-fill common form fields from profile."""
        profile = self._config

        name_parts = profile.profile_name.split() if profile.profile_name else []
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""

        field_mappings = {
            'input[name*="first" i]': first_name,
            'input[name*="last" i]': last_name,
            'input[name*="email" i]': profile.target.linkedin_email,
            'input[name*="phone" i]': "",
        }

        for selector, value in field_mappings.items():
            if value:
                el = await page.query_selector(selector)
                if el:
                    try:
                        await el.fill(value)
                    except Exception as e:
                        logger.debug("Could not fill %s: %s", selector, e)

    async def check_already_applied(self, job: JobListing) -> bool:
        """Check if already applied to this job."""
        async with self._browser.new_page() as page:
            await navigate(page, str(job.url))
            await random_delay(1.0, 2.0)

            applied = await wait_for_selector(page, 'button:has-text("Applied")', timeout_ms=3_000)
            return bool(applied)
