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
from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    DryRunValidation,
    JobListing,
)
from job_applicator.utils.logging import get_logger

logger = get_logger("applicators.linkedin")


class LinkedInApplicator(BaseApplicator):
    """Submits job applications on LinkedIn."""

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config

    async def apply(
        self, job: JobListing, cover_letter: str | None = None, submit: bool = False
    ) -> ApplicationResult:
        """Apply to a LinkedIn job.

        Runs in the manager's shared persistent context so the authenticated
        session (seeded via ``job-applicator login`` / ``import-cookies``) is
        reused — Easy Apply requires being logged in.

        When ``submit`` is False (default), the form is filled but NOT submitted
        (a dry run); the final "Submit application" click only happens when
        ``submit`` is True. This prevents automated runs from sending real
        applications without explicit opt-in.
        """
        page: Page | None = None
        try:
            async with self._browser.persistent_page() as page:
                await navigate(page, str(job.url))
                await random_delay(2.0, 3.0)

                # Skip if already applied (avoids duplicate submissions). Use a
                # non-blocking query: the Applied state renders with the page, and
                # wait_for_selector would block the full timeout on every fresh
                # job where the element is absent (~3s wasted per listing).
                if await page.query_selector('button:has-text("Applied")'):
                    logger.info("Already applied to %s at %s", job.title, job.company)
                    return ApplicationResult(job=job, status=ApplicationStatus.ALREADY_APPLIED)

                # Check for "Easy Apply" button
                easy_apply = await wait_for_selector(
                    page, 'button:has-text("Easy Apply")', timeout_ms=5_000
                )

                if easy_apply:
                    return await self._easy_apply(page, job, cover_letter, submit)
                else:
                    return await self._external_apply(page, job)

        except Exception as exc:
            logger.error("Failed to apply to %s at %s: %s", job.title, job.company, exc)
            # ``page`` is None if persistent_page() entry failed (e.g. browser
            # not started); only screenshot when we actually have a page.
            if self._config.screenshot_on_error and page is not None:
                # Capture the page in its actual failure state rather than
                # re-navigating to a fresh page (which hid the real error).
                try:
                    await screenshot(page, Path(f"error_{job.company}_{job.title}.png"))
                except Exception as e:
                    logger.debug("Screenshot failed: %s", e)
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.FAILED,
                error_message=str(exc),
            )

    async def _easy_apply(
        self, page: Page, job: JobListing, cover_letter: str | None, submit: bool
    ) -> ApplicationResult:
        """Handle LinkedIn Easy Apply flow.

        Fills the form and advances through multi-step pages, then stops at the
        final "Submit application" step. The submit click only happens when
        ``submit`` is True; otherwise this is a dry run that submits nothing.
        """
        validation = DryRunValidation(easy_apply_button_found=True)

        await click(page, 'button:has-text("Easy Apply")')
        await random_delay(1.0, 2.0)

        # Fill contact info if present
        fields_filled = await self._fill_form_fields(page)
        validation.fields_filled = fields_filled

        # Upload resume if file input exists
        resume_input = await page.query_selector('input[type="file"]')
        if resume_input and self._config.resume_path:
            await resume_input.set_input_files(self._config.resume_path)
            validation.resume_uploaded = True
            await random_delay(1.0, 2.0)

        # Fill cover letter if provided and field exists
        if cover_letter:
            cl_field = await page.query_selector('textarea[aria-label*="cover" i]')
            if cl_field:
                await cl_field.fill(cover_letter)
                validation.cover_letter_field_found = True

        # Advance through multi-step forms (Next / Review) — never Submit here.
        for _ in range(6):
            advance = await page.query_selector(
                'button:has-text("Next"), button:has-text("Review")'
            )
            if not advance:
                break
            await advance.click()
            await random_delay(0.5, 1.0)
            fields_filled.extend(await self._fill_form_fields(page))
            validation.fields_filled = fields_filled

        # Match the final submit by either label ("Submit application" is the
        # usual text; fall back to a bare "Submit") so a label change/locale
        # doesn't make a real application silently fail to send.
        submit_btn = await page.query_selector(
            'button:has-text("Submit application"), button:has-text("Submit")'
        )
        if not submit_btn:
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.FAILED,
                error_message="Could not reach the Submit step of the Easy Apply flow",
                dry_run=validation,
            )

        validation.reached_submit = True

        if not submit:
            logger.info(
                "DRY RUN — Easy Apply form prepared for %s at %s; NOT submitted "
                "(re-run with --submit to actually apply).",
                job.title,
                job.company,
            )

        async def _do_submit() -> ApplicationResult:
            await submit_btn.click()
            await random_delay(2.0, 3.0)
            confirmed = await wait_for_selector(
                page, 'div:has-text("Application sent")', timeout_ms=5_000
            )
            if confirmed:
                logger.info("Successfully applied to %s at %s", job.title, job.company)
                return ApplicationResult(
                    job=job,
                    status=ApplicationStatus.SUBMITTED,
                    cover_letter=cover_letter,
                    dry_run=validation,
                )
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.FAILED,
                error_message="Submit clicked but no confirmation was detected",
                dry_run=validation,
            )

        # The dry-run gate lives in the base class so it cannot be bypassed.
        result = await self._gated_submit(
            submit=submit,
            job=job,
            cover_letter=cover_letter,
            do_submit=_do_submit,
            dry_run_note="DRY RUN: form prepared but not submitted. Use --submit to apply.",
        )
        result.dry_run = validation
        return result

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

    async def _fill_form_fields(self, page: Page) -> list[str]:
        """Auto-fill common form fields from profile.

        Returns the list of field selectors that were actually filled.
        """
        profile = self._config
        filled: list[str] = []

        name_parts = profile.profile_name.split() if profile.profile_name else []
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""

        field_mappings = {
            'input[name*="first" i]': (first_name, "firstName"),
            'input[name*="last" i]': (last_name, "lastName"),
            'input[name*="email" i]': (profile.target.linkedin_email, "email"),
            'input[name*="phone" i]': ("", "phone"),
        }

        for selector, (value, label) in field_mappings.items():
            if value:
                el = await page.query_selector(selector)
                if el:
                    try:
                        await el.fill(value)
                        filled.append(label)
                    except Exception as e:
                        logger.debug("Could not fill %s: %s", selector, e)
        return filled

    async def check_already_applied(self, job: JobListing) -> bool:
        """Check if already applied to this job."""
        async with self._browser.persistent_page() as page:
            await navigate(page, str(job.url))
            await random_delay(1.0, 2.0)

            applied = await wait_for_selector(page, 'button:has-text("Applied")', timeout_ms=3_000)
            return bool(applied)
