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
from job_applicator.exceptions import FormFillingError
from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    DryRunValidation,
    JobListing,
)
from job_applicator.utils.logging import get_logger
from job_applicator.utils.path import safe_filename_slug, set_owner_only

logger = get_logger("applicators.linkedin")

# Error screenshots land here (not cwd), with slugified names — see the apply() failure path.
_DEBUG_DIR = Path.home() / ".job-applicator" / "debug"

# LinkedIn's resume upload accepts these document types.
_RESUME_UPLOAD_TYPES = (".pdf", ".doc", ".docx")
_ADVANCE_BUTTON_SELECTOR = (
    'button:has-text("Next"), '
    'button:has-text("Continue"), '
    'button[aria-label*="Next" i], '
    'button[aria-label*="Continue" i], '
    'button:has-text("Review"), '
    'button[aria-label*="Review" i]'
)
_SUBMIT_BUTTON_SELECTOR = (
    'button:has-text("Submit application"), '
    'button:has-text("Submit"), '
    'button[aria-label*="Submit application" i], '
    'button[aria-label*="Submit" i]'
)


def _validated_resume_upload_path(config: AppSettings) -> Path:
    """The configured resume validated for upload — it EXISTS (via ``config.get_resume_path``) and
    is a LinkedIn-supported type. Raises a clean typed error BEFORE ``set_input_files`` so a
    missing / wrong-type resume fails with a clear message, not an opaque Playwright error
    mid-apply (which would otherwise surface as a bare FAILED with a stack-trace-ish detail).
    """
    path = config.get_resume_path()  # ResumeNotFoundError if missing (previously dead code)
    if path.suffix.lower() not in _RESUME_UPLOAD_TYPES:
        raise FormFillingError(
            f"Resume type {path.suffix or '(none)'!r} is not supported for LinkedIn upload "
            "(use .pdf / .doc / .docx); convert it and retry."
        )
    return path


def _resume_contact(config: AppSettings) -> tuple[str, str]:
    """Best-effort email/phone parsed from the configured résumé."""
    if not config.resume_path:
        return "", ""
    try:
        from job_applicator.documents.resume import ResumeLoader

        resume = ResumeLoader().load(config.resume_path)
    except Exception as exc:
        logger.debug("Could not parse resume contact for Easy Apply autofill: %s", exc)
        return "", ""
    return resume.email, resume.phone


class LinkedInApplicator(BaseApplicator):
    """Submits job applications on LinkedIn."""

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config
        self._resume_contact: tuple[str, str] | None = None

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
                    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                    set_owner_only(_DEBUG_DIR, 0o700)  # screenshots may show the authed profile
                    name = (
                        f"error_{safe_filename_slug(job.company)}_"
                        f"{safe_filename_slug(job.title)}.png"
                    )
                    await screenshot(page, _DEBUG_DIR / name)
                except Exception as e:
                    logger.debug("Screenshot failed: %s", e)
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.FAILED,
                error_message=str(exc),
                cover_letter=cover_letter,
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
        fields_filled, fill_errors = await self._fill_form_fields(page)
        validation.fields_filled = fields_filled
        validation.fill_errors = fill_errors

        # Upload resume if file input exists
        if self._config.resume_path:
            validation.resume_uploaded = await self._upload_resume_if_present(page)
            if validation.resume_uploaded:
                await random_delay(1.0, 2.0)

        # Fill cover letter if provided and field exists
        if cover_letter:
            cl_field = await page.query_selector('textarea[aria-label*="cover" i]')
            if cl_field:
                # Paste-like: focus + a brief human pause so the sequence reads as a deliberate
                # paste (click → text appears) rather than a value materializing on its own. The
                # focus-click is GUARDED — click imposes Receives-Events/Stable actionability that
                # fill does not, so a present-but-obscured textarea must NOT abort the apply; on any
                # click failure, fall straight through to the plain fill (the prior behaviour).
                # fill() still sets the whole value in one shot, so this only ADDS a trusted event +
                # pause — it does not make the value-set itself non-atomic.
                try:
                    await cl_field.click()
                    await random_delay(0.5, 1.0)
                except Exception as e:
                    logger.debug("Cover-letter focus-click skipped (%s); filling directly", e)
                await cl_field.fill(cover_letter)
                validation.cover_letter_field_found = True

        # Advance through multi-step forms (Next / Review) — never Submit here.
        for _ in range(6):
            advance = await page.query_selector(_ADVANCE_BUTTON_SELECTOR)
            if not advance:
                break
            await advance.click()
            await random_delay(0.5, 1.0)
            more_filled, more_errors = await self._fill_form_fields(page)
            fields_filled.extend(more_filled)
            fill_errors.extend(more_errors)
            validation.fields_filled = fields_filled
            validation.fill_errors = fill_errors
            if self._config.resume_path and not validation.resume_uploaded:
                validation.resume_uploaded = await self._upload_resume_if_present(page)
                if validation.resume_uploaded:
                    await random_delay(1.0, 2.0)

        # Match the final submit by either label ("Submit application" is the
        # usual text; fall back to a bare "Submit") so a label change/locale
        # doesn't make a real application silently fail to send.
        submit_btn = await page.query_selector(_SUBMIT_BUTTON_SELECTOR)
        if not submit_btn:
            dump = await self._dump_apply_debug(page, job, validation)
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.FAILED,
                error_message=(
                    "Could not reach the Submit step of the Easy Apply flow"
                    + (f" (debug saved to {dump})" if dump else "")
                ),
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

    async def _upload_resume_if_present(self, page: Page) -> bool:
        """Upload the configured résumé when LinkedIn exposes a file input or upload button."""
        resume_input = await page.query_selector('input[type="file"]')
        if resume_input:
            resume_path = _validated_resume_upload_path(self._config)  # exists + supported type
            await resume_input.set_input_files(str(resume_path))
            return True

        upload_button = await page.query_selector(
            'button:has-text("Upload resume"), button[aria-label*="Upload resume" i]'
        )
        if not upload_button:
            return False
        resume_path = _validated_resume_upload_path(self._config)
        async with page.expect_file_chooser() as chooser_info:
            await upload_button.click()
        chooser = await chooser_info.value
        await chooser.set_files(str(resume_path))
        return True

    async def _dump_apply_debug(
        self, page: Page, job: JobListing, validation: DryRunValidation
    ) -> Path | None:
        """Write live Easy Apply diagnostics when the final Submit step is unreachable."""
        try:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            set_owner_only(_DEBUG_DIR, 0o700)
            slug = (
                f"linkedin-apply-{safe_filename_slug(job.company)}-{safe_filename_slug(job.title)}"
            )
            path = _DEBUG_DIR / f"{slug}.txt"
            buttons = await page.query_selector_all("button")
            inputs = await page.query_selector_all("input, textarea, select")
            lines = [
                f"url: {page.url}",
                f"job: {job.title} at {job.company}",
                f"reached_submit: {validation.reached_submit}",
                f"resume_uploaded: {validation.resume_uploaded}",
                f"fields_filled: {', '.join(validation.fields_filled) or 'none'}",
                f"fill_errors: {', '.join(validation.fill_errors) or 'none'}",
                "",
                "buttons:",
            ]
            for i, button in enumerate(buttons[:40], start=1):
                try:
                    text = (await button.inner_text()).strip().replace("\n", " ")
                except Exception:
                    text = "<unreadable>"
                try:
                    aria = await button.get_attribute("aria-label")
                except Exception:
                    aria = None
                lines.append(f"{i}. text={text!r} aria={aria!r}")
            lines += ["", "inputs:"]
            for i, field in enumerate(inputs[:40], start=1):
                attrs: list[str] = []
                for attr in ("type", "name", "id", "aria-label", "placeholder", "required"):
                    try:
                        value = await field.get_attribute(attr)
                    except Exception:
                        value = None
                    if value:
                        attrs.append(f"{attr}={value!r}")
                try:
                    tag = await field.evaluate("el => el.tagName.toLowerCase()")
                except Exception:
                    tag = "field"
                lines.append(f"{i}. {tag} {' '.join(attrs) or '<no attrs>'}")
            path.write_text("\n".join(lines), encoding="utf-8")
            logger.warning("LinkedIn Easy Apply debug dump written to %s", path)
            return path
        except Exception as exc:
            logger.warning("Could not write LinkedIn Easy Apply debug dump: %s", exc)
            return None

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

    async def _fill_form_fields(self, page: Page) -> tuple[list[str], list[str]]:
        """Auto-fill common form fields from profile.

        Returns ``(filled, errors)``: labels that were filled, and labels of fields that were
        PRESENT but failed to fill (distinct from absent fields, which are silently skipped).
        A present-but-failed field is surfaced (warned + carried into DryRunValidation) so a real
        submit isn't sent with a silently-missing required field.
        """
        profile = self._config
        filled: list[str] = []
        errors: list[str] = []

        name_parts = profile.profile_name.split() if profile.profile_name else []
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""
        if self._resume_contact is None:
            self._resume_contact = _resume_contact(profile)
        resume_email, resume_phone = self._resume_contact

        field_mappings = [
            (
                (
                    'input[name*="first" i]',
                    'input[id*="first" i]',
                    'input[aria-label*="first" i]',
                ),
                first_name,
                "firstName",
            ),
            (
                (
                    'input[name*="last" i]',
                    'input[id*="last" i]',
                    'input[aria-label*="last" i]',
                ),
                last_name,
                "lastName",
            ),
            (
                (
                    'input[name*="email" i]',
                    'input[id*="email" i]',
                    'input[aria-label*="email" i]',
                ),
                profile.target.linkedin_email or resume_email,
                "email",
            ),
            (
                (
                    'input[name*="phone" i]',
                    'input[id*="phone" i]',
                    'input[aria-label*="phone" i]',
                    'input[aria-label*="mobile" i]',
                    'input[id*="mobile" i]',
                    'input[type="tel"]',
                ),
                resume_phone,
                "phone",
            ),
        ]

        for selectors, value, label in field_mappings:
            if value:
                el = None
                for selector in selectors:
                    el = await page.query_selector(selector)
                    if el:
                        break
                if el:
                    try:
                        # D4 (finding 8b): don't clobber a value the site already pre-filled
                        # (e.g. a session-prefilled email) with a possibly-stale config value.
                        if (await el.input_value()).strip():
                            filled.append(label)
                            continue
                        await el.fill(value)
                        filled.append(label)
                    except Exception as e:
                        # Field is PRESENT but won't fill — surface it (a required one going
                        # unfilled would otherwise reach Submit silently).
                        logger.warning("Field %s present but could not fill: %s", label, e)
                        errors.append(label)
        return filled, errors

    async def check_already_applied(self, job: JobListing) -> bool:
        """Check if already applied to this job."""
        async with self._browser.persistent_page() as page:
            await navigate(page, str(job.url))
            await random_delay(1.0, 2.0)

            applied = await wait_for_selector(page, 'button:has-text("Applied")', timeout_ms=3_000)
            return bool(applied)
