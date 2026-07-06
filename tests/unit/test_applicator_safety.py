"""Safety tests for the LinkedIn Easy Apply dry-run gate.

The critical guarantee: an automated `apply` run must NOT submit a real
application unless the caller explicitly opts in with submit=True.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from job_applicator.applicators.linkedin import LinkedInApplicator
from job_applicator.config import AppSettings
from job_applicator.models import ApplicationStatus, JobBoard, JobListing


def _job() -> JobListing:
    return JobListing(
        title="X", company="Y", url="https://www.linkedin.com/jobs/1", board=JobBoard.LINKEDIN
    )


def _page_reaching_submit(submit_btn: AsyncMock) -> AsyncMock:
    """A page whose only matched selector is the final 'Submit application' button."""
    page = AsyncMock()

    async def query_selector(selector: str) -> object | None:
        return submit_btn if "Submit application" in selector else None

    page.query_selector = query_selector
    return page


def _page_with_cover(cl_field: AsyncMock, submit_btn: AsyncMock) -> AsyncMock:
    """A page exposing only the cover-letter textarea and the final Submit button."""
    page = AsyncMock()

    async def query_selector(selector: str) -> object | None:
        if "cover" in selector:
            return cl_field
        if "Submit application" in selector:
            return submit_btn
        return None

    page.query_selector = query_selector
    return page


def _read_first_apply_dump(path: Path) -> str:
    dump = next(path.glob("linkedin-apply-*.txt"))
    return dump.read_text(encoding="utf-8")


class _ChooserContext:
    def __init__(self, chooser: AsyncMock) -> None:
        async def _value() -> AsyncMock:
            return chooser

        self.value = _value()

    async def __aenter__(self) -> _ChooserContext:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


@pytest.mark.asyncio
async def test_easy_apply_dry_run_does_not_submit(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    submit_btn = AsyncMock()
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(
        _page_reaching_submit(submit_btn), _job(), None, submit=False
    )

    assert result.status == ApplicationStatus.SKIPPED
    assert "DRY RUN" in result.notes
    submit_btn.click.assert_not_awaited()  # the critical guarantee — nothing submitted


@pytest.mark.asyncio
async def test_easy_apply_submits_only_with_opt_in(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    monkeypatch.setattr(
        "job_applicator.applicators.linkedin.wait_for_selector", AsyncMock(return_value=True)
    )
    submit_btn = AsyncMock()
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(
        _page_reaching_submit(submit_btn), _job(), None, submit=True
    )

    assert result.status == ApplicationStatus.SUBMITTED
    submit_btn.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_easy_apply_dry_run_reports_validation_details(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run captures whether the form reached the Submit button."""
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    submit_btn = AsyncMock()
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(
        _page_reaching_submit(submit_btn), _job(), None, submit=False
    )

    assert result.dry_run is not None
    assert result.dry_run.reached_submit is True
    assert result.dry_run.easy_apply_button_found is True


@pytest.mark.asyncio
async def test_easy_apply_missing_submit_button_reports_failed_validation(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the Submit button is not found, dry_run.reached_submit is False."""
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())

    page = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(page, _job(), None, submit=False)

    assert result.status == ApplicationStatus.FAILED
    assert result.dry_run is not None
    assert result.dry_run.reached_submit is False


@pytest.mark.asyncio
async def test_easy_apply_advances_continue_variant_to_submit(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LinkedIn variants label the first advance button Continue instead of Next."""
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    advance = AsyncMock()
    submit_btn = AsyncMock()
    advanced = False
    page = AsyncMock()

    async def query_selector(selector: str) -> object | None:
        nonlocal advanced
        if "Continue" in selector and not advanced:
            advanced = True
            return advance
        if "Submit" in selector and advanced:
            return submit_btn
        return None

    page.query_selector = query_selector
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(page, _job(), None, submit=False)

    advance.click.assert_awaited_once()
    assert result.dry_run is not None
    assert result.dry_run.reached_submit is True


@pytest.mark.asyncio
async def test_easy_apply_uploads_resume_via_upload_button(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LinkedIn may expose an Upload resume button instead of a visible file input."""
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    resume = tmp_path / "resume.pdf"
    resume.write_text("%PDF")
    app_settings.resume_path = str(resume)
    upload_button = AsyncMock()
    chooser = AsyncMock()
    submit_btn = AsyncMock()
    page = AsyncMock()
    page.expect_file_chooser = MagicMock(return_value=_ChooserContext(chooser))

    async def query_selector(selector: str) -> object | None:
        if "Upload resume" in selector:
            return upload_button
        if "Submit" in selector:
            return submit_btn
        return None

    page.query_selector = query_selector
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(page, _job(), None, submit=False)

    chooser.set_files.assert_awaited_once_with(str(resume))
    assert result.dry_run is not None
    assert result.dry_run.resume_uploaded is True


@pytest.mark.asyncio
async def test_easy_apply_missing_submit_writes_debug_dump(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live selector miss should leave button diagnostics for follow-up."""
    import job_applicator.applicators.linkedin as li

    monkeypatch.setattr(li, "_DEBUG_DIR", tmp_path)
    monkeypatch.setattr(li, "click", AsyncMock())
    monkeypatch.setattr(li, "random_delay", AsyncMock())
    button = AsyncMock()
    button.inner_text = AsyncMock(return_value="Continue to next step")
    button.get_attribute = AsyncMock(return_value="Continue to next step")
    page = AsyncMock()
    page.url = "https://www.linkedin.com/jobs/view/1"
    page.query_selector = AsyncMock(return_value=None)
    page.query_selector_all = AsyncMock(return_value=[button])
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(page, _job(), None, submit=False)

    assert result.status == ApplicationStatus.FAILED
    assert "debug saved" in (result.error_message or "")
    assert "Continue to next step" in _read_first_apply_dump(tmp_path)


@pytest.mark.asyncio
async def test_easy_apply_cover_letter_is_focused_then_pasted(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cover letter is pasted human-like: the textarea is focused (click) BEFORE the fill,
    so the sequence reads as a deliberate paste rather than a value appearing on its own."""
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    cl_field = AsyncMock()
    letter = "Dear hiring manager, I would be glad to apply."
    page = _page_with_cover(cl_field, AsyncMock())
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    await applicator._easy_apply(page, _job(), letter, submit=False)

    cl_field.click.assert_awaited_once()
    cl_field.fill.assert_awaited_once_with(letter)
    names = [c[0] for c in cl_field.mock_calls]
    assert names.index("click") < names.index("fill")  # focus precedes the paste


@pytest.mark.asyncio
async def test_easy_apply_cover_letter_focus_click_failure_still_fills(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If focusing the cover-letter textarea fails (obscured/animating — click imposes
    actionability that fill does not), the apply must NOT abort: it falls through to the plain
    fill (exactly the pre-paste behaviour); a present-but-unclickable textarea filled fine
    before."""
    monkeypatch.setattr("job_applicator.applicators.linkedin.click", AsyncMock())
    monkeypatch.setattr("job_applicator.applicators.linkedin.random_delay", AsyncMock())
    cl_field = AsyncMock()
    cl_field.click = AsyncMock(side_effect=RuntimeError("textarea obscured by overlay"))
    letter = "Dear hiring manager, I would be glad to apply."
    page = _page_with_cover(cl_field, AsyncMock())
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    result = await applicator._easy_apply(page, _job(), letter, submit=False)

    cl_field.fill.assert_awaited_once_with(letter)  # still filled despite the click failure
    assert result.status == ApplicationStatus.SKIPPED  # dry run completed, not aborted to FAILED
    assert result.dry_run is not None
    assert result.dry_run.cover_letter_field_found is True


@pytest.mark.asyncio
async def test_fill_form_fields_skips_already_populated_field(app_settings: AppSettings) -> None:
    """D4 (finding 8b): a field the site already pre-filled (e.g. a session-prefilled email) is NOT
    clobbered with a possibly-stale config value — it's left as-is and reported as filled."""
    app_settings.profile_name = "Jane Doe"
    app_settings.target.linkedin_email = "config@example.com"
    applicator = LinkedInApplicator(MagicMock(), app_settings)

    el = AsyncMock()
    el.input_value = AsyncMock(return_value="prefilled@site.com")  # site already populated it

    async def query_selector(selector: str) -> object | None:
        return el if "email" in selector else None  # only the email field is present

    page = AsyncMock()
    page.query_selector = query_selector

    filled, _errors = await applicator._fill_form_fields(page)

    assert "email" in filled
    el.fill.assert_not_awaited()  # NOT overwritten with the config value


@pytest.mark.asyncio
async def test_fill_form_fields_uses_resume_phone_for_mobile_aria_field(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LinkedIn phone inputs often expose only aria/id labels, not name*=phone."""
    import job_applicator.applicators.linkedin as li

    monkeypatch.setattr(
        li, "_resume_contact", lambda _settings: ("resume@example.com", "5145550199")
    )
    applicator = LinkedInApplicator(MagicMock(), app_settings)
    phone = AsyncMock()
    phone.input_value = AsyncMock(return_value="")

    async def query_selector(selector: str) -> object | None:
        return phone if "mobile" in selector else None

    page = AsyncMock()
    page.query_selector = query_selector

    filled, errors = await applicator._fill_form_fields(page)

    assert errors == []
    assert "phone" in filled
    phone.fill.assert_awaited_once_with("5145550199")


@pytest.mark.asyncio
async def test_fill_form_fields_uses_resume_phone_for_unlabeled_tel_input(
    app_settings: AppSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live LinkedIn forms may expose only ``input[type=tel]`` with generated ids."""
    import job_applicator.applicators.linkedin as li

    monkeypatch.setattr(li, "_resume_contact", lambda _settings: ("", "5145550199"))
    applicator = LinkedInApplicator(MagicMock(), app_settings)
    phone = AsyncMock()
    phone.input_value = AsyncMock(return_value="")

    async def query_selector(selector: str) -> object | None:
        return phone if 'type="tel"' in selector else None

    page = AsyncMock()
    page.query_selector = query_selector

    filled, errors = await applicator._fill_form_fields(page)

    assert errors == []
    assert "phone" in filled
    phone.fill.assert_awaited_once_with("5145550199")


def test_validated_resume_upload_path_checks_existence_and_type(
    app_settings: AppSettings, tmp_path: Path
) -> None:
    """The resume is validated (exists + a LinkedIn-supported type) before upload, so a missing /
    wrong-type file fails with a clean typed error, not an opaque Playwright failure mid-apply."""
    from job_applicator.applicators.linkedin import _validated_resume_upload_path
    from job_applicator.exceptions import FormFillingError, ResumeNotFoundError

    app_settings.resume_path = str(tmp_path / "missing.pdf")
    with pytest.raises(ResumeNotFoundError):
        _validated_resume_upload_path(app_settings)

    txt = tmp_path / "resume.txt"
    txt.write_text("plain text resume")
    app_settings.resume_path = str(txt)
    with pytest.raises(FormFillingError):  # exists but unsupported type
        _validated_resume_upload_path(app_settings)

    pdf = tmp_path / "resume.pdf"
    pdf.write_text("%PDF-fake")
    app_settings.resume_path = str(pdf)
    assert _validated_resume_upload_path(app_settings) == pdf  # exists + supported
