from __future__ import annotations

import pytest

from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedExperienceEntry,
    FormattedResume,
)


def test_resume_valid() -> None:
    resume = FormattedResume(
        name="Alex Rivera",
        email="alex@example.com",
        experience=[
            FormattedExperienceEntry(
                title="Engineer",
                company="Acme",
                start_date="2020",
                end_date="Present",
                bullets=["Built things"],
            ),
        ],
    )
    assert resume.name == "Alex Rivera"


def test_resume_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        FormattedResume(
            name="Alex Rivera",
            email="alex@example.com",
            experience=[],
            unknown_field="x",
        )


def test_resume_rejects_invalid_nested_experience() -> None:
    with pytest.raises(ValueError):
        FormattedResume(
            name="Alex Rivera",
            experience=[
                FormattedExperienceEntry(
                    title="Engineer",
                    company="Acme",
                    start_date="2020",
                    bullets=["Built things"],
                ),
            ],
        )


def test_cover_letter_valid() -> None:
    letter = FormattedCoverLetter(
        recipient_company="Acme",
        date="2026-06-25",
        greeting="Dear Hiring Manager,",
        paragraphs=["I am excited to apply."],
        closing="Sincerely,",
        signature="Alex Rivera",
    )
    assert letter.signature == "Alex Rivera"


def test_cover_letter_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        FormattedCoverLetter(
            recipient_company="Acme",
            date="2026-06-25",
            greeting="Dear Hiring Manager,",
            paragraphs=["I am excited to apply."],
            closing="Sincerely,",
            signature="Alex Rivera",
            unknown_field="x",
        )
