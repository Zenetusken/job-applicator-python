from __future__ import annotations

import pytest

from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedEducationEntry,
    FormattedExperienceEntry,
    FormattedProjectEntry,
    FormattedResume,
    FormattedSkillGroup,
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


def test_resume_rejects_invalid_nested_experience_type() -> None:
    with pytest.raises(ValueError):
        FormattedResume(
            name="Alex Rivera",
            experience=[
                FormattedExperienceEntry(
                    title="Engineer",
                    company="Acme",
                    start_date="2020",
                    bullets="Built things",
                ),
            ],
        )


def test_resume_rejects_unknown_field_in_nested_experience() -> None:
    with pytest.raises(ValueError):
        FormattedResume(
            name="Alex Rivera",
            experience=[
                FormattedExperienceEntry(
                    title="Engineer",
                    company="Acme",
                    start_date="2020",
                    bullets=["Built things"],
                    unknown_field="x",
                ),
            ],
        )


def test_education_entry_valid() -> None:
    education = FormattedEducationEntry(
        institution="State University",
        degree="B.S. Computer Science",
        location="Anytown",
        start_date="2015",
        end_date="2019",
    )
    assert education.institution == "State University"
    assert education.degree == "B.S. Computer Science"


def test_education_entry_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        FormattedEducationEntry(
            institution="State University",
            degree="B.S. Computer Science",
            unknown_field="x",
        )


def test_skill_group_valid() -> None:
    group = FormattedSkillGroup(
        category="Languages",
        skills=["Python", "Rust"],
    )
    assert group.category == "Languages"
    assert group.skills == ["Python", "Rust"]


def test_skill_group_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        FormattedSkillGroup(
            skills=["Python"],
            unknown_field="x",
        )


def test_project_entry_valid() -> None:
    project = FormattedProjectEntry(
        name="Portfolio Site",
        description="Personal portfolio built with Flask",
        url="https://example.com",
    )
    assert project.name == "Portfolio Site"
    assert project.url == "https://example.com"


def test_project_entry_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        FormattedProjectEntry(
            name="Portfolio Site",
            unknown_field="x",
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
