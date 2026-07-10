"""Tests for generic source-aware packet integrity checks."""

from __future__ import annotations

from job_applicator.documents.source_integrity import assess_source_integrity
from job_applicator.models import EducationEntry, ExperienceEntry, ResumeData


def _source() -> ResumeData:
    return ResumeData(
        raw_text=(
            "Alex Morgan\n"
            "alex@example.com | 438-555-0100\n"
            "Support Analyst, Acme Support, 2020 - 2022\n"
            "Resolved customer tickets by phone and email.\n"
            "Diploma, Information Technology, Metro College, 2018 - 2020\n"
        ),
        name="Alex Morgan",
        email="alex@example.com",
        phone="438-555-0100",
        experience=[
            ExperienceEntry(
                title="Support Analyst",
                company="Acme Support",
                start_date="2020",
                end_date="2022",
            )
        ],
        education=[
            EducationEntry(
                institution="Metro College",
                degree="Diploma, Information Technology",
                start_date="2018",
                end_date="2020",
            )
        ],
    )


def _generated_resume() -> str:
    return (
        "Alex Morgan\n"
        "alex@example.com | 438-555-0100\n\n"
        "Experience\nSupport Analyst, Acme Support, 2020 - 2022\n\n"
        "Education\nDiploma, Information Technology, Metro College, 2018 - 2020"
    )


def test_source_integrity_accepts_preserved_source_structure() -> None:
    report = assess_source_integrity(
        source=_source(),
        generated_resume=_generated_resume(),
        generated_cover="I resolved customer tickets by phone and email.",
    )

    assert report.source_checked
    assert report.failures == []


def test_source_integrity_allows_localized_titles_when_entities_are_preserved() -> None:
    localized = (
        _generated_resume()
        .replace("Support Analyst", "Analyste du soutien")
        .replace(
            "Diploma, Information Technology",
            "Diplôme en technologies de l'information",
        )
    )

    report = assess_source_integrity(
        source=_source(),
        generated_resume=localized,
        generated_cover="",
    )

    assert report.failures == []


def test_source_integrity_rejects_missing_contact_employer_and_school() -> None:
    report = assess_source_integrity(
        source=_source(),
        generated_resume="Alex Morgan\nExperience\nDifferent Employer",
        generated_cover="",
    )

    assert "email" in report.missing_contact_fields
    assert "phone" in report.missing_contact_fields
    assert report.missing_experience_companies == ["Acme Support"]
    assert report.missing_education_institutions == ["Metro College"]


def test_source_integrity_rejects_new_metric() -> None:
    report = assess_source_integrity(
        source=_source(),
        generated_resume=_generated_resume(),
        generated_cover="Certified analyst who resolved 40+ tickets daily.",
    )

    assert report.unsupported_numeric_claims == ["cover_letter: 40+ tickets"]


def test_cover_only_integrity_skips_resume_structure_requirements() -> None:
    report = assess_source_integrity(
        source=_source(),
        generated_resume="",
        generated_cover="I resolved customer tickets by phone and email.",
        require_resume_structure=False,
    )

    assert report.failures == []
    assert report.missing_contact_fields == []


def test_source_integrity_does_not_treat_product_number_as_ticket_metric() -> None:
    report = assess_source_integrity(
        source=_source(),
        generated_resume=_generated_resume() + "\nMicrosoft 365, ticketing, and escalation",
        generated_cover="",
    )

    assert report.unsupported_numeric_claims == []
