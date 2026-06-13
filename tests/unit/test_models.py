"""Unit tests for models."""

from __future__ import annotations

from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    JobBoard,
    JobListing,
    ResumeData,
    UserProfile,
)


def test_job_listing_creation() -> None:
    job = JobListing(
        title="Python Dev",
        company="Acme",
        url="https://example.com/job/1",
        board=JobBoard.LINKEDIN,
    )
    assert job.title == "Python Dev"
    assert job.company == "Acme"
    assert job.board == JobBoard.LINKEDIN


def test_application_result_status() -> None:
    job = JobListing(
        title="Dev",
        company="Co",
        url="https://example.com/1",
        board=JobBoard.INDEED,
    )
    result = ApplicationResult(job=job, status=ApplicationStatus.SUBMITTED)
    assert result.status == ApplicationStatus.SUBMITTED
    assert result.job.title == "Dev"


def test_user_profile() -> None:
    user = UserProfile(
        first_name="Jane",
        last_name="Smith",
        email="jane@example.com",
        phone="555-0199",
    )
    assert user.first_name == "Jane"
    assert user.email == "jane@example.com"


def test_resume_data_defaults() -> None:
    resume = ResumeData(raw_text="test content")
    assert resume.skills == []
    assert resume.experience == []
    assert resume.education == []


def test_job_listing_extra_forbid() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JobListing(
            title="Dev",
            company="Co",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
            unknown_field="should fail",
        )
