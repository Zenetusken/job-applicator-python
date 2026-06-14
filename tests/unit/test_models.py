"""Unit tests for models."""

from __future__ import annotations

import pytest

from job_applicator.models import (
    ApplicationResult,
    ApplicationStatus,
    CoverLetterResult,
    CoverLetterSession,
    JobBoard,
    JobListing,
    ResumeData,
    TailoredResume,
    TailorSession,
    UserProfile,
    detect_seniority,
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


class TestTailorSession:
    def test_session_creation(self) -> None:
        session = TailorSession(
            original_text="Original resume text",
            job_title="Developer",
            job_company="TechCo",
        )
        assert session.attempts == []
        assert session.current_index == -1

    def test_add_attempt(self) -> None:
        session = TailorSession(
            original_text="Original",
            job_title="Dev",
            job_company="Co",
        )
        result = TailoredResume(
            original_path="",
            tailored_text="Tailored v1",
            job_title="Dev",
            job_company="Co",
            match_score=0.7,
            semantic_score=0.7,
            skill_score=0.7,
            changes_summary="changes",
            attempt=1,
        )
        session.add_attempt(result)
        assert len(session.attempts) == 1
        assert session.current_index == 0

    def test_current_property(self) -> None:
        session = TailorSession(
            original_text="Original",
            job_title="Dev",
            job_company="Co",
        )
        result = TailoredResume(
            original_path="",
            tailored_text="Tailored v1",
            job_title="Dev",
            job_company="Co",
            match_score=0.7,
            semantic_score=0.7,
            skill_score=0.7,
            changes_summary="changes",
            attempt=1,
        )
        session.add_attempt(result)
        assert session.current.tailored_text == "Tailored v1"

    def test_current_empty_session_raises(self) -> None:
        session = TailorSession(
            original_text="Original",
            job_title="Dev",
            job_company="Co",
        )
        with pytest.raises(IndexError):
            _ = session.current

    def test_select_attempt(self) -> None:
        session = TailorSession(
            original_text="Original",
            job_title="Dev",
            job_company="Co",
        )
        for i in range(3):
            session.add_attempt(
                TailoredResume(
                    original_path="",
                    tailored_text=f"Version {i}",
                    job_title="Dev",
                    job_company="Co",
                    match_score=0.5 + i * 0.1,
                    semantic_score=0.5,
                    skill_score=0.5,
                    changes_summary="changes",
                    attempt=i + 1,
                )
            )
        session.select(1)
        assert session.current.tailored_text == "Version 1"
        assert session.current_index == 1


class TestCoverLetterResult:
    def test_model_creation(self) -> None:
        result = CoverLetterResult(
            job_title="Developer",
            job_company="TechCo",
            cover_letter_text="Dear Hiring Manager...",
        )
        assert result.attempt == 1
        assert result.user_modifications == ""
        assert result.output_path == ""

    def test_model_serialization(self) -> None:
        result = CoverLetterResult(
            job_title="Dev",
            job_company="Co",
            cover_letter_text="Letter text",
        )
        data = result.model_dump()
        assert "cover_letter_text" in data
        assert "created_at" in data


class TestCoverLetterSession:
    def test_session_creation(self) -> None:
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        assert session.attempts == []
        assert session.current_index == -1

    def test_add_attempt(self) -> None:
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        result = CoverLetterResult(
            job_title="Dev",
            job_company="Co",
            cover_letter_text="Letter v1",
        )
        session.add_attempt(result)
        assert len(session.attempts) == 1
        assert session.current.cover_letter_text == "Letter v1"

    def test_current_empty_raises(self) -> None:
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        with pytest.raises(IndexError):
            _ = session.current

    def test_select_attempt(self) -> None:
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        for i in range(3):
            session.add_attempt(
                CoverLetterResult(
                    job_title="Dev",
                    job_company="Co",
                    cover_letter_text=f"Version {i}",
                    attempt=i + 1,
                )
            )
        session.select(1)
        assert session.current.cover_letter_text == "Version 1"
        assert session.current_index == 1

    def test_select_out_of_range(self) -> None:
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        with pytest.raises(IndexError):
            session.select(99)


class TestDetectSeniority:
    """Tests for seniority detection from job titles."""

    def test_senior(self) -> None:
        assert detect_seniority("Senior Python Developer") == "senior"

    def test_junior(self) -> None:
        assert detect_seniority("Junior Software Engineer") == "junior"

    def test_lead(self) -> None:
        assert detect_seniority("Lead Backend Engineer") == "lead"

    def test_principal(self) -> None:
        assert detect_seniority("Principal Architect") == "principal"

    def test_staff(self) -> None:
        assert detect_seniority("Staff Engineer") == "staff"

    def test_intern(self) -> None:
        assert detect_seniority("Software Engineering Intern") == "intern"

    def test_director(self) -> None:
        assert detect_seniority("Director of Engineering") == "director"

    def test_no_seniority(self) -> None:
        assert detect_seniority("Software Engineer") is None

    def test_entry_level(self) -> None:
        assert detect_seniority("Entry Level Developer") == "junior"

    def test_sr_abbreviation(self) -> None:
        assert detect_seniority("Sr. DevOps Engineer") == "senior"

    def test_word_boundary(self) -> None:
        """Ensure 'senior' doesn't match 'seniority'."""
        assert detect_seniority("Questions about seniority") is None


class TestPromptVersion:
    """Tests for prompt_version field on models."""

    def test_tailored_resume_default_version(self) -> None:
        result = TailoredResume(
            original_path="",
            tailored_text="text",
            job_title="Dev",
            job_company="Co",
            match_score=0.8,
            semantic_score=0.5,
            skill_score=0.3,
            changes_summary="changes",
        )
        assert result.prompt_version == "1.0"

    def test_tailored_resume_custom_version(self) -> None:
        result = TailoredResume(
            original_path="",
            tailored_text="text",
            job_title="Dev",
            job_company="Co",
            match_score=0.8,
            semantic_score=0.5,
            skill_score=0.3,
            changes_summary="changes",
            prompt_version="2.1",
        )
        assert result.prompt_version == "2.1"

    def test_cover_letter_default_version(self) -> None:
        result = CoverLetterResult(
            job_title="Dev",
            job_company="Co",
            cover_letter_text="letter",
        )
        assert result.prompt_version == "1.0"

    def test_cover_letter_custom_version(self) -> None:
        result = CoverLetterResult(
            job_title="Dev",
            job_company="Co",
            cover_letter_text="letter",
            prompt_version="2.0",
        )
        assert result.prompt_version == "2.0"


class TestJobListingSeniorityField:
    """Tests for seniority field on JobListing."""

    def test_default_none(self) -> None:
        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
        )
        assert job.seniority is None

    def test_set_seniority(self) -> None:
        job = JobListing(
            title="Senior Dev",
            company="Co",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
            seniority="senior",
        )
        assert job.seniority == "senior"


class TestScoreFields:
    """Tests for semantic_score and skill_score on TailoredResume."""

    def test_scores_can_be_nonzero(self) -> None:
        result = TailoredResume(
            original_path="",
            tailored_text="text",
            job_title="Dev",
            job_company="Co",
            match_score=0.8,
            semantic_score=0.6,
            skill_score=0.4,
            changes_summary="changes",
        )
        assert result.semantic_score == 0.6
        assert result.skill_score == 0.4
        assert result.match_score == pytest.approx(0.8)
