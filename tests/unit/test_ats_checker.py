"""Tests for ATS compatibility checker."""

from __future__ import annotations

import pytest

from job_applicator.documents.ats_checker import ATSChecker
from job_applicator.models import ATSCompatibilityResult, ResumeData


@pytest.fixture
def checker() -> ATSChecker:
    return ATSChecker()


@pytest.fixture
def good_resume() -> ResumeData:
    return ResumeData(
        raw_text=(
            "John Doe\n"
            "john@example.com\n"
            "555-123-4567\n"
            "Summary\n"
            "Experienced Python developer with 5 years of experience.\n"
            "Experience\n"
            "Senior Developer at TechCorp (2020 - Present)\n"
            "- Built REST APIs using FastAPI\n"
            "- Led team of 3 developers\n"
            "Junior Developer at StartupInc (2018 - 2020)\n"
            "- Developed web applications\n"
            "Education\n"
            "BS Computer Science, State University (2014 - 2018)\n"
            "Skills\n"
            "Python, FastAPI, PostgreSQL, Docker, AWS\n"
        ),
        name="John Doe",
        email="john@example.com",
        phone="555-123-4567",
        summary="Experienced Python developer with 5 years of experience.",
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
    )


@pytest.fixture
def bad_resume() -> ResumeData:
    return ResumeData(
        raw_text="me\nstuff I did\nsome skills",
        name="me",
        email="",
        phone="",
        summary="",
        skills=[],
    )


class TestATSCheckerInterface:
    def test_check_returns_result(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        assert isinstance(result, ATSCompatibilityResult)

    def test_check_has_score(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        assert 0.0 <= result.score <= 1.0

    def test_check_has_checks_list(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        assert len(result.checks) > 0
        for check in result.checks:
            assert "name" in check
            assert "passed" in check
            assert "details" in check


class TestContactInfoChecks:
    def test_email_present(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        email_check = next(c for c in result.checks if c["name"] == "email_present")
        assert email_check["passed"] is True

    def test_email_missing(self, checker: ATSChecker, bad_resume: ResumeData) -> None:
        result = checker.check(bad_resume)
        email_check = next(c for c in result.checks if c["name"] == "email_present")
        assert email_check["passed"] is False
        assert any("email" in w.lower() for w in result.warnings)

    def test_phone_present(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        phone_check = next(c for c in result.checks if c["name"] == "phone_present")
        assert phone_check["passed"] is True

    def test_phone_missing(self, checker: ATSChecker, bad_resume: ResumeData) -> None:
        result = checker.check(bad_resume)
        phone_check = next(c for c in result.checks if c["name"] == "phone_present")
        assert phone_check["passed"] is False


class TestSectionHeaderChecks:
    def test_experience_section(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        exp_check = next(c for c in result.checks if c["name"] == "experience_section")
        assert exp_check["passed"] is True

    def test_education_section(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        edu_check = next(c for c in result.checks if c["name"] == "education_section")
        assert edu_check["passed"] is True

    def test_skills_section(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        skills_check = next(c for c in result.checks if c["name"] == "skills_section")
        assert skills_check["passed"] is True

    def test_missing_sections_detected(self, checker: ATSChecker, bad_resume: ResumeData) -> None:
        result = checker.check(bad_resume)
        exp_check = next(c for c in result.checks if c["name"] == "experience_section")
        assert exp_check["passed"] is False
        assert any("experience" in w.lower() for w in result.warnings)


class TestLengthChecks:
    def test_reasonable_length(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        length_check = next(c for c in result.checks if c["name"] == "text_length")
        assert length_check["passed"] is True

    def test_too_short_detected(self, checker: ATSChecker, bad_resume: ResumeData) -> None:
        result = checker.check(bad_resume)
        length_check = next(c for c in result.checks if c["name"] == "text_length")
        assert length_check["passed"] is False


class TestFormatChecks:
    def test_no_tables_detected(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        table_check = next(c for c in result.checks if c["name"] == "no_tables")
        assert table_check["passed"] is True

    def test_tables_detected(self, checker: ATSChecker) -> None:
        table_text = (
            "John Doe\njohn@example.com\n555-123-4567\n+----+----+\n| a  | b  |\n+----+----+"
        )
        resume = ResumeData(
            raw_text=table_text,
            name="John Doe",
            email="john@example.com",
            phone="555-123-4567",
        )
        result = checker.check(resume)
        table_check = next(c for c in result.checks if c["name"] == "no_tables")
        assert table_check["passed"] is False


class TestScoring:
    def test_perfect_resume_scores_high(self, checker: ATSChecker, good_resume: ResumeData) -> None:
        result = checker.check(good_resume)
        assert result.score >= 0.8
        assert result.is_compatible is True

    def test_bad_resume_scores_low(self, checker: ATSChecker, bad_resume: ResumeData) -> None:
        result = checker.check(bad_resume)
        assert result.score < 0.5
        assert result.is_compatible is False

    def test_suggestions_provided_for_low_score(
        self, checker: ATSChecker, bad_resume: ResumeData
    ) -> None:
        result = checker.check(bad_resume)
        assert len(result.suggestions) > 0


class TestATSPostTailor:
    def test_post_tailor_shows_improvement(self) -> None:
        from job_applicator.cli import _run_ats_post_tailor

        original_text = "Bob\nbob@email.com\nstuff"
        tailored_text = (
            "Bob\nbob@email.com\n555-123-4567\n"
            "Summary\nExperienced developer.\n"
            "Experience\nSenior Dev at Corp (2020-Present)\n"
            "Education\nBS CS (2016-2020)\n"
            "Skills\nPython, Java"
        )
        _run_ats_post_tailor(original_text, tailored_text)

    def test_post_tailor_detects_regression(self) -> None:
        from job_applicator.cli import _run_ats_post_tailor

        original_text = (
            "John\njohn@example.com\n555-123-4567\n"
            "Experience\nSenior Dev (2020-Present)\n"
            "Education\nBS CS (2016-2020)\n"
            "Skills\nPython"
        )
        tailored_text = "John\njohn@example.com\nTailored summary without sections."
        _run_ats_post_tailor(original_text, tailored_text)


class TestATSPreflight:
    def test_preflight_warns_when_incompatible(self) -> None:
        from job_applicator.cli import _run_ats_preflight

        resume = ResumeData(
            raw_text="Bob\nbob@email.com\nstuff",
            name="Bob",
            email="bob@email.com",
            phone="",
        )
        _run_ats_preflight(resume)

    def test_preflight_silent_when_compatible(self) -> None:
        from job_applicator.cli import _run_ats_preflight

        resume = ResumeData(
            raw_text=(
                "John Doe\n"
                "john@example.com\n"
                "555-123-4567\n"
                "Summary\n"
                "Experienced developer.\n"
                "Experience\n"
                "Senior Dev at Corp (2020-Present)\n"
                "- Built stuff\n"
                "Education\n"
                "BS CS (2016-2020)\n"
                "Skills\n"
                "Python, Java"
            ),
            name="John Doe",
            email="john@example.com",
            phone="555-123-4567",
            skills=["Python", "Java"],
        )
        _run_ats_preflight(resume)
