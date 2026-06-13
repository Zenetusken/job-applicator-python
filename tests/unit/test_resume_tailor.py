"""Tests for resume tailoring engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.documents.resume_tailor import (
    CHANGES_PROMPT_TEMPLATE,
    TAILOR_PROMPT_TEMPLATE,
    ResumeTailor,
    parse_sections,
)
from job_applicator.models import (
    JobBoard,
    JobListing,
    ResumeData,
    TailoredResume,
)


@pytest.fixture
def sample_resume():
    return ResumeData(
        raw_text=("ANDREI PETROV\nandre@example.com\nSkills\nWindows, Office 365, Troubleshooting"),
        name="ANDREI PETROV",
        email="andre@example.com",
        skills=["Windows", "Office 365", "Troubleshooting"],
    )


@pytest.fixture
def sample_job():
    return JobListing(
        title="Technical Support Specialist",
        company="CGI",
        url="https://example.com/job",
        description="Provide technical support.",
        requirements=["Windows", "Office 365", "ServiceNow"],
        location="Montreal, QC",
        board=JobBoard.INDEED,
    )


@pytest.fixture
def llm_config():
    from job_applicator.config import LLMConfig

    return LLMConfig(
        api_base="http://localhost:8000/v1",
        model="test-model",
    )


class TestResumeTailor:
    def test_init(self, llm_config):
        tailor = ResumeTailor(llm_config)
        assert tailor._config == llm_config

    def test_prompt_template_formatting(self):
        prompt = TAILOR_PROMPT_TEMPLATE.format(
            job_title="Test Job",
            job_company="Test Co",
            job_location="Remote",
            job_description="Test desc",
            requirements="Skill1, Skill2",
            resume_text="Resume text",
            skills="Skill1, Skill2",
            education_entries="1. Test University, 2020-2024",
            user_instructions="No instructions.",
        )
        assert "Test Job" in prompt
        assert "Test Co" in prompt
        assert "Resume text" in prompt

    def test_changes_prompt_template(self):
        prompt = CHANGES_PROMPT_TEMPLATE.format(
            original_preview="Original text",
            tailored_preview="Tailored text",
        )
        assert "Original text" in prompt
        assert "Tailored text" in prompt

    @pytest.mark.asyncio
    async def test_tailor_returns_result(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "ANDREI PETROV\nandre@example.com\n"
            "Skills: Windows, Office 365, Troubleshooting, ServiceNow\n"
            "Experience: Technical Support..."
        )

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tailor.tailor(sample_resume, sample_job)

        assert isinstance(result, TailoredResume)
        assert result.job_title == "Technical Support Specialist"
        assert result.job_company == "CGI"
        assert result.attempt == 1
        assert len(result.tailored_text) > 0

    @pytest.mark.asyncio
    async def test_refine_increments_attempt(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        initial = TailoredResume(
            original_path="",
            tailored_text="Initial tailored text",
            job_title="Technical Support Specialist",
            job_company="CGI",
            match_score=0.7,
            semantic_score=0.76,
            skill_score=0.6,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            changes_summary="Initial changes",
            attempt=1,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Refined resume text"

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tailor.refine(sample_resume, initial, "Add more detail", sample_job)

        assert result.attempt == 2
        assert result.user_modifications == "Add more detail"


class TestTailoredResumeModel:
    def test_model_creation(self):
        resume = TailoredResume(
            original_path="/path/to/resume.pdf",
            tailored_text="Tailored content",
            job_title="Test Job",
            job_company="Test Co",
            match_score=0.75,
            semantic_score=0.8,
            skill_score=0.65,
            matched_skills=["Python"],
            missing_skills=["AWS"],
            changes_summary="Emphasized Python skills",
        )
        assert resume.attempt == 1
        assert resume.user_modifications == ""
        assert resume.output_path == ""

    def test_model_serialization(self):
        resume = TailoredResume(
            original_path="",
            tailored_text="text",
            job_title="Job",
            job_company="Co",
            match_score=0.5,
            semantic_score=0.5,
            skill_score=0.5,
            changes_summary="changes",
        )
        data = resume.model_dump()
        assert "tailored_text" in data
        assert "match_score" in data
        assert "created_at" in data


class TestParseSections:
    def test_parse_standard_sections(self):
        text = (
            "JOHN DOE\njohn@example.com\n\n"
            "SUMMARY\nExperienced developer.\n\n"
            "EXPERIENCE\nSoftware Engineer at Corp\n2020-2024\n\n"
            "SKILLS\nPython, JavaScript, Docker\n\n"
            "EDUCATION\nBS Computer Science, MIT, 2016-2020\n"
        )
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "SUMMARY" in names
        assert "EXPERIENCE" in names
        assert "SKILLS" in names
        assert "EDUCATION" in names

    def test_parse_mixed_case_headers(self):
        text = "Summary\nSome text.\n\nExperience\nJob stuff.\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "Summary" in names
        assert "Experience" in names

    def test_parse_no_sections_returns_single(self):
        text = "Just a plain resume with no section headers at all."
        sections = parse_sections(text)
        assert len(sections) == 1
        assert sections[0].name == "Full Document"
        assert sections[0].text == text

    def test_section_text_preserved(self):
        text = "SKILLS\nPython, JavaScript\nDocker, Kubernetes\n\nEXPERIENCE\nJob one.\n"
        sections = parse_sections(text)
        skills = next(s for s in sections if s.name == "SKILLS")
        assert "Python" in skills.text
        assert "Docker" in skills.text

    def test_header_with_colon(self):
        text = "Technical Skills:\nPython, Java\n\nWork Experience:\nJob stuff.\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "Technical Skills:" in names
        assert "Work Experience:" in names
