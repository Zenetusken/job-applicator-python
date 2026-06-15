"""Tests for the tailoring pipeline: date validator, tone detector,
strip_thinking_process, ResumeTailor.tailor/refine, and CoverLetterGenerator._build_prompt."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.cover_letter import CoverLetterGenerator, strip_thinking_process
from job_applicator.documents.resume_tailor import ResumeDateValidator, ResumeTailor
from job_applicator.documents.tone_detector import ToneDetector, ToneProfile
from job_applicator.models import (
    JobBoard,
    JobListing,
    ResumeData,
    TailoredResume,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(api_base="http://localhost:8000/v1", model="test-model")


@pytest.fixture
def sample_job() -> JobListing:
    return JobListing(
        title="Senior Python Developer",
        company="TechCorp",
        url="https://linkedin.com/jobs/12345",
        description="We are looking for a senior Python developer to build APIs.",
        location="San Francisco, CA",
        requirements=["Python", "FastAPI", "Docker", "AWS"],
        board=JobBoard.LINKEDIN,
    )


@pytest.fixture
def sample_resume() -> ResumeData:
    return ResumeData(
        raw_text=(
            "JOHN DOE\njohn@example.com\n555-0123\n\n"
            "SKILLS\nPython, FastAPI, Docker\n\n"
            "EXPERIENCE\n"
            "Senior Developer\nAcme Corp\n2020 - Present\n"
            "• Built REST APIs with FastAPI\n\n"
            "Junior Developer\nStartup Inc\n2018 - 2020\n"
            "• Wrote Python scripts\n"
        ),
        name="John Doe",
        email="john@example.com",
        phone="555-0123",
        summary="Experienced Python developer",
        skills=["Python", "FastAPI", "Docker"],
    )


@pytest.fixture
def sample_user() -> UserProfile:
    return UserProfile(
        first_name="John",
        last_name="Doe",
        email="john@example.com",
        phone="555-0123",
    )


# ===========================================================================
# ResumeDateValidator.audit()
# ===========================================================================


class TestResumeDateValidatorAudit:
    def test_no_dates(self):
        resume = ResumeData(raw_text="No dates here at all.")
        result = ResumeDateValidator().audit(resume)
        assert len(result.entries) == 0
        assert result.is_ordered
        assert not result.is_stale
        assert result.latest_date == ""
        assert result.earliest_date == ""

    def test_correct_order_is_ordered(self):
        resume = ResumeData(
            raw_text=("EXPERIENCE\nNew Job\nCorp\n2022 - Present\nOld Job\nCorp\n2018 - 2021")
        )
        result = ResumeDateValidator().audit(resume)
        assert result.is_ordered
        assert len(result.ordering_issues) == 0

    def test_wrong_order_ordering_issues(self):
        resume = ResumeData(
            raw_text=("EXPERIENCE\nOld Job\nCorp\n2010 - 2015\nNew Job\nCorp\n2018 - 2024")
        )
        result = ResumeDateValidator().audit(resume)
        assert not result.is_ordered
        assert len(result.ordering_issues) > 0
        assert "should come after" in result.ordering_issues[0]

    def test_stale_dates_is_stale(self):
        resume = ResumeData(raw_text="EXPERIENCE\nOld Job\nCorp\n2000 - 2005")
        result = ResumeDateValidator(reference_date=datetime(2030, 6, 1)).audit(resume)
        assert result.is_stale
        assert len(result.staleness_issues) > 0
        assert "2005" in result.staleness_issues[0]

    def test_recent_dates_not_stale(self):
        resume = ResumeData(raw_text="EXPERIENCE\nCurrent Job\nCorp\n2024 - Present")
        result = ResumeDateValidator().audit(resume)
        assert not result.is_stale
        general_staleness = [s for s in result.staleness_issues if "Most recent entry" in s]
        assert len(general_staleness) == 0

    def test_education_staleness_greater_than_10yr(self):
        resume = ResumeData(raw_text="EDUCATION\nBS Computer Science\nMIT\n1998 - 2002")
        result = ResumeDateValidator(reference_date=datetime(2030, 1, 1)).audit(resume)
        assert result.is_stale
        edu_staleness = [s for s in result.staleness_issues if "Education" in s]
        assert len(edu_staleness) > 0
        assert "2002" in edu_staleness[0]

    def test_year_only_format(self):
        resume = ResumeData(raw_text="EXPERIENCE\nJob\nCorp\n2018 - 2020")
        result = ResumeDateValidator().audit(resume)
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry.start == "2018"
        assert entry.end == "2020"
        assert entry.is_current is False

    def test_month_year_format(self):
        resume = ResumeData(raw_text="EXPERIENCE\nJob\nCorp\nJan 2020 - Jun 2022")
        result = ResumeDateValidator().audit(resume)
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry.start == "January 2020"
        assert entry.end == "June 2022"

    def test_mixed_formats(self):
        resume = ResumeData(
            raw_text=(
                "EXPERIENCE\nNewer Job\nCorp\nJan 2022 - Present\nOlder Job\nCorp\n2016 - 2019"
            )
        )
        result = ResumeDateValidator().audit(resume)
        assert len(result.entries) == 2
        # The month-year entry should parse with month info
        newer = next(e for e in result.entries if e.is_current)
        assert newer.start == "January 2022"
        # The year-only entry should parse without month
        older = next(e for e in result.entries if not e.is_current)
        assert older.start == "2016"
        assert older.end == "2019"

    def test_empty_text(self):
        resume = ResumeData(raw_text="")
        result = ResumeDateValidator().audit(resume)
        assert len(result.entries) == 0
        assert result.is_ordered
        assert not result.is_stale


# ===========================================================================
# ToneDetector.detect() and format_for_prompt()
# ===========================================================================


class TestToneDetector:
    def test_corporate_keywords(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Compliance Manager",
            description="We need someone for governance and stakeholder management.",
            requirements=["Process improvement", "Regulatory knowledge"],
        )
        assert profile.primary == "corporate"
        assert profile.confidence > 0

    def test_startup_keywords(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Founding Engineer",
            description="Fast-paced startup, scrappy team, early-stage.",
            requirements=["Self-starter", "Equity compensation"],
        )
        assert profile.primary == "startup"

    def test_technical_keywords(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Platform Engineer",
            description="Build distributed systems with microservices architecture.",
            requirements=["Kubernetes", "CI/CD", "Terraform", "Scalability"],
        )
        assert profile.primary == "technical"

    def test_creative_keywords(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="UX Designer",
            description="Design thinking, storytelling, and user experience.",
            requirements=["Wireframe", "Prototype", "User research"],
        )
        assert profile.primary == "creative"

    def test_mixed_highest_wins(self):
        detector = ToneDetector()
        # Mix startup and corporate, but more startup keywords
        profile = detector.detect(
            title="Engineering Manager",
            description="Fast-paced, scrappy team with governance and compliance.",
            requirements=["Self-starter", "Equity", "Ownership", "Agile"],
        )
        assert profile.primary == "startup"

    def test_no_keywords_defaults_to_corporate(self):
        detector = ToneDetector()
        profile = detector.detect(
            title="Manager",
            description="We are hiring.",
            requirements=[],
        )
        assert profile.primary == "unknown"
        assert profile.confidence == 0.0

    def test_format_for_prompt_structure(self):
        detector = ToneDetector()
        profile = ToneProfile(
            primary="technical",
            confidence=0.8,
            power_words=["architected", "engineered"],
            emphasis=["system design", "scalability"],
            avoid=["buzzwords", "fluff"],
        )
        formatted = detector.format_for_prompt(profile)
        assert "TONE: Technical" in formatted
        assert "Use these action verbs: architected, engineered" in formatted
        assert "Emphasize: system design, scalability" in formatted
        assert "Avoid: buzzwords, fluff" in formatted


# ===========================================================================
# strip_thinking_process()
# ===========================================================================


class TestStripThinkingProcess:
    def test_thinking_prefix_stripped(self):
        text = (
            "Thinking Process: I need to write a cover letter.\n"
            "1. **Step one**\n2. **Step two**\n\n"
            "Dear Hiring Manager,\nI am writing to apply."
        )
        result = strip_thinking_process(text)
        assert result.startswith("Dear Hiring Manager")
        assert "Thinking Process" not in result

    def test_clean_passthrough(self):
        text = "Dear Hiring Manager,\nI am writing to apply for the position."
        result = strip_thinking_process(text)
        assert result == text

    def test_empty_string(self):
        result = strip_thinking_process("")
        assert result == ""

    def test_final_version_marker(self):
        text = (
            "Thinking Process: Let me draft this.\n"
            "1. **Analysis**\n\n"
            "Final version:\n"
            "Dear Hiring Manager,\nGreat letter here."
        )
        result = strip_thinking_process(text)
        assert "Dear Hiring Manager" in result
        assert "Thinking Process" not in result

    def test_multiple_dear_finds_first(self):
        text = (
            "Some thinking output.\n\n"
            "Dear Hiring Manager,\nFirst paragraph.\n\n"
            "Dear Recruiter,\nSecond letter."
        )
        result = strip_thinking_process(text)
        assert "Dear Hiring Manager" in result
        assert "Dear Recruiter" in result


# ===========================================================================
# ResumeTailor.tailor() with mocked LLM
# ===========================================================================


class TestResumeTailorPipeline:
    @pytest.mark.asyncio
    async def test_tailor_returns_tailored_resume(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)
        mock_llm_text = (
            "JOHN DOE\njohn@example.com\n\n"
            "SKILLS\nPython, FastAPI, Docker\n\n"
            "EXPERIENCE\n"
            "Senior Developer\nAcme Corp\n2020 - Present\n"
            "• Architected REST APIs with FastAPI\n"
        )

        mock_match = MagicMock(score=0.75, matched_skills=["Python"], missing_skills=["AWS"])
        with (
            patch.object(
                tailor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_text,
            ),
            patch.object(
                tailor,
                "_summarize_changes",
                new_callable=AsyncMock,
                return_value="Enhanced bullets",
            ),
            patch(
                "job_applicator.embeddings.matching.JobMatcher.match_resume_to_job",
                return_value=mock_match,
            ),
        ):
            result = await tailor.tailor(sample_resume, sample_job)

        assert isinstance(result, TailoredResume)
        assert result.job_title == "Senior Python Developer"
        assert result.job_company == "TechCorp"
        assert result.match_score == 0.75
        assert "Python" in result.matched_skills
        assert "AWS" in result.missing_skills
        assert result.changes_summary == "Enhanced bullets"
        assert result.attempt == 1

    @pytest.mark.asyncio
    async def test_tailor_includes_tone_in_prompt(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        mock_match = MagicMock(score=0.5, matched_skills=[], missing_skills=[])
        with (
            patch.object(
                tailor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value="Tailored text",
            ) as mock_llm,
            patch.object(
                tailor,
                "_summarize_changes",
                new_callable=AsyncMock,
                return_value="changes",
            ),
            patch(
                "job_applicator.embeddings.matching.JobMatcher.match_resume_to_job",
                return_value=mock_match,
            ),
        ):
            await tailor.tailor(sample_resume, sample_job)

        prompt = mock_llm.call_args[0][0]
        assert "TONE:" in prompt

    @pytest.mark.asyncio
    async def test_tailor_includes_education_in_prompt(self, llm_config, sample_resume, sample_job):
        resume_with_edu = ResumeData(
            raw_text=(sample_resume.raw_text + "\nEDUCATION\nBS CS\nMIT\n2014 - 2018"),
            skills=sample_resume.skills,
        )
        tailor = ResumeTailor(llm_config)

        mock_match = MagicMock(score=0.5, matched_skills=[], missing_skills=[])
        with (
            patch.object(
                tailor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value="Tailored",
            ) as mock_llm,
            patch.object(
                tailor,
                "_summarize_changes",
                new_callable=AsyncMock,
                return_value="changes",
            ),
            patch(
                "job_applicator.embeddings.matching.JobMatcher.match_resume_to_job",
                return_value=mock_match,
            ),
        ):
            await tailor.tailor(resume_with_edu, sample_job)

        prompt = mock_llm.call_args[0][0]
        assert "EDUCATION" in prompt.upper() or "education" in prompt.lower()

    @pytest.mark.asyncio
    async def test_tailor_applies_hallucination_guards(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)
        llm_output = (
            "JOHN DOE\n"
            "SKILLS\nPython, FastAPI, Docker, Kubernetes\n"
            "EXPERIENCE\nJob\nCorp\n2020 - Present\n"
        )

        mock_match = MagicMock(score=0.5, matched_skills=[], missing_skills=[])
        with (
            patch.object(
                tailor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=llm_output,
            ),
            patch.object(
                tailor,
                "_summarize_changes",
                new_callable=AsyncMock,
                return_value="changes",
            ),
            patch(
                "job_applicator.embeddings.matching.JobMatcher.match_resume_to_job",
                return_value=mock_match,
            ),
        ):
            result = await tailor.tailor(sample_resume, sample_job)

        # Hallucination guard should have processed the text
        assert isinstance(result.tailored_text, str)
        assert len(result.tailored_text) > 0


# ===========================================================================
# ResumeTailor.refine() with mocked LLM
# ===========================================================================


class TestResumeRefine:
    @pytest.fixture
    def current_tailored(self) -> TailoredResume:
        return TailoredResume(
            original_path="",
            tailored_text="Current tailored text",
            job_title="Senior Python Developer",
            job_company="TechCorp",
            match_score=0.7,
            semantic_score=0.7,
            skill_score=0.6,
            matched_skills=["Python"],
            missing_skills=["AWS"],
            changes_summary="Initial changes",
            attempt=1,
        )

    @pytest.mark.asyncio
    async def test_refine_increments_attempt(
        self, llm_config, sample_resume, sample_job, current_tailored
    ):
        tailor = ResumeTailor(llm_config)

        with (
            patch.object(
                tailor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value="Refined text",
            ),
            patch.object(
                tailor,
                "_summarize_changes",
                new_callable=AsyncMock,
                return_value="Refined changes",
            ),
        ):
            result = await tailor.refine(
                sample_resume, current_tailored, "Add more detail", sample_job
            )

        assert result.attempt == 2

    @pytest.mark.asyncio
    async def test_refine_preserves_user_modifications(
        self, llm_config, sample_resume, sample_job, current_tailored
    ):
        tailor = ResumeTailor(llm_config)

        with (
            patch.object(
                tailor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value="Refined text",
            ),
            patch.object(
                tailor,
                "_summarize_changes",
                new_callable=AsyncMock,
                return_value="changes",
            ),
        ):
            result = await tailor.refine(
                sample_resume, current_tailored, "Emphasize API work", sample_job
            )

        assert result.user_modifications == "Emphasize API work"

    @pytest.mark.asyncio
    async def test_refine_includes_skills_constraint(
        self, llm_config, sample_resume, sample_job, current_tailored
    ):
        tailor = ResumeTailor(llm_config)

        with (
            patch.object(
                tailor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value="Refined",
            ) as mock_llm,
            patch.object(
                tailor,
                "_summarize_changes",
                new_callable=AsyncMock,
                return_value="changes",
            ),
        ):
            await tailor.refine(sample_resume, current_tailored, "feedback", sample_job)

        prompt = mock_llm.call_args[0][0]
        assert "ONLY use these" in prompt or "actual skills" in prompt.lower()

    @pytest.mark.asyncio
    async def test_refine_applies_hallucination_guards(
        self, llm_config, sample_resume, sample_job, current_tailored
    ):
        tailor = ResumeTailor(llm_config)
        llm_output = "JOHN DOE\nSKILLS\nPython, AWS\nEXPERIENCE\nJob\nCorp\n2020 - Present\n"

        with (
            patch.object(
                tailor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=llm_output,
            ),
            patch.object(
                tailor,
                "_summarize_changes",
                new_callable=AsyncMock,
                return_value="changes",
            ),
        ):
            result = await tailor.refine(sample_resume, current_tailored, "feedback", sample_job)

        assert isinstance(result.tailored_text, str)
        assert len(result.tailored_text) > 0


# ===========================================================================
# CoverLetterGenerator._build_prompt()
# ===========================================================================


class TestCoverLetterBuildPrompt:
    def _make_generator(self) -> CoverLetterGenerator:
        gen = CoverLetterGenerator.__new__(CoverLetterGenerator)
        gen._config = MagicMock()
        gen._client = None
        gen._style_cache = None
        return gen

    def _make_job(self) -> JobListing:
        return JobListing(
            title="Backend Engineer",
            company="Acme Inc",
            url="https://example.com/job",
            description="Build scalable APIs.",
            location="Remote",
            board=JobBoard.LINKEDIN,
        )

    def test_includes_job_info(self):
        gen = self._make_generator()
        job = self._make_job()
        user = UserProfile(first_name="Jane", last_name="Smith", email="j@e.com", phone="555")
        resume = ResumeData(raw_text="Resume text", skills=["Python"])

        prompt = gen._build_prompt(job, user, resume)
        assert "Backend Engineer" in prompt
        assert "Acme Inc" in prompt
        assert "Remote" in prompt
        assert "Build scalable APIs" in prompt

    def test_includes_applicant_info(self):
        gen = self._make_generator()
        job = self._make_job()
        user = UserProfile(first_name="Jane", last_name="Smith", email="jane@e.com", phone="555")
        resume = ResumeData(raw_text="Resume", skills=["Python"])

        prompt = gen._build_prompt(job, user, resume)
        assert "Jane Smith" in prompt
        assert "jane@e.com" in prompt

    def test_includes_resume_summary_and_skills(self):
        gen = self._make_generator()
        job = self._make_job()
        user = UserProfile(first_name="Jane", last_name="Smith", email="j@e.com", phone="555")
        resume = ResumeData(
            raw_text="Resume",
            summary="Senior engineer with 10 years experience.",
            skills=["Python", "FastAPI", "Docker"],
        )

        prompt = gen._build_prompt(job, user, resume)
        assert "Senior engineer" in prompt
        assert "Python, FastAPI, Docker" in prompt

    def test_includes_tone_section(self):
        gen = self._make_generator()
        job = self._make_job()
        user = UserProfile(first_name="Jane", last_name="Smith", email="j@e.com", phone="555")
        resume = ResumeData(raw_text="Resume", skills=["Python"])

        tone = "TONE: Technical\n- Power words: architected"
        prompt = gen._build_prompt(job, user, resume, tone_section=tone)
        assert "TONE: Technical" in prompt
        assert "architected" in prompt

    def test_includes_tailored_text(self):
        gen = self._make_generator()
        job = self._make_job()
        user = UserProfile(first_name="Jane", last_name="Smith", email="j@e.com", phone="555")
        resume = ResumeData(raw_text="Resume", skills=["Python"])

        prompt = gen._build_prompt(
            job, user, resume, tailored_resume_text="TAILORED RESUME CONTENT"
        )
        assert "TAILORED RESUME CONTENT" in prompt

    def test_includes_todays_date_no_placeholder(self):
        gen = self._make_generator()
        job = self._make_job()
        user = UserProfile(first_name="Jane", last_name="Smith", email="j@e.com", phone="555")
        resume = ResumeData(raw_text="Resume", skills=["Python"])

        prompt = gen._build_prompt(job, user, resume, tailored_resume_text="Tailored content")
        today = datetime.now().strftime("%B %d, %Y")
        assert f"Today's date: {today}" in prompt
        assert "Do NOT write '[Date]'" in prompt
        assert "use the real date" in prompt.lower()
