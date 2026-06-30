"""Tests for resume tailoring engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.resume_tailor import (
    CHANGES_PROMPT_TEMPLATE,
    TAILOR_PROMPT_TEMPLATE,
    TAILOR_SYSTEM_PROMPT,
    ResumeTailor,
    parse_sections,
)
from job_applicator.exceptions import GroundingUnavailableError
from job_applicator.models import (
    ClaimCheck,
    GroundingReport,
    JobBoard,
    JobListing,
    ResumeData,
    TailoredResume,
)


def _tr(text: str) -> TailoredResume:
    return TailoredResume(
        original_path="",
        tailored_text=text,
        job_title="T",
        job_company="C",
        match_score=0.5,
        semantic_score=0.5,
        skill_score=0.5,
        changes_summary="",
    )


async def test_verify_tailored_attaches_report_for_review() -> None:
    # The résumé path SURFACES the grounding report; it never auto-strips (spec §6).
    tailor = ResumeTailor(LLMConfig(model="m"))
    tailor._verifier.verify = AsyncMock(  # type: ignore[method-assign]
        return_value=GroundingReport(unsupported=[ClaimCheck(claim="x", grounded=False)])
    )
    out = await tailor.verify_tailored(
        _tr("Maintained 100%."), ResumeData(raw_text="src", skills=[])
    )
    assert out.grounding_report is not None
    assert len(out.grounding_report.unsupported) == 1
    assert out.tailored_text == "Maintained 100%."  # text untouched — flagged, not stripped


async def test_verify_tailored_failsafe_leaves_none() -> None:
    # fail-safe (#4): a verifier failure leaves grounding_report=None — never blocked, never a
    # false "verified clean".
    tailor = ResumeTailor(LLMConfig(model="m"))
    tailor._verifier.verify = AsyncMock(  # type: ignore[method-assign]
        side_effect=GroundingUnavailableError("down")
    )
    out = await tailor.verify_tailored(_tr("anything"), ResumeData(raw_text="src", skills=[]))
    assert out.grounding_report is None


async def test_refine_verified_reverifies_the_refined_result() -> None:
    # #4: an interactively refined résumé gets the SAME grounding pass as the primary — refine()
    # then verify_tailored(), the report attached for review (never auto-stripped).
    tailor = ResumeTailor(LLMConfig(model="m"))
    tailor.refine = AsyncMock(return_value=_tr("REFINED text"))  # type: ignore[method-assign]
    tailor._verifier.verify = AsyncMock(  # type: ignore[method-assign]
        return_value=GroundingReport(unsupported=[ClaimCheck(claim="y", grounded=False)])
    )
    resume = ResumeData(raw_text="src", skills=[])
    out = await tailor.refine_verified(resume, _tr("CURRENT"), "feedback", MagicMock())
    assert out.tailored_text == "REFINED text"
    assert out.grounding_report is not None and len(out.grounding_report.unsupported) == 1
    tailor.refine.assert_awaited_once()  # type: ignore[attr-defined]


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

    @pytest.mark.asyncio
    async def test_call_llm_honors_configured_max_tokens(self):
        """_call_llm must pass the configured max_tokens, not a hardcoded value."""
        from job_applicator.config import LLMConfig

        config = LLMConfig(api_base="http://localhost:8000/v1", model="m", max_tokens=1234)
        tailor = ResumeTailor(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acompletion:
            await tailor._call_llm("prompt")

        assert mock_acompletion.call_args.kwargs["max_tokens"] == 1234

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
            tone_section="TONE: Corporate",
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

    def test_system_prompt_has_few_shot_examples(self):
        """System prompt should contain before/after examples."""
        assert "BEFORE summary" in TAILOR_SYSTEM_PROMPT
        assert "AFTER summary" in TAILOR_SYSTEM_PROMPT
        assert "BEFORE bullet" in TAILOR_SYSTEM_PROMPT
        assert "AFTER bullet" in TAILOR_SYSTEM_PROMPT

    def test_system_prompt_has_third_person_rule(self):
        """System prompt should enforce third person in summaries."""
        assert "THIRD PERSON" in TAILOR_SYSTEM_PROMPT
        assert "'I'" in TAILOR_SYSTEM_PROMPT or "never use" in TAILOR_SYSTEM_PROMPT.lower()

    def test_system_prompt_has_power_word_limits(self):
        """System prompt should limit power word usage."""
        assert "sparingly" in TAILOR_SYSTEM_PROMPT.lower()

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

    @pytest.mark.asyncio
    async def test_tailor_populates_scores(self, llm_config, sample_resume, sample_job):
        """TailoredResume should have non-zero semantic_score and skill_score."""
        from job_applicator.embeddings.matching import MatchResult

        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Tailored text"

        mock_match = MatchResult(
            job=sample_job,
            score=0.72,
            semantic_score=0.5,
            skill_score=0.3,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            summary="Good match",
        )
        mock_matcher = MagicMock()
        mock_matcher.match_resume_to_job = AsyncMock(return_value=mock_match)

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tailor.tailor(sample_resume, sample_job, matcher=mock_matcher)

        assert result.semantic_score > 0.0
        assert result.skill_score > 0.0
        assert result.match_score == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_tailor_rejects_empty_completion(self, llm_config, sample_resume, sample_job):
        """An empty LLM completion must raise LLMError (typed), not yield an empty TailoredResume
        that silently flows into cover-letter generation + PDF rendering."""
        from job_applicator.embeddings.matching import MatchResult
        from job_applicator.exceptions import LLMError

        tailor = ResumeTailor(llm_config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "   "  # empty after stripping

        mock_matcher = MagicMock()
        mock_matcher.match_resume_to_job = AsyncMock(
            return_value=MatchResult(
                job=sample_job,
                score=0.5,
                semantic_score=0.5,
                skill_score=0.5,
                matched_skills=[],
                missing_skills=[],
                summary="",
            )
        )
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(LLMError):
                await tailor.tailor(sample_resume, sample_job, matcher=mock_matcher)

    @pytest.mark.asyncio
    async def test_tailor_accepts_matcher_param(self, llm_config, sample_resume, sample_job):
        """Passing a matcher should reuse it instead of creating a new one."""
        from job_applicator.embeddings.matching import MatchResult

        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Tailored text"

        mock_match = MatchResult(
            job=sample_job,
            score=0.8,
            semantic_score=0.6,
            skill_score=0.4,
            matched_skills=["Windows"],
            missing_skills=[],
            summary="Strong match",
        )
        mock_matcher = MagicMock()
        mock_matcher.match_resume_to_job = AsyncMock(return_value=mock_match)

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await tailor.tailor(sample_resume, sample_job, matcher=mock_matcher)

        mock_matcher.match_resume_to_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_refine_accepts_matcher_param(self, llm_config, sample_resume, sample_job):
        """Refine should accept and use a matcher parameter."""
        from job_applicator.embeddings.matching import MatchResult

        tailor = ResumeTailor(llm_config)
        initial = TailoredResume(
            original_path="",
            tailored_text="Initial text",
            job_title="Technical Support Specialist",
            job_company="CGI",
            match_score=0.7,
            semantic_score=0.5,
            skill_score=0.3,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            changes_summary="changes",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Refined text"

        mock_match = MatchResult(
            job=sample_job,
            score=0.85,
            semantic_score=0.65,
            skill_score=0.5,
            matched_skills=["Windows", "Office 365"],
            missing_skills=[],
            summary="Strong match",
        )
        mock_matcher = MagicMock()
        mock_matcher.match_resume_to_job = AsyncMock(return_value=mock_match)

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tailor.refine(
                sample_resume,
                initial,
                "Add detail",
                sample_job,
                matcher=mock_matcher,
            )

        assert result.semantic_score > 0.0
        assert result.skill_score > 0.0
        mock_matcher.match_resume_to_job.assert_called_once()

    def test_call_llm_temperature_default(self, llm_config):
        """_call_llm should default to temperature=0.4."""
        tailor = ResumeTailor(llm_config)
        import inspect

        sig = inspect.signature(tailor._call_llm)
        assert sig.parameters["temperature"].default == 0.4


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


class TestTailorWithTone:
    @pytest.mark.asyncio
    async def test_tailor_includes_tone_in_prompt(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Tailored with tone"

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            await tailor.tailor(sample_resume, sample_job)

        first_call = mock_call.call_args_list[0]
        assert "TONE:" in str(first_call)


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


class TestTailorWorkflow:
    def test_tailor_session_workflow(self):
        """Test the full accept/retry/input workflow with mock data."""
        from job_applicator.models import TailorSession

        session = TailorSession(
            original_text="Original resume",
            job_title="Dev",
            job_company="Co",
        )

        for i in range(3):
            result = TailoredResume(
                original_path="",
                tailored_text=f"Tailored version {i + 1}",
                job_title="Dev",
                job_company="Co",
                match_score=0.5 + i * 0.1,
                semantic_score=0.5,
                skill_score=0.5,
                changes_summary=f"Changes for attempt {i + 1}",
                attempt=i + 1,
                user_modifications="" if i == 0 else "more detail",
            )
            session.add_attempt(result)

        assert len(session.attempts) == 3
        assert session.current.tailored_text == "Tailored version 3"

        session.select(0)
        assert session.current.tailored_text == "Tailored version 1"

        with pytest.raises(IndexError):
            session.select(99)

    def test_parse_sections_and_select(self):
        """Test section parsing for editing workflow."""
        from job_applicator.documents.resume_tailor import parse_sections

        text = (
            "John Doe - Developer\n\n"
            "SUMMARY\nExperienced developer.\n\n"
            "SKILLS\nPython, JavaScript\n\n"
            "EXPERIENCE\nSoftware engineer at Corp (2020-2024)\n"
        )
        sections = parse_sections(text)
        assert len(sections) == 3
        assert sections[0].name == "SUMMARY"
        assert "Experienced developer" in sections[0].text


class TestCoverLetterWorkflow:
    def test_cover_letter_session_workflow(self):
        from job_applicator.models import CoverLetterResult, CoverLetterSession

        session = CoverLetterSession(job_title="Dev", job_company="Co")

        for i in range(3):
            session.add_attempt(
                CoverLetterResult(
                    job_title="Dev",
                    job_company="Co",
                    cover_letter_text=f"Letter version {i + 1}",
                    attempt=i + 1,
                )
            )

        assert len(session.attempts) == 3
        assert session.current.cover_letter_text == "Letter version 3"
        assert session.attempts[0].cover_letter_text == "Letter version 1"

        session.select(0)
        assert session.current.cover_letter_text == "Letter version 1"


class TestAuditFixes:
    """Tests for the 5 audit fixes: date, power words, job titles, education order, first person."""

    def test_tailor_prompt_forbids_first_person(self):
        """Fix 5: System prompt should forbid 'I', 'my', 'me' in summary."""
        from job_applicator.documents.resume_tailor import TAILOR_SYSTEM_PROMPT

        has_third = (
            "THIRD PERSON" in TAILOR_SYSTEM_PROMPT or "third person" in TAILOR_SYSTEM_PROMPT.lower()
        )
        assert has_third
        assert "'I'" in TAILOR_SYSTEM_PROMPT or "'my'" in TAILOR_SYSTEM_PROMPT

    def test_tailor_prompt_limits_power_words(self):
        """Fix 2: System prompt should limit ornate power verbs."""
        from job_applicator.documents.resume_tailor import TAILOR_SYSTEM_PROMPT

        assert "sparingly" in TAILOR_SYSTEM_PROMPT.lower() or "2-3 per job" in TAILOR_SYSTEM_PROMPT

    def test_tailor_prompt_preserves_job_titles(self):
        """Fix 3: System prompt should preserve complete job titles."""
        from job_applicator.documents.resume_tailor import TAILOR_SYSTEM_PROMPT

        assert "NEVER remove or shorten job titles" in TAILOR_SYSTEM_PROMPT
        assert "Dental & Medical" in TAILOR_SYSTEM_PROMPT

    def test_tailor_prompt_enforces_reverse_chronological_education(self):
        """Fix 4: System prompt should enforce reverse-chronological education."""
        from job_applicator.documents.resume_tailor import TAILOR_SYSTEM_PROMPT

        has_order = (
            "REVERSE-CHRONOLOGICAL" in TAILOR_SYSTEM_PROMPT
            or "most recent first" in TAILOR_SYSTEM_PROMPT.lower()
        )
        assert has_order

    def test_cover_letter_prompt_includes_date(self):
        """Fix 1: Cover letter prompt should include today's date."""
        from datetime import datetime as dt

        from job_applicator.documents.cover_letter import CoverLetterGenerator

        generator = CoverLetterGenerator.__new__(CoverLetterGenerator)
        generator._config = MagicMock()

        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = MagicMock()
        user.first_name = "John"
        user.last_name = "Doe"
        user.email = "j@e.com"
        resume = ResumeData(raw_text="Resume", skills=["Python"])

        prompt = generator._build_prompt(
            job,
            user,
            resume,
            tailored_resume_text="Tailored resume text",
        )

        today = dt.now().strftime("%B %d, %Y")
        assert today in prompt
        assert "Today's date:" in prompt
        assert "Do NOT write" in prompt  # instruction to not use [Date] placeholder

    def test_cover_letter_prompt_no_date_without_tailored_text(self):
        """Cover letter prompt without tailored_resume_text should not inject date."""
        from job_applicator.documents.cover_letter import CoverLetterGenerator

        generator = CoverLetterGenerator.__new__(CoverLetterGenerator)
        generator._config = MagicMock()

        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = MagicMock()
        user.first_name = "John"
        user.last_name = "Doe"
        user.email = "j@e.com"
        resume = ResumeData(raw_text="Resume", skills=["Python"])

        prompt = generator._build_prompt(job, user, resume)

        assert "Today's date:" not in prompt


class TestResumeDateValidator:
    def test_audit_with_no_dates(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="No dates here at all.")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) == 0
        assert not result.is_stale
        assert result.is_ordered
        assert result.latest_date == ""
        assert result.earliest_date == ""

    def test_audit_with_present_date(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EXPERIENCE\nSoftware Engineer\nCorp, City\n2020 - Present")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) > 0
        assert any(e.is_current for e in result.entries)
        present_entry = next(e for e in result.entries if e.is_current)
        assert present_entry.end == "Present"
        assert present_entry.start == "2020"

    def test_audit_detects_staleness(self):
        from datetime import datetime

        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EXPERIENCE\nOld Job\nCorp, City\n2000 - 2005")
        validator = ResumeDateValidator(reference_date=datetime(2030, 1, 1))
        result = validator.audit(resume)
        assert result.is_stale
        assert len(result.staleness_issues) > 0
        assert "2005" in result.staleness_issues[0]

    def test_audit_ordering_issues(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=("EXPERIENCE\nOld Job\nCorp\n2010 - 2015\nNew Job\nCorp\n2018 - 2024")
        )
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.ordering_issues) > 0
        assert not result.is_ordered
        assert any("should come after" in issue for issue in result.ordering_issues)

    def test_audit_year_only_dates(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EXPERIENCE\nJob\nCorp\n2018 - 2020")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) > 0
        entry = result.entries[0]
        assert entry.start == "2018"
        assert entry.end == "2020"
        assert entry.is_current is False

    def test_audit_month_year_format(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EXPERIENCE\nJob\nCorp\nJan 2020 - Jun 2022")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) > 0
        entry = result.entries[0]
        assert entry.start == "January 2020"
        assert entry.end == "June 2022"

    def test_audit_empty_text(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) == 0
        assert not result.is_stale
        assert result.is_ordered

    def test_audit_multiple_entries_chronological(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=("EXPERIENCE\nNew Job\nCorp\n2020 - Present\nOld Job\nCorp\n2015 - 2019")
        )
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) == 2
        assert result.is_ordered
        assert result.latest_date != ""
        assert result.earliest_date != ""

    def test_audit_latest_and_earliest_dates(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=("EXPERIENCE\nNewest Job\nCorp\n2022 - Present\nOldest Job\nCorp\n2010 - 2014")
        )
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        # "Present" resolves to current date (June 2026), earliest is 2010
        assert result.latest_date != ""
        assert result.earliest_date != ""
        assert "2010" in result.earliest_date

    def test_audit_education_staleness(self):
        from datetime import datetime

        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EDUCATION\nBS Computer Science\nMIT\n1998 - 2002")
        validator = ResumeDateValidator(reference_date=datetime(2030, 1, 1))
        result = validator.audit(resume)
        assert len(result.staleness_issues) > 0
        assert result.is_stale

    def test_audit_education_old_but_current_work_not_stale(self):
        from datetime import datetime

        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=(
                "EXPERIENCE\nCurrent Job\nCorp\n2020 - Present\n\n"
                "EDUCATION\nBS CS\nMIT\n2000 - 2004"
            )
        )
        validator = ResumeDateValidator(reference_date=datetime(2030, 1, 1))
        result = validator.audit(resume)
        # General staleness check passes (Present entry is current),
        # but education-specific staleness is still flagged
        general_staleness = [s for s in result.staleness_issues if "Most recent entry" in s]
        edu_staleness = [s for s in result.staleness_issues if "Education" in s]
        assert len(general_staleness) == 0
        assert len(edu_staleness) > 0

    def test_audit_entries_from_different_sections(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=(
                "EXPERIENCE\nEngineer\nCorp\n2018 - 2022\n\nEDUCATION\nBS CS\nMIT\n2014 - 2018"
            )
        )
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        sections = {e.section for e in result.entries}
        assert "Experience" in sections
        assert "Education" in sections

    def test_audit_section_detection_case_insensitive(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="experience\nEngineer\nCorp\n2020 - 2023")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) == 1
        assert result.entries[0].section == "Experience"


async def test_resume_tailor_call_llm_uses_breaker() -> None:
    """Cycle 2b: ResumeTailor routes its LLM calls through the injected breaker, so
    repeated failures open it and a subsequent call fails fast (CircuitOpenError)
    without hitting the endpoint — the app's largest LLM calls are now protected."""
    from job_applicator.config import LLMConfig
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitBreaker, CircuitOpenError, LLMRuntime

    runtime = LLMRuntime(breaker=CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=60.0))
    tailor = ResumeTailor(LLMConfig(), runtime=runtime)
    assert tailor._breaker is runtime.breaker

    # acompletion fails → _call_llm wraps as LLMError → breaker records the failure.
    with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("down")):
        with pytest.raises(LLMError):
            await tailor._call_llm("prompt")

    # threshold=1 → breaker now OPEN → next call fails fast, endpoint untouched.
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        with pytest.raises(CircuitOpenError):
            await tailor._call_llm("prompt")
        mock_acomp.assert_not_called()


def test_cover_letter_and_tailor_share_one_runtime_breaker() -> None:
    """Cycle 2b: one LLMRuntime passed to both consumers yields ONE shared breaker —
    the mechanism the batch/tailor commands rely on (they build a single runtime via
    _make_runtime and pass it to both CoverLetterGenerator and ResumeTailor, so a
    down endpoint trips one breaker that guards the whole run)."""
    from job_applicator.config import LLMConfig
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.utils.llm import CircuitBreaker, LLMRuntime

    runtime = LLMRuntime(breaker=CircuitBreaker(name="shared"))
    gen = CoverLetterGenerator(LLMConfig(), runtime=runtime)
    tailor = ResumeTailor(LLMConfig(), runtime=runtime)
    assert gen._breaker is tailor._breaker
    assert gen._breaker is runtime.breaker


@pytest.mark.asyncio
async def test_refine_passes_style_guide_to_prompt(sample_resume, sample_job):
    """Cycle 2b polish: refinements must preserve the writing style guide."""
    from job_applicator.documents.style_analyzer import StyleAnalyzer
    from job_applicator.models import StyleGuide

    config = LLMConfig(api_base="http://localhost:8000/v1", model="test")
    tailor = ResumeTailor(config)

    current = TailoredResume(
        original_path="",
        tailored_text="Original tailored text",
        job_title=sample_job.title,
        job_company=sample_job.company,
        match_score=0.75,
        semantic_score=0.0,
        skill_score=0.0,
        matched_skills=["Windows"],
        missing_skills=[],
        changes_summary="Initial tailoring",
        attempt=1,
    )

    style = StyleGuide(
        tone="casual",
        sentence_structure="short",
        vocabulary_level="simple",
        paragraph_style="brief",
        formatting_notes="",
        sample_paragraph="",
    )

    captured_prompt = ""

    async def _capture_call(prompt: str, temperature: float = 0.7):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "Refined text"

    with patch.object(tailor, "_call_llm", side_effect=_capture_call):
        with patch.object(tailor, "_summarize_changes", new_callable=AsyncMock) as mock_changes:
            mock_changes.return_value = "Refined based on feedback"
            await tailor.refine(
                sample_resume,
                current,
                "Make it more concise",
                sample_job,
                style_guide=style,
            )

    style_section = StyleAnalyzer.format_style_for_prompt(style)
    assert style_section in captured_prompt
    assert "Maintain this writing style" in captured_prompt


class TestEmptySectionStripping:
    """Tests for removing empty Certifications/Languages sections."""

    def test_strip_empty_certifications_languages_removes_both_when_absent(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original = "Name\nSkills\nPython\nExperience\nJob"
        tailored = (
            "Name\n"
            "Skills\nPython\n"
            "Experience\nJob\n"
            "**Certifications**\n"
            "**Languages**\n"
            "References\nAvailable"
        )
        result = tailor._strip_empty_certifications_languages(tailored, original)
        assert "**Certifications**" not in result
        assert "**Languages**" not in result
        assert "Skills" in result
        assert "Experience" in result
        assert "References" in result

    def test_strip_empty_certifications_keeps_section_when_original_has_it(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original = "Name\nCertifications\nAWS CPA\n"
        tailored = "Name\n**Certifications**\nAWS CPA\n"
        result = tailor._strip_empty_certifications_languages(tailored, original)
        assert "**Certifications**" in result

    def test_strip_empty_languages_keeps_section_when_original_has_it(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original = "Name\nLanguages\nEnglish, French\n"
        tailored = "Name\n**Languages**\nEnglish, French\n"
        result = tailor._strip_empty_certifications_languages(tailored, original)
        assert "**Languages**" in result

    def test_validate_skills_keeps_comma_separated_original_skills(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original_skills = ["Python", "FastAPI", "PostgreSQL", "Docker"]
        tailored = "**Skills**\n• Python, FastAPI, PostgreSQL, Docker\n\n**Experience**\n"
        result = tailor._validate_skills(tailored, original_skills)
        assert "Python, FastAPI, PostgreSQL, Docker" in result

    def test_validate_skills_drops_hallucinated_skills_in_comma_list(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original_skills = ["Python", "FastAPI"]
        tailored = "**Skills**\n• Python, Kubernetes, FastAPI\n\n**Experience**\n"
        result = tailor._validate_skills(tailored, original_skills)
        assert "Python" in result
        assert "FastAPI" in result
        assert "Kubernetes" not in result

    def test_validate_skills_keeps_extra_words_when_core_skill_matches(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original_skills = ["Python", "Docker"]
        tailored = "**Skills**\n• Python (advanced)\n• Docker & Kubernetes\n"
        result = tailor._validate_skills(tailored, original_skills)
        assert "Python (advanced)" in result
        assert "Docker & Kubernetes" in result


@pytest.mark.asyncio
async def test_summarize_changes_raises_not_fabricated(llm_config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A summary LLM failure RAISES the typed error — never returns a fabricated
    'summary generation failed' string that masks the failure as a real summary."""
    from job_applicator.exceptions import LLMError

    tailor = ResumeTailor(llm_config)

    async def _boom(*_a: object, **_k: object) -> str:
        raise LLMError("summary call failed")

    monkeypatch.setattr(tailor, "_call_llm", _boom)
    with pytest.raises(LLMError):
        await tailor._summarize_changes("original resume text", "tailored resume text")


class TestStripUnearnedCredentials:
    """Deterministic backstop: remove credential/status claims the candidate doesn't hold from the
    tailored SUMMARY — the freeform section the structured-section guards don't cover. Surfaced by
    a live tailor that emitted 'Accredited security operations professional' for a candidate with
    only in-progress coursework."""

    def test_strips_unearned_credential_and_keeps_sentence_grammatical(self, llm_config) -> None:
        tailor = ResumeTailor(llm_config)
        original = "SUMMARY\nOperations professional seeking a security role.\n\nSKILLS\n• Linux\n"
        tailored = (
            "SUMMARY\nAccredited security operations professional with 10+ years.\n\n"
            "SKILLS\n• Linux\n"
        )
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "accredited" not in out.lower()  # the unearned credential is gone
        # the sentence is repaired (capitalized, no leading gap, no double space)
        assert "Security operations professional with 10+ years." in out
        assert "• Linux" in out  # other sections untouched

    def test_preserves_credential_the_candidate_actually_holds(self, llm_config) -> None:
        tailor = ResumeTailor(llm_config)
        original = "SUMMARY\nAnalyst.\n\nCERTIFICATIONS\n• Certified Ethical Hacker\n"
        tailored = "SUMMARY\nCertified security analyst seeking a SOC role.\n\nSKILLS\n• Nmap\n"
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "Certified security analyst" in out  # a real, held credential is kept

    def test_benign_credential_word_elsewhere_does_not_license_overclaim(self, llm_config) -> None:
        # 'certified' as a verb in an EXPERIENCE bullet must NOT license a summary credential claim
        # (the whole-document leak the summary-scoping closes).
        tailor = ResumeTailor(llm_config)
        original = (
            "SUMMARY\nSupport specialist.\n\n"
            "EXPERIENCE\nAgent\n• Certified that tickets were resolved before closing.\n"
        )
        tailored = (
            "SUMMARY\nCertified support specialist seeking a security role.\n\nSKILLS\n• Linux\n"
        )
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "Certified support specialist" not in out
        assert "Support specialist seeking a security role." in out

    def test_handles_x_and_y_credential_phrase(self, llm_config) -> None:
        tailor = ResumeTailor(llm_config)
        original = "SUMMARY\nAnalyst.\n\nSKILLS\n• Linux\n"
        tailored = "SUMMARY\nCertified and licensed analyst with experience.\n\nSKILLS\n• Linux\n"
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "certified" not in out.lower() and "licensed" not in out.lower()
        assert "Analyst with experience." in out  # 'X and Y' collapses cleanly

    def test_only_scrubs_summary_not_other_sections(self, llm_config) -> None:
        tailor = ResumeTailor(llm_config)
        original = "SUMMARY\nAnalyst.\n\nSKILLS\n• Linux\n"
        tailored = (
            "SUMMARY\nAccredited analyst seeking a role.\n\n"
            "EXPERIENCE\nAuditor\n• Certified compliance reports each quarter.\n"
        )
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "Accredited analyst" not in out  # summary scrubbed
        assert "Certified compliance reports each quarter." in out  # experience bullet untouched

    def test_strips_overclaim_under_unlabelled_and_variant_headers(self, llm_config) -> None:
        """Covers a header-less leading paragraph and odd/missing summary labels — not just a bare
        'SUMMARY' — since a 4B that overclaims often omits or renames the header (the silent-bypass
        the adversarial review caught: the scrub region is the leading block, not one header)."""
        tailor = ResumeTailor(llm_config)
        original = "Jane Roe\njane@x.com\n\nOperations lead.\n\nSKILLS\n• Linux\n"
        for header in ("", "PROFILE SUMMARY", "About", "Summary:"):
            lead = f"{header}\n" if header else ""
            tailored = (
                "Jane Roe\njane@x.com\n\n"
                f"{lead}"
                "Accredited security analyst seeking a SOC role.\n\nSKILLS\n• Linux\n"
            )
            out = tailor._strip_unearned_credentials(tailored, original)
            assert "accredited" not in out.lower(), f"header={header!r}"
            assert "Security analyst seeking a SOC role." in out, f"header={header!r}"
            assert "• Linux" in out  # the skills section is never scrubbed
