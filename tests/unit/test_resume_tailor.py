"""Contract tests for source-overlay résumé targeting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.resume_document import ResumeDocument
from job_applicator.documents.resume_tailor import ResumeTailor, parse_sections
from job_applicator.documents.source_facts import build_source_fact_catalog
from job_applicator.documents.source_realization import realize_resume_statement
from job_applicator.embeddings.matching import MatchResult
from job_applicator.exceptions import ConfigError, LLMError, TailorIntegrityError
from job_applicator.models import (
    ExperienceEntry,
    JobBoard,
    JobListing,
    ResumeData,
    ResumeOverlay,
    StyleGuide,
    TailoredResume,
)
from job_applicator.utils.llm import CircuitBreaker, LLMRuntime


def _source_text() -> str:
    return (
        "ALEX MORGAN\n"
        "alex@example.com | 438-555-0100 | Montreal, QC\n\n"
        "SUMMARY\n"
        "Technical support professional with evidence-focused troubleshooting experience.\n\n"
        "EXPERIENCE\n"
        "Technical Support Advisor | UpClick | 2022 - Present\n"
        "• Resolved customer tickets by phone and email.\n"
        "• Documented incidents and coordinated technical escalations.\n\n"
        "Support Analyst | Acme Support | 2020 - 2022\n"
        "• Investigated workstation, account, and network issues.\n\n"
        "PROJECTS\n"
        "• Built a Fedora networking lab for routing practice.\n\n"
        "EDUCATION\n"
        "Certificate in Cybersecurity | Metro College | 2024\n\n"
        "SKILLS\n"
        "Windows, Office 365, Python, Linux, networking"
    )


@pytest.fixture
def sample_resume() -> ResumeData:
    return ResumeData(
        raw_text=_source_text(),
        name="ALEX MORGAN",
        email="alex@example.com",
        phone="438-555-0100",
        summary="Technical support professional with evidence-focused troubleshooting experience.",
        skills=["Windows", "Office 365", "Python", "Linux", "networking"],
        experience=[
            ExperienceEntry(
                title="Technical Support Advisor",
                company="UpClick",
                start_date="2022",
                end_date="Present",
                bullets=[
                    "Resolved customer tickets by phone and email.",
                    "Documented incidents and coordinated technical escalations.",
                ],
            ),
            ExperienceEntry(
                title="Support Analyst",
                company="Acme Support",
                start_date="2020",
                end_date="2022",
                bullets=["Investigated workstation, account, and network issues."],
            ),
        ],
    )


@pytest.fixture
def sample_job() -> JobListing:
    return JobListing(
        title="Technical Support Specialist",
        company="CGI",
        url="https://example.com/job",
        description="Provide English technical support for Windows and networking issues.",
        requirements=["Windows", "Office 365", "networking"],
        location="Montreal, QC",
        board=JobBoard.INDEED,
    )


def _matcher(job: JobListing) -> MagicMock:
    matcher = MagicMock()
    matcher.match_resume_to_job = AsyncMock(
        return_value=MatchResult(
            job=job,
            score=0.72,
            semantic_score=0.5,
            skill_score=0.3,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            summary="Good match",
        )
    )
    return matcher


def _overlay_result(resume: ResumeData) -> tuple[str, ResumeOverlay]:
    source_document = ResumeDocument.parse(resume.raw_text)
    facts = build_source_fact_catalog(resume).facts
    selected = [
        next(fact for fact in facts if text in fact.text)
        for text in (
            "Resolved customer tickets",
            "Documented incidents",
            "Investigated workstation",
        )
    ]
    sentences = [realize_resume_statement(fact) for fact in selected]
    overlay = ResumeOverlay(
        summary_sentences=sentences,
        source_body_sha256=source_document.non_summary_sha256(),
        source_language="en",
    )
    summary = " ".join(sentence.text for sentence in sentences)
    return source_document.with_summary(summary, language="English").render(), overlay


def _tailored(resume: ResumeData, *, text: str | None = None) -> TailoredResume:
    generated, overlay = _overlay_result(resume)
    return TailoredResume(
        original_path="",
        tailored_text=text or generated,
        job_title="Technical Support Specialist",
        job_company="CGI",
        match_score=0.72,
        semantic_score=0.5,
        skill_score=0.3,
        changes_summary="summary overlay",
        grounding_report=None,
        overlay=overlay,
    )


async def test_tailor_requires_matcher_or_match_result(
    sample_resume: ResumeData,
    sample_job: JobListing,
) -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    with pytest.raises(ConfigError, match="configured JobMatcher"):
        await tailor.tailor(sample_resume, sample_job)


async def test_tailor_applies_only_overlay_and_populates_scores(
    sample_resume: ResumeData,
    sample_job: JobListing,
) -> None:
    tailor = ResumeTailor(LLMConfig(model="m", language="en"))
    generated, overlay = _overlay_result(sample_resume)
    tailor._overlay_generator.generate = AsyncMock(  # type: ignore[method-assign]
        return_value=(generated, overlay)
    )

    result = await tailor.tailor(sample_resume, sample_job, matcher=_matcher(sample_job))

    assert result.overlay == overlay
    assert result.prompt_version == "source-overlay-v6"
    assert result.match_score == pytest.approx(0.72)
    assert result.grounding_report is not None and result.grounding_report.clean
    assert ResumeDocument.parse(result.tailored_text).non_summary_sha256() == (
        ResumeDocument.parse(sample_resume.raw_text).non_summary_sha256()
    )


async def test_tailor_passes_style_and_user_focus_only_to_overlay(
    sample_resume: ResumeData,
    sample_job: JobListing,
) -> None:
    tailor = ResumeTailor(LLMConfig(model="m", language="en"))
    generated, overlay = _overlay_result(sample_resume)
    tailor._overlay_generator.generate = AsyncMock(  # type: ignore[method-assign]
        return_value=(generated, overlay)
    )
    style = StyleGuide(
        tone="direct",
        sentence_structure="short",
        vocabulary_level="plain",
        paragraph_style="concise",
        formatting_notes="",
        sample_paragraph="",
    )

    await tailor.tailor(
        sample_resume,
        sample_job,
        user_instructions="Emphasize support evidence",
        style_guide=style,
        matcher=_matcher(sample_job),
    )

    call = tailor._overlay_generator.generate.await_args  # type: ignore[attr-defined]
    assert call.kwargs["resume"] is sample_resume
    assert call.kwargs["style_guide"] is style
    assert call.kwargs["user_instructions"] == "Emphasize support evidence"


async def test_verify_tailored_accepts_exact_overlay(
    sample_resume: ResumeData,
) -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    result = await tailor.verify_tailored(_tailored(sample_resume), sample_resume)
    assert result.grounding_report is not None and result.grounding_report.clean


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda text: text.replace("UpClick", "Different Employer"), "outside the generated"),
        (lambda text: text.replace("Resolved customer", "Handled customer", 1), "provenance"),
    ],
)
async def test_verify_tailored_rejects_text_mutation(
    sample_resume: ResumeData,
    mutation,
    message: str,
) -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    result = _tailored(sample_resume)
    result.tailored_text = mutation(result.tailored_text)
    with pytest.raises(TailorIntegrityError, match=message):
        await tailor.verify_tailored(result, sample_resume)


async def test_verify_tailored_rejects_missing_or_non_deterministic_provenance(
    sample_resume: ResumeData,
) -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    missing = _tailored(sample_resume)
    missing.overlay = None
    with pytest.raises(TailorIntegrityError, match="missing source-overlay"):
        await tailor.verify_tailored(missing, sample_resume)

    drifted = _tailored(sample_resume)
    assert drifted.overlay is not None
    first = drifted.overlay.summary_sentences[0]
    first.text += " Ensured success."
    with pytest.raises(TailorIntegrityError, match="differs from deterministic"):
        await tailor.verify_tailored(drifted, sample_resume)


async def test_refine_regenerates_from_original_and_increments_attempt(
    sample_resume: ResumeData,
    sample_job: JobListing,
) -> None:
    tailor = ResumeTailor(LLMConfig(model="m", language="en"))
    generated, overlay = _overlay_result(sample_resume)
    tailor._overlay_generator.generate = AsyncMock(  # type: ignore[method-assign]
        return_value=(generated, overlay)
    )
    current = _tailored(sample_resume)

    result = await tailor.refine(
        sample_resume,
        current,
        "Use a more concise voice",
        sample_job,
        matcher=_matcher(sample_job),
    )

    assert result.attempt == 2
    assert result.user_modifications == "Use a more concise voice"
    call = tailor._overlay_generator.generate.await_args  # type: ignore[attr-defined]
    assert call.kwargs["resume"] is sample_resume
    assert "Use a more concise voice" in call.kwargs["user_instructions"]
    assert current.tailored_text not in call.kwargs["user_instructions"]


def test_cross_language_tailoring_fails_closed() -> None:
    with pytest.raises(LLMError, match="Cross-language resume tailoring is unavailable"):
        ResumeTailor._require_matching_source_language(
            "Professional experience and technical skills",
            "French",
        )
    ResumeTailor._require_matching_source_language(
        "Professional experience and technical skills",
        "English",
    )


def test_tailor_and_cover_letter_can_share_runtime() -> None:
    from job_applicator.documents.cover_letter import CoverLetterGenerator

    runtime = LLMRuntime(breaker=CircuitBreaker(name="shared"))
    cover = CoverLetterGenerator(LLMConfig(), runtime=runtime)
    tailor = ResumeTailor(LLMConfig(), runtime=runtime)
    assert cover._runtime is runtime
    assert tailor._overlay_generator._runtime is runtime


class TestParseSections:
    def test_standard_and_mixed_case_sections(self) -> None:
        sections = parse_sections(
            "JOHN DOE\n\nSummary\nExperienced developer.\n\n"
            "EXPERIENCE\nEngineer\n\nTechnical Skills:\nPython"
        )
        assert [section.name for section in sections] == [
            "Summary",
            "EXPERIENCE",
            "Technical Skills:",
        ]

    def test_unstructured_text_returns_full_document(self) -> None:
        text = "Just a plain resume with no section headers."
        sections = parse_sections(text)
        assert len(sections) == 1
        assert sections[0].name == "Full Document"
        assert sections[0].text == text

    def test_section_text_is_preserved(self) -> None:
        sections = parse_sections("SKILLS\nPython\nLinux\n\nEXPERIENCE\nSupport")
        skills = next(section for section in sections if section.name == "SKILLS")
        assert skills.text == "Python\nLinux"


def test_tailored_resume_overlay_round_trip(sample_resume: ResumeData) -> None:
    result = _tailored(sample_resume)
    restored = TailoredResume.model_validate_json(result.model_dump_json())
    assert restored.overlay == result.overlay
    assert restored.overlay is not None
    assert restored.overlay.architecture_version == "source-overlay-v6"
