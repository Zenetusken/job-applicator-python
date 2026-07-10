"""Cross-cutting date-audit and source-context tests for document generation."""

from __future__ import annotations

from datetime import datetime

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.cover_letter import (
    CoverLetterGenerator,
)
from job_applicator.documents.resume_tailor import ResumeDateValidator
from job_applicator.documents.source_facts import (
    build_source_fact_catalog,
    is_substantive_source_fact,
)
from job_applicator.exceptions import LLMError
from job_applicator.models import JobBoard, JobListing, ResumeData, SourceFactCatalog, UserProfile


def _audit(text: str, *, year: int = 2026):
    return ResumeDateValidator(reference_date=datetime(year, 7, 1)).audit(ResumeData(raw_text=text))


def test_date_audit_handles_empty_and_year_only_ranges() -> None:
    empty = _audit("No dates")
    assert empty.entries == []
    assert empty.is_ordered

    result = _audit("EXPERIENCE\nRole\nCompany\n2018 - 2020")
    assert len(result.entries) == 1
    assert result.entries[0].start == "2018"
    assert result.entries[0].end == "2020"


def test_date_audit_normalizes_month_names_and_present() -> None:
    result = _audit("EXPERIENCE\nRole\nCompany\nJan 2020 - Present")
    assert result.entries[0].start == "January 2020"
    assert result.entries[0].is_current


def test_date_audit_detects_reverse_chronology() -> None:
    result = _audit("EXPERIENCE\nOld Role\nA\n2015 - 2018\n\nNew Role\nB\n2021 - Present")
    assert not result.is_ordered
    assert result.ordering_issues


def test_date_audit_current_work_prevents_education_staleness_noise() -> None:
    result = _audit(
        "EXPERIENCE\nCurrent Role\nA\n2020 - Present\n\nEDUCATION\nDegree\nSchool\n2000 - 2004",
        year=2030,
    )
    assert result.staleness_issues == []
    assert not result.is_stale


def test_date_audit_flags_old_latest_entry() -> None:
    result = _audit("EXPERIENCE\nRole\nA\n2010 - 2015", year=2030)
    assert result.is_stale
    assert any("Most recent entry" in issue for issue in result.staleness_issues)


def test_date_audit_surfaces_gap_and_overlap_as_advisory() -> None:
    gap = _audit("EXPERIENCE\nNew\nA\n2022 - Present\n\nOld\nB\n2015 - 2018")
    assert gap.employment_gaps

    overlap = _audit("EXPERIENCE\nNew\nA\n2020 - 2024\n\nOld\nB\n2018 - 2022")
    assert overlap.overlap_issues


def _job(description: str = "English support role") -> JobListing:
    return JobListing(
        title="Support Analyst",
        company="Target Corp",
        url="https://example.test/job",
        description=description,
        requirements=["Windows", "networking"],
        board=JobBoard.INDEED,
    )


def _resume() -> ResumeData:
    return ResumeData(
        raw_text=(
            "Alex Morgan\n\nSUMMARY\nSupport professional.\n\n"
            "EXPERIENCE\nAdvisor | Source Employer | 2022 - Present\n"
            "• Resolved Windows support tickets.\n"
            "• Documented and escalated unresolved incidents.\n"
            "• Supported users by phone and email.\n\n"
            "PROJECTS\n• Built a networking lab.\n\n"
            "EDUCATION\nCertificate | Source College | 2024\n\n"
            "SKILLS\nWindows, networking"
        ),
        summary="Support professional.",
        skills=["Windows", "networking"],
    )


async def test_cover_targeting_returns_only_ranked_source_facts() -> None:
    resume = _resume()
    job = _job("SECRET_JOB_FACT Kubernetes ownership")
    generator = CoverLetterGenerator(LLMConfig(language="en"))
    facts = SourceFactCatalog(
        facts=[
            fact
            for fact in build_source_fact_catalog(resume).facts
            if is_substantive_source_fact(fact)
        ]
    )
    selected = await generator._select_source_facts(job, facts)

    assert len(selected.facts) == 3
    assert all(fact in facts.facts for fact in selected.facts)
    assert all("SECRET_JOB_FACT" not in fact.text for fact in selected.facts)
    assert selected.facts[0].text == "Resolved Windows support tickets."


async def test_cover_generation_fails_closed_for_cross_language_source() -> None:
    generator = CoverLetterGenerator(LLMConfig(language="fr"))
    with pytest.raises(LLMError, match="Cross-language cover-letter generation is unavailable"):
        await generator.generate_with_overlay(
            _job("Poste français avec soutien technique et réseaux"),
            UserProfile(
                first_name="Alex",
                last_name="Morgan",
                email="alex@example.test",
                phone="514-555-0100",
            ),
            _resume(),
        )
