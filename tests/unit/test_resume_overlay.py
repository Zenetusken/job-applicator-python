"""Tests for bounded source-backed résumé summary generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from job_applicator.config import LLMConfig
from job_applicator.documents.resume_document import ResumeDocument
from job_applicator.documents.resume_overlay import (
    ResumeOverlayGenerator,
    _substantive_candidates,
)
from job_applicator.embeddings.matching import SourceFactRankingResult
from job_applicator.models import (
    ExperienceEntry,
    JobBoard,
    JobListing,
    RankedSourceFact,
    ResumeData,
    SourceFactRanking,
    TargetCriteria,
    TargetCriterion,
)
from job_applicator.utils.llm import LLMRuntime


def _resume() -> ResumeData:
    text = (
        "ALEX MORGAN\n"
        "alex@example.com | 438-555-0100\n\n"
        "SUMMARY\n"
        "Technical support professional.\n\n"
        "EXPERIENCE\n"
        "Technical Support Advisor | UpClick | 2022 - Present\n"
        "• Resolved customer tickets by phone and email.\n"
        "• Documented incidents and coordinated technical escalations.\n\n"
        "Support Analyst | Acme | 2020 - 2022\n"
        "• Investigated workstation, account, and network issues.\n\n"
        "PROJECTS\n"
        "• Built a Fedora networking lab for routing practice.\n\n"
        "EDUCATION\n"
        "Certificate in Cybersecurity | Metro College | 2024\n\n"
        "SKILLS\n"
        "Windows, Python, Linux, networking"
    )
    return ResumeData(
        raw_text=text,
        summary="Technical support professional.",
        skills=["Windows", "Python", "Linux", "networking"],
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
            )
        ],
    )


def _job() -> JobListing:
    return JobListing(
        title="Technical Support Specialist",
        company="Target Company",
        url="https://example.test/job",
        description="Support Windows users and troubleshoot networking incidents.",
        requirements=["Windows", "networking", "incident documentation"],
        board=JobBoard.INDEED,
    )


def _generator() -> ResumeOverlayGenerator:
    return ResumeOverlayGenerator(
        LLMConfig(model="test", language="en"),
        LLMRuntime.defaults(name="overlay-test"),
    )


def _matcher(candidates):
    selected = type(candidates)(facts=candidates.facts[:3])
    ranking = SourceFactRanking(
        target_criteria=TargetCriteria(
            job_source_sha256="a" * 64,
            criteria=[TargetCriterion(name="Windows support", evidence="Windows users")],
        ),
        ranked_facts=[
            RankedSourceFact(
                fact_id=fact.fact_id,
                score=score,
                strongest_similarity=score,
                strongest_criterion_index=0,
            )
            for fact, score in zip(selected.facts, (0.9, 0.8, 0.7), strict=True)
        ],
    )
    matcher = MagicMock()
    matcher.rank_source_facts = AsyncMock(
        return_value=SourceFactRankingResult(facts=selected, ranking=ranking)
    )
    return matcher


async def test_generate_changes_only_summary_and_records_provenance() -> None:
    resume = _resume()
    job = _job()
    generator = _generator()
    candidates = _substantive_candidates(resume)
    matcher = _matcher(candidates)
    selected = (await generator._select(job, candidates, matcher)).facts.facts
    tailored, overlay = await generator.generate(
        resume=resume,
        job=job,
        language="English",
        style_guide=None,
        user_instructions="Use a direct voice",
        matcher=matcher,
    )

    source_document = ResumeDocument.parse(resume.raw_text)
    tailored_document = ResumeDocument.parse(tailored)
    assert tailored_document.non_summary_sha256() == source_document.non_summary_sha256()
    assert overlay.source_body_sha256 == source_document.non_summary_sha256()
    assert [statement.text for statement in overlay.summary_sentences] == [
        fact.text for fact in selected
    ]
    assert "UpClick" in tailored


async def test_selection_uses_grounded_criteria_but_realization_excludes_context() -> None:
    resume = _resume()
    job = _job()
    generator = _generator()
    candidates = _substantive_candidates(resume)
    matcher = _matcher(candidates)
    selected = (await generator._select(job, candidates, matcher)).facts.facts
    _tailored, overlay = await generator.generate(
        resume=resume,
        job=job,
        language="English",
        style_guide=None,
        user_instructions="",
        matcher=matcher,
    )

    assert [statement.text for statement in overlay.summary_sentences] == [
        fact.text for fact in selected
    ]
    assert all(
        fact.context not in statement.text
        for fact, statement in zip(selected, overlay.summary_sentences, strict=True)
        if fact.context
    )


async def test_selection_returns_three_distinct_source_facts() -> None:
    resume = _resume()
    job = _job()
    generator = _generator()
    candidates = _substantive_candidates(resume)
    selected = await generator._select(job, candidates, _matcher(candidates))

    assert len(selected.facts.facts) == 3
    assert len({fact.fact_id for fact in selected.facts.facts}) == 3
    assert all(fact in candidates.facts for fact in selected.facts.facts)


def test_summary_realization_preserves_selected_fact_text() -> None:
    resume = _resume()
    generator = _generator()
    candidates = _substantive_candidates(resume)
    selected = candidates.facts[:3]
    statements = generator._realize_summary(type(candidates)(facts=selected))

    assert [statement.text for statement in statements] == [fact.text for fact in selected]
