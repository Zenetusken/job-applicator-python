"""Bounded, source-preserving résumé summary generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from job_applicator.config import LLMConfig
from job_applicator.documents.resume_document import ResumeDocument
from job_applicator.documents.source_facts import (
    build_source_fact_catalog,
    is_substantive_source_fact,
)
from job_applicator.documents.source_realization import realize_resume_statement
from job_applicator.exceptions import LLMError
from job_applicator.models import (
    JobListing,
    ResumeData,
    ResumeOverlay,
    SourceBackedStatement,
    SourceFactCatalog,
    StyleGuide,
)
from job_applicator.utils.llm import LLMRuntime

if TYPE_CHECKING:
    from job_applicator.embeddings.matching import JobMatcher, SourceFactRankingResult


def _substantive_candidates(resume: ResumeData) -> SourceFactCatalog:
    catalog = build_source_fact_catalog(resume)
    evidence = SourceFactCatalog(
        facts=[fact for fact in catalog.facts if is_substantive_source_fact(fact)]
    )
    if len(evidence.facts) < 3:
        raise LLMError(
            "Source-preserving tailoring requires at least three substantive source facts."
        )
    return evidence


class ResumeOverlayGenerator:
    """Generate the only mutable résumé field from deterministically ranked source facts."""

    def __init__(
        self,
        config: LLMConfig,
        runtime: LLMRuntime,
    ) -> None:
        self._config = config
        self._runtime = runtime

    async def _select(
        self,
        job: JobListing,
        candidates: SourceFactCatalog,
        matcher: JobMatcher,
        user_instructions: str = "",
    ) -> SourceFactRankingResult:
        selected = await matcher.rank_source_facts(
            job,
            candidates,
            max_facts=3,
            selection_focus=user_instructions,
        )
        if len(selected.facts.facts) != 3:
            raise LLMError("Source-preserving tailoring requires three ranked source facts.")
        return selected

    def _realize_summary(self, selected: SourceFactCatalog) -> list[SourceBackedStatement]:
        return [realize_resume_statement(fact) for fact in selected.facts]

    async def generate(
        self,
        *,
        resume: ResumeData,
        job: JobListing,
        language: str,
        style_guide: StyleGuide | None,
        user_instructions: str,
        matcher: JobMatcher,
    ) -> tuple[str, ResumeOverlay]:
        del style_guide
        source_document = ResumeDocument.parse(resume.raw_text)
        candidates = _substantive_candidates(resume)
        selected = await self._select(job, candidates, matcher, user_instructions)
        statements = self._realize_summary(selected.facts)
        summary = " ".join(sentence.text.strip() for sentence in statements)
        tailored = source_document.with_summary(summary, language=language)
        overlay = ResumeOverlay(
            summary_sentences=statements,
            source_body_sha256=source_document.non_summary_sha256(),
            source_language="fr" if language == "French" else "en",
            evidence_ranking=selected.ranking,
        )
        return tailored.render(), overlay
