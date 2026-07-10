"""Bounded, source-preserving résumé summary generation."""

from __future__ import annotations

from job_applicator.config import LLMConfig
from job_applicator.documents.resume_document import ResumeDocument
from job_applicator.documents.source_facts import (
    build_source_fact_catalog,
    format_job_target_context,
    is_substantive_source_fact,
    select_relevant_source_facts,
)
from job_applicator.documents.source_realization import realize_resume_statement
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor
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


def _substantive_candidates(resume: ResumeData, job: JobListing) -> SourceFactCatalog:
    catalog = build_source_fact_catalog(resume)
    evidence = SourceFactCatalog(
        facts=[fact for fact in catalog.facts if is_substantive_source_fact(fact)]
    )
    if len(evidence.facts) < 3:
        raise LLMError(
            "Source-preserving tailoring requires at least three substantive source facts."
        )
    return select_relevant_source_facts(
        evidence,
        job,
        max_chars=10_000,
        max_facts=32,
        relevance_order=False,
        include_identity=False,
    )


class ResumeOverlayGenerator:
    """Generate the only mutable résumé field from deterministically ranked source facts."""

    def __init__(
        self,
        config: LLMConfig,
        runtime: LLMRuntime,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._target_skill_extractor = LLMSkillExtractor(config)

    async def _select(
        self,
        job: JobListing,
        candidates: SourceFactCatalog,
        user_instructions: str = "",
    ) -> SourceFactCatalog:
        target_criteria = list(job.requirements)
        if not target_criteria:
            target_criteria = await self._target_skill_extractor.extract(
                format_job_target_context(job, max_description_chars=1_200),
                runtime=self._runtime,
            )
        criteria_text = ", ".join(target_criteria)
        selection_text = " ".join(
            part for part in (criteria_text, user_instructions.strip()) if part
        )
        selection_target = job.model_copy(
            update={"description": selection_text, "requirements": target_criteria}
        )
        selected = select_relevant_source_facts(
            candidates,
            selection_target,
            max_chars=8_000,
            max_facts=3,
            relevance_order=True,
            include_identity=False,
        )
        if len(selected.facts) != 3:
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
    ) -> tuple[str, ResumeOverlay]:
        del style_guide
        source_document = ResumeDocument.parse(resume.raw_text)
        candidates = _substantive_candidates(resume, job)
        selected = await self._select(job, candidates, user_instructions)
        statements = self._realize_summary(selected)
        summary = " ".join(sentence.text.strip() for sentence in statements)
        tailored = source_document.with_summary(summary, language=language)
        overlay = ResumeOverlay(
            summary_sentences=statements,
            source_body_sha256=source_document.non_summary_sha256(),
            source_language="fr" if language == "French" else "en",
        )
        return tailored.render(), overlay
