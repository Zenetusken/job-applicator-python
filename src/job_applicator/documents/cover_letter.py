"""Source-backed cover letter generation from grounded job criteria."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from job_applicator.config import LLMConfig
from job_applicator.documents.resume import ResumeLoader
from job_applicator.documents.resume_document import ResumeDocument
from job_applicator.documents.source_facts import (
    build_source_fact_catalog,
    format_job_target_context,
    is_substantive_source_fact,
    select_relevant_source_facts,
)
from job_applicator.documents.source_integrity import assess_source_integrity
from job_applicator.documents.source_realization import realize_cover_statements
from job_applicator.documents.style_analyzer import StyleAnalyzer
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor
from job_applicator.exceptions import LLMError
from job_applicator.models import (
    CoverLetterOverlay,
    GroundingReport,
    JobListing,
    ResumeData,
    SourceBackedStatement,
    SourceFactCatalog,
    StyleGuide,
    UserProfile,
)
from job_applicator.utils.language import detect_language, resolve_output_language
from job_applicator.utils.llm import LLMRuntime
from job_applicator.utils.llm import (
    strip_thinking_process as strip_thinking_process,
)
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.cover_letter")


class CoverLetterOutput(BaseModel):
    """Structured output from LLM for cover letter generation."""

    model_config = {"extra": "forbid"}

    cover_letter: str = Field(description="The generated cover letter text")
    key_points: list[str] = Field(
        description="Key points highlighted in the letter", default_factory=list
    )


SourceBackedSentence = SourceBackedStatement


class CoverLetterDraft(BaseModel):
    """Source-backed body sentences for a structurally assembled cover letter."""

    model_config = {"extra": "forbid"}

    body_facts: list[SourceBackedStatement] = Field(
        min_length=3,
        max_length=3,
        description=(
            "Three direct source-backed sentences from coherent contexts. Each sentence "
            "must stand on its own without a capability, readiness, fit, or contribution "
            "conclusion."
        ),
    )


@dataclass(frozen=True)
class _GeneratedCover:
    """One structured model draft plus its assembled application text and evidence."""

    text: str
    draft: CoverLetterDraft
    source_facts: SourceFactCatalog
    language: str


def _canonical_sign_off(language: str) -> str:
    """The sign-off word the app appends and expects, per resolved output language."""
    return "Cordialement" if language == "French" else "Sincerely"


def _application_frame(
    job: JobListing,
    language: str,
    evidence_kinds: Sequence[str],
) -> tuple[str, str]:
    """Return non-factual opening and closing paragraphs for the target application."""

    kind_labels = {
        "experience": ("professional experience", "expérience professionnelle"),
        "education": ("education", "formation"),
        "projects": ("project work", "projets"),
    }
    language_index = 1 if language == "French" else 0
    fallback = ("background", "parcours")[language_index]
    labels = [
        kind_labels.get(kind, (fallback, fallback))[language_index] for kind in evidence_kinds
    ]
    conjunction = "et" if language == "French" else "and"
    if len(labels) == 1:
        evidence_label = labels[0]
    elif len(labels) == 2:
        evidence_label = f"{labels[0]} {conjunction} {labels[1]}"
    else:
        serial_comma = "," if language != "French" else ""
        evidence_label = f"{', '.join(labels[:-1])}{serial_comma} {conjunction} {labels[-1]}"
    if language == "French":
        return (
            f"Je vous présente ma candidature au poste de {job.title} chez {job.company}. "
            f"Les exemples ci-dessous mettent en valeur les éléments de mon {evidence_label} "
            "les plus pertinents pour ce poste.",
            "Ensemble, ces exemples donnent un aperçu concis de mon parcours pertinent. "
            "Je serais disponible pour en discuter plus en détail et pour en apprendre davantage "
            "sur les priorités du poste et les besoins de votre équipe.",
        )
    return (
        f"I am applying for the {job.title} position at {job.company}. The examples below "
        f"highlight the parts of my {evidence_label} most relevant to this role.",
        "Together, these examples provide a concise view of my relevant background. I would "
        "welcome the opportunity to discuss them in more detail and learn more about the role's "
        "priorities and your team's needs.",
    )


class CoverLetterGenerator:
    """Generate structurally targeted cover letters without model-written applicant claims."""

    def __init__(
        self,
        config: LLMConfig,
        runtime: LLMRuntime | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime or LLMRuntime.defaults(name="cover-letter")
        self._target_skill_extractor = LLMSkillExtractor(config)

    async def load_style_guide(self, style_guide_path: str, ocr_mode: str = "auto") -> StyleGuide:
        """Load and analyze one or more style-guide files into a single StyleGuide.

        ``style_guide_path`` may be a single file or a comma-separated list. All
        files are parsed through ``ResumeLoader``, so PDFs, text files, DOCX, and
        images are supported with the same OCR fallback used for résumés. A single
        file is analyzed directly; multiple files are analyzed individually and
        merged. Per-text caching lives in ``StyleAnalyzer``, so repeated calls for
        the same path are cheap.
        """
        from pathlib import Path

        from job_applicator.exceptions import DocumentError

        paths = [p.strip() for p in style_guide_path.split(",") if p.strip()]
        if not paths:
            raise DocumentError("No style guide paths provided")

        loader = ResumeLoader()
        texts: list[str] = []
        for path_str in paths:
            path = Path(path_str)
            try:
                resume_data = loader.load(path, ocr_mode=ocr_mode)
            except DocumentError as exc:
                # Re-raise with a style-guide-specific prefix so callers know
                # which path failed without leaking raw ResumeLoader internals.
                raise DocumentError(f"Could not load style guide {path}: {exc}") from exc
            texts.append(resume_data.raw_text)

        analyzer = StyleAnalyzer(self._config, runtime=self._runtime)
        if len(texts) == 1:
            style = await analyzer.analyze(texts[0])
        else:
            style = await analyzer.analyze_multiple(texts)

        logger.info("Loaded style guide from %s: tone=%s", style_guide_path, style.tone)
        return style

    def _validate_output(self, generated: _GeneratedCover, resume: ResumeData) -> None:
        """Validate a generated cover letter.

        Rejects structural failures and generic source-integrity violations. It never rewrites
        generated prose.
        """
        self._validate_source_fact_citations(
            generated.draft,
            generated.source_facts,
            language=generated.language,
        )
        text = generated.text
        stripped = text.strip()
        if not stripped:
            raise LLMError("Generated cover letter is empty")
        if len(stripped) < 50:
            raise LLMError("Generated cover letter is too short")
        placeholders = r"company\s*name|hiring\s*manager|position\s*title|your\s*name|date|address"
        placeholder_pattern = rf"\[\s*(?:{placeholders})\s*\]"
        if re.search(placeholder_pattern, text, re.IGNORECASE):
            raise LLMError("Generated cover letter contains placeholder text")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]
        if len(paragraphs) != 4:
            raise LLMError("Generated cover letter must contain exactly four paragraphs")
        if re.search(r"(?m)^\s*(?:#{1,6}|[-*+]\s+)|`|\*\*", text):
            raise LLMError("Generated cover letter contains markdown formatting")

        seen_sentences: set[str] = set()
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            normalized = re.sub(r"\W+", " ", sentence).casefold().strip()
            if len(normalized.split()) < 8:
                continue
            if normalized in seen_sentences:
                raise LLMError("Generated cover letter repeats a substantive sentence")
            seen_sentences.add(normalized)

        integrity = assess_source_integrity(
            source=resume,
            generated_resume="",
            generated_cover=text,
            require_resume_structure=False,
        )
        if integrity.failures:
            raise LLMError(
                "Generated cover letter failed source integrity: " + " | ".join(integrity.failures)
            )

    @staticmethod
    def _append_sign_off(text: str, user: UserProfile, language: str = "English") -> str:
        """Append the canonical sign-off after the deterministic body."""

        name = f"{user.first_name} {user.last_name}".strip()
        return f"{text.rstrip()}\n\n{_canonical_sign_off(language)},\n{name}"

    def _validate_source_fact_citations(
        self,
        draft: CoverLetterDraft,
        source_facts: SourceFactCatalog,
        *,
        language: str = "English",
    ) -> None:
        """Reject unknown citations or claims outside deterministic realization."""

        facts_by_id = {fact.fact_id: fact for fact in source_facts.facts}
        cited = [fact_id for sentence in draft.body_facts for fact_id in sentence.fact_ids]
        unknown = sorted(set(cited) - facts_by_id.keys())
        if unknown:
            raise LLMError(f"Cover letter cited unknown source fact IDs: {', '.join(unknown)}")
        if len(cited) != 3 or set(cited) != facts_by_id.keys():
            raise LLMError("Cover-letter body must use each selected source fact exactly once")

        expected = realize_cover_statements(
            [facts_by_id[sentence.fact_ids[0]] for sentence in draft.body_facts],
            language=language,
        )
        non_deterministic = [
            sentence.text
            for sentence, expected_sentence in zip(draft.body_facts, expected, strict=True)
            if sentence.text != expected_sentence.text
        ]
        if non_deterministic:
            raise LLMError(
                "Cover-letter body differs from deterministic source realization: "
                + "; ".join(non_deterministic)
            )

    async def _select_source_facts(
        self,
        job: JobListing,
        candidates: SourceFactCatalog,
        selection_focus: str = "",
    ) -> SourceFactCatalog:
        """Rank source facts against a grounded target profile and take the top three."""

        target_criteria = list(job.requirements)
        if not target_criteria:
            target_criteria = await self._target_skill_extractor.extract(
                format_job_target_context(job, max_description_chars=1_200),
                runtime=self._runtime,
            )
        criteria_text = ", ".join(target_criteria)
        selection_text = " ".join(part for part in (criteria_text, selection_focus.strip()) if part)
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
            raise LLMError("Source-backed cover letters require three ranked source facts.")
        return selected

    async def _generate_raw(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None,
        tone_section: str,
        tailored_resume_text: str,
        selection_focus: str = "",
    ) -> _GeneratedCover:
        """Select source facts, then realize their claims without an LLM prose pass."""

        del user, style_guide, tone_section, tailored_resume_text
        language = resolve_output_language(self._config.language, job.description)
        source_language = "French" if detect_language(resume.raw_text) == "fr" else "English"
        if source_language != language:
            raise LLMError(
                "Cross-language cover-letter generation is unavailable: the source resume is "
                f"{source_language}, but the requested output is {language}. Provide a {language} "
                "source resume so generation and grounding stay in one language."
            )
        candidates = self._cover_evidence_candidates(resume, job)
        source_facts = await self._select_source_facts(
            job,
            candidates,
            selection_focus=selection_focus,
        )
        draft = CoverLetterDraft(
            body_facts=realize_cover_statements(source_facts.facts, language=language)
        )
        self._validate_source_fact_citations(draft, source_facts, language=language)
        primary_body = " ".join(sentence.text.strip() for sentence in draft.body_facts[:2])
        supporting_body = draft.body_facts[2].text.strip()
        evidence_kinds = list(dict.fromkeys(fact.kind for fact in source_facts.facts))
        opening, closing = _application_frame(job, language, evidence_kinds)
        return _GeneratedCover(
            text=f"{opening}\n\n{primary_body}\n\n{supporting_body}\n\n{closing}",
            draft=draft,
            source_facts=source_facts,
            language=language,
        )

    @staticmethod
    def _cover_evidence_candidates(resume: ResumeData, job: JobListing) -> SourceFactCatalog:
        """Bound the substantive source evidence before grounded-criteria ranking."""

        catalog = build_source_fact_catalog(resume)
        evidence = SourceFactCatalog(
            facts=[fact for fact in catalog.facts if is_substantive_source_fact(fact)]
        )
        if len(evidence.facts) < 3:
            raise LLMError("Source-backed cover letters require at least three primary body facts.")
        return select_relevant_source_facts(
            evidence,
            job,
            max_chars=10_000,
            max_facts=32,
            relevance_order=False,
            include_identity=False,
        )

    async def generate(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
        tailored_resume_text: str = "",
    ) -> str:
        """Generate a cover letter while retaining the legacy text-only API."""

        letter, _overlay = await self.generate_with_overlay(
            job,
            user,
            resume,
            style_guide,
            tone_section,
            tailored_resume_text,
        )
        return letter

    async def generate_with_overlay(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
        tailored_resume_text: str = "",
        *,
        selection_focus: str = "",
    ) -> tuple[str, CoverLetterOverlay]:
        """Generate deterministic source-backed claims and return their provenance."""

        language = resolve_output_language(self._config.language, job.description)
        logger.info(
            "Generating cover letter in %s (language setting=%s) for %s at %s",
            language,
            self._config.language,
            job.title,
            job.company,
        )
        generated = await self._generate_raw(
            job,
            user,
            resume,
            style_guide,
            tone_section,
            tailored_resume_text,
            selection_focus=selection_focus,
        )
        self._validate_output(generated, resume=resume)
        letter = self._append_sign_off(generated.text, user, language)
        overlay = CoverLetterOverlay(
            body_sentences=generated.draft.body_facts,
            source_body_sha256=ResumeDocument.parse(resume.raw_text).non_summary_sha256(),
            source_language="fr" if language == "French" else "en",
        )

        logger.info(
            "Generated cover letter for %s at %s (%d chars)",
            job.title,
            job.company,
            len(letter),
        )
        return letter, overlay

    async def generate_verified(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
        tailored_resume_text: str = "",
    ) -> str:
        """Generate with source-citation and deterministic-realization validation."""

        letter, _overlay = await self.generate_verified_with_overlay(
            job,
            user,
            resume,
            style_guide,
            tone_section,
            tailored_resume_text,
        )
        return letter

    async def generate_verified_with_overlay(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
        tailored_resume_text: str = "",
        *,
        selection_focus: str = "",
    ) -> tuple[str, CoverLetterOverlay]:
        """Generate a verified letter together with its source-evidence overlay."""

        return await self.generate_with_overlay(
            job,
            user,
            resume,
            style_guide,
            tone_section,
            tailored_resume_text,
            selection_focus=selection_focus,
        )

    async def refine_verified(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        current_text: str,
        user_feedback: str,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
    ) -> tuple[str, GroundingReport | None]:
        """Refine fact selection while retaining the legacy text/report API."""

        letter, _overlay, report = await self.refine_verified_with_overlay(
            job,
            user,
            resume,
            current_text,
            user_feedback,
            style_guide,
            tone_section,
        )
        return letter, report

    async def refine_verified_with_overlay(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        current_text: str,
        user_feedback: str,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
    ) -> tuple[str, CoverLetterOverlay, GroundingReport]:
        """Re-select source facts from user focus and return auditable provenance."""

        del current_text
        letter, overlay = await self.generate_verified_with_overlay(
            job,
            user,
            resume,
            style_guide,
            tone_section,
            selection_focus=user_feedback,
        )
        return letter, overlay, GroundingReport()

    async def refine(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        current_text: str,
        user_feedback: str,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
    ) -> str:
        """Refine by re-selecting source facts from the user's focus instruction."""

        del current_text
        letter, _overlay = await self.generate_with_overlay(
            job,
            user,
            resume,
            style_guide,
            tone_section,
            selection_focus=user_feedback,
        )
        return letter

    def generate_from_template(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
    ) -> str:
        """Generate a cover letter using local template (no LLM)."""
        from jinja2 import Template

        template = Template("""Write a cover letter for the following position:

Job Title: {{ job.title }}
Company: {{ job.company }}
Location: {{ job.location }}
{% if job.description %}
Job Description:
{{ job.description }}
{% endif %}

Applicant Profile:
Name: {{ user.first_name }} {{ user.last_name }}
Email: {{ user.email }}
{% if resume.summary %}
Summary: {{ resume.summary }}
{% endif %}
{% if resume.skills %}
Key Skills: {{ resume.skills | join(', ') }}
{% endif %}

Generate a professional cover letter:""")

        return template.render(job=job, user=user, resume=resume)
