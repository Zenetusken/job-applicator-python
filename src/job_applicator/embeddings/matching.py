"""Job matching using embeddings - semantic similarity between resumes and jobs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from job_applicator.config import EmbeddingConfig, LLMConfig
from job_applicator.embeddings.service import EmbeddingService, EmbeddingVector
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor
from job_applicator.models import JobListing, ResumeData, coverage_measured
from job_applicator.utils.llm import LLMRuntime
from job_applicator.utils.logging import get_logger
from job_applicator.utils.verbose import VerboseReporter

logger = get_logger("embeddings.matching")


@dataclass
class MatchResult:
    """Result of matching a resume to a job."""

    job: JobListing
    score: float
    semantic_score: float
    skill_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    summary: str


@dataclass
class SkillMatch:
    """Detailed skill matching result."""

    skill: str
    similarity: float
    matched: bool


class JobMatcher:
    """Match resumes to job listings using embeddings.

    Uses mxbai-embed-large-v1 for semantic similarity between:
    - Resume text and job descriptions
    - Individual skills and requirements
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        llm_config: LLMConfig | None = None,
        runtime: LLMRuntime | None = None,
        reporter: VerboseReporter | None = None,
        *,
        grounding_mode: str = "evidence_span",
    ) -> None:
        self._config = embedding_config
        self._service = EmbeddingService(embedding_config)
        self._skill_extractor = LLMSkillExtractor(
            llm_config or LLMConfig(), grounding_mode=grounding_mode
        )
        self._runtime = runtime
        self._reporter = reporter

    def embed_text(self, text: str, prefix: str = "") -> EmbeddingVector:
        """Generate embedding for text with optional query prefix.

        Args:
            text: Text to embed
            prefix: Optional prefix for asymmetric retrieval

        Returns:
            Embedding vector
        """
        return self._service.embed(prefix + text if prefix else text)

    @staticmethod
    def _is_pii_or_noise(line: str, name_lower: str) -> bool:
        """Whether a raw-text line is bullet noise or personal contact info.

        Filters generically (no hardcoded names): bullet glyphs, the
        candidate's own name, and lines that are just an email address.
        """
        if line in ("•", "·", "-"):
            return True
        if name_lower and line.lower() == name_lower:
            return True
        # A bare email/contact line (single token containing "@").
        return "@" in line and " " not in line

    def compute_resume_embedding(self, resume: ResumeData) -> EmbeddingVector:
        """Compute embedding for a resume.

        Uses the search query prefix for asymmetric retrieval since the resume
        is the "query" side of the job matching search.
        """
        # Build rich text representation from multiple sources
        parts = []

        # Add summary if available
        if resume.summary:
            parts.append(resume.summary)

        # Add skills
        if resume.skills and resume.skills[0] != "•":
            parts.append("Skills: " + ", ".join(resume.skills))

        # Add experience
        for exp in resume.experience[:5]:
            if exp.title:
                parts.append(f"{exp.title} at {exp.company}")

        # Fall back to raw text sections if structured data is sparse
        if len(parts) < 2:
            # Extract key sections from raw text
            lines = resume.raw_text.split("\n")
            current_section = ""
            section_text: list[str] = []
            # Skip the candidate's own name and contact lines: PII that adds
            # noise rather than signal to the match embedding.
            name_lower = resume.name.strip().lower() if resume.name else ""

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # Detect section headers
                if stripped in (
                    "Skills",
                    "Experience",
                    "Education",
                    "Certifications",
                    "Summary",
                    "Objective",
                    "Technical Skills",
                ):
                    if section_text and current_section:
                        parts.append(f"{current_section}: {' '.join(section_text)}")
                    current_section = stripped
                    section_text = []
                elif self._is_pii_or_noise(stripped, name_lower):
                    continue
                else:
                    section_text.append(stripped)

            # Add last section
            if section_text and current_section:
                parts.append(f"{current_section}: {' '.join(section_text)}")

        # Combine all parts, respecting token limits
        text = " | ".join(parts)[:1500]
        # Use search prefix for asymmetric retrieval (resume = query)
        prefix = "Represent this sentence for searching relevant passages: "
        return self._service.embed(prefix + text)

    def compute_job_embedding(self, job: JobListing) -> EmbeddingVector:
        """Compute embedding for a job listing.

        Combines title, description, and requirements for rich representation.
        """
        parts = []

        # Title and company
        parts.append(f"Job: {job.title} at {job.company}")

        if job.location:
            parts.append(f"Location: {job.location}")

        if job.description:
            # Use more of the description for better matching
            parts.append(job.description[:500])

        if job.requirements:
            parts.append(f"Requirements: {', '.join(job.requirements)}")

        text = " | ".join(parts)[:1500]
        return self._service.embed(text)

    async def match_resume_to_job(
        self,
        resume: ResumeData,
        job: JobListing,
    ) -> MatchResult:
        """Compute match score between resume and job.

        Score combines:
        - Semantic similarity (60% weight)
        - Skill coverage (40% weight)

        Returns:
            MatchResult with score, matched/missing skills, and summary
        """
        # Compute embeddings off the event loop (encode is blocking CPU work).
        resume_emb = await asyncio.to_thread(self.compute_resume_embedding, resume)
        job_emb = await asyncio.to_thread(self.compute_job_embedding, job)

        # Compute semantic similarity
        semantic_score = self._service.similarity(resume_emb, job_emb)

        # Skill matching
        matched_skills, missing_skills = await self._match_skills(
            resume.skills, job.requirements, resume.raw_text, job.description
        )

        # Combined score: 60% semantic + 40% skill coverage (semantic-only when skill is unknown)
        score, skill_score = self._combined_score(semantic_score, matched_skills, missing_skills)

        # Generate summary
        summary = self._generate_match_summary(score, matched_skills, missing_skills)

        return MatchResult(
            job=job,
            score=score,
            semantic_score=semantic_score,
            skill_score=skill_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            summary=summary,
        )

    async def rank_jobs(
        self,
        resume: ResumeData,
        jobs: list[JobListing],
        top_k: int = 10,
    ) -> list[MatchResult]:
        """Rank jobs by match score to resume.

        Args:
            resume: Resume to match against
            jobs: List of jobs to rank
            top_k: Number of top results to return

        Returns:
            List of MatchResult sorted by score descending
        """
        if not jobs:
            return []

        # Compute resume embedding once (off the event loop — blocking CPU encode).
        resume_emb = await asyncio.to_thread(self.compute_resume_embedding, resume)

        # Compute job embeddings in batch
        job_texts = []
        for job in jobs:
            parts = [f"Job: {job.title} at {job.company}"]
            if job.location:
                parts.append(f"Location: {job.location}")
            if job.description:
                parts.append(job.description[:500])
            if job.requirements:
                parts.append(f"Requirements: {', '.join(job.requirements)}")
            job_texts.append(" | ".join(parts)[:1500])

        job_embs = await asyncio.to_thread(self._service.embed_batch, job_texts)

        # Compute similarities with combined scoring
        matches = []
        for job, job_emb in zip(jobs, job_embs, strict=False):
            semantic_score = self._service.similarity(resume_emb, job_emb)
            matched, missing = await self._match_skills(
                resume.skills, job.requirements, resume.raw_text, job.description
            )
            # Combined score: 60% semantic + 40% skill coverage (semantic-only when unknown)
            score, skill_score = self._combined_score(semantic_score, matched, missing)
            summary = self._generate_match_summary(score, matched, missing)

            matches.append(
                MatchResult(
                    job=job,
                    score=score,
                    semantic_score=semantic_score,
                    skill_score=skill_score,
                    matched_skills=matched,
                    missing_skills=missing,
                    summary=summary,
                )
            )

        # Sort by score and return top_k
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]

    async def _match_skills(
        self,
        resume_skills: list[str],
        job_requirements: list[str],
        resume_text: str = "",
        job_description: str = "",
    ) -> tuple[list[str], list[str]]:
        """Match resume skills to job requirements using embeddings.

        Args:
            resume_skills: Extracted skills from resume
            job_requirements: Required skills from job listing
            resume_text: Full resume text for fallback matching
            job_description: Job description text; used to infer requirements when
                none are explicitly provided.

        Returns:
            Tuple of (matched_skills, missing_requirements)
        """
        from job_applicator.skills import is_hard_negative, normalize_skill

        if not job_requirements:
            job_requirements = await self._skill_extractor.extract(
                job_description,
                runtime=self._runtime,
                reporter=self._reporter,
            )

        # Normalize and drop generic traits/hard negatives.
        norm_skills = [normalize_skill(s) for s in resume_skills]
        valid_skills = [
            s
            for s in norm_skills
            # >= 2 so short skills (Go, C#, AI, ML) aren't silently dropped from coverage.
            if len(s.strip()) >= 2 and s.strip() != "•" and not is_hard_negative(s)
        ]

        # Preserve original requirement text for reporting while matching on
        # normalized forms.
        norm_reqs = [normalize_skill(r) for r in job_requirements]
        req_lookup = {n: r for n, r in zip(norm_reqs, job_requirements, strict=False) if n}
        valid_reqs = [
            (n, r)
            for n, r in zip(norm_reqs, job_requirements, strict=False)
            if n and not is_hard_negative(n)
        ]

        # If no valid skills, use resume text lines as skills
        if not valid_skills and resume_text:
            # Extract potential skills from resume text
            lines = resume_text.split("\n")
            valid_skills = [
                normalize_skill(line.strip())
                for line in lines
                if 10 < len(line.strip()) < 80
                and not line.strip().startswith(("•", "·", "-"))
                and not is_hard_negative(line.strip())
            ]

        if not valid_skills or not valid_reqs:
            fallback_missing = [
                req_lookup.get(n, r) for n, r in zip(norm_reqs, job_requirements, strict=False) if n
            ]
            return [], fallback_missing

        # Compute embeddings on normalized texts (off the event loop — blocking CPU encode).
        skill_embs = await asyncio.to_thread(self._service.embed_batch, valid_skills)
        req_texts = [n for n, _ in valid_reqs]
        req_embs = await asyncio.to_thread(self._service.embed_batch, req_texts)

        # Find matches using similarity threshold
        matched: list[str] = []
        missing: list[str] = []
        # Empirically tuned (2026-06-22): mxbai-embed-large-v1 scores ANY two same-domain tech
        # terms ~0.55-0.73 (Java~Python 0.73, Kubernetes~Docker 0.70, React~Python 0.62), so the
        # old 0.55 marked unrelated skills "covered" (a Python résumé reported NO missing skills
        # for a React job). Genuine matches/synonyms/supersets score >=0.78 (Postgres~PostgreSQL
        # 0.91, containerization~Docker 0.78); 0.75 sits in the gap — drops the false-positives,
        # keeps real coverage. (Tuned on a Python-résumé/tech-job sample; if false-negatives show
        # up in other domains — a genuine match dipping below 0.75 — revisit the value.)
        threshold = 0.75

        used_skills: set[str] = set()
        for i, (_norm_req, original_req) in enumerate(valid_reqs):
            best_score = 0.0
            best_skill = ""

            # Pick the best skill that hasn't already been claimed by an
            # earlier requirement, so two requirements never fight over the
            # same skill (which used to mark the loser as falsely "missing").
            for j, skill in enumerate(valid_skills):
                if skill in used_skills:
                    continue
                sim = self._service.similarity(req_embs[i], skill_embs[j])
                if sim > best_score:
                    best_score = sim
                    best_skill = skill

            if best_score >= threshold and best_skill:
                matched.append(best_skill)
                used_skills.add(best_skill)
            else:
                missing.append(original_req)

        return matched, missing

    def _compute_skill_score(
        self,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> float:
        """Compute skill coverage score (0.0 to 1.0).

        Score = matched / total_requirements
        """
        total = len(matched_skills) + len(missing_skills)
        if total == 0:
            return 0.5  # Neutral if no requirements
        return len(matched_skills) / total

    def _combined_score(
        self,
        semantic_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> tuple[float, float]:
        """Blend semantic similarity (60%) + skill coverage (40%) → (combined, skill).

        When skill coverage is genuinely UNKNOWN — no requirements to compare against (e.g. none
        listed AND none extractable, or the extractor LLM is down) — rank on semantic similarity
        ALONE rather than injecting a neutral 0.5 floor (which would add a uniform +0.2 to every
        such job). The reported skill is 0.0 in that case (no coverage measured)."""
        if not coverage_measured(matched_skills, missing_skills):
            return semantic_score, 0.0
        skill_score = self._compute_skill_score(matched_skills, missing_skills)
        return (0.6 * semantic_score) + (0.4 * skill_score), skill_score

    def _generate_match_summary(
        self,
        score: float,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> str:
        """Generate human-readable match summary."""
        parts = []

        # Overall match
        if score >= 0.8:
            parts.append("Strong match")
        elif score >= 0.6:
            parts.append("Good match")
        elif score >= 0.4:
            parts.append("Moderate match")
        else:
            parts.append("Weak match")

        parts.append(f"({score:.0%} similarity)")

        # Skill coverage
        if matched_skills:
            parts.append(f"✓ Skills: {', '.join(matched_skills[:3])}")

        if missing_skills:
            parts.append(f"✗ Missing: {', '.join(missing_skills[:3])}")

        return " | ".join(parts)
