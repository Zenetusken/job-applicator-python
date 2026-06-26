"""Job matching using embeddings - semantic similarity between resumes and jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from job_applicator.config import EmbeddingConfig
from job_applicator.embeddings.service import EmbeddingService, EmbeddingVector
from job_applicator.models import JobListing, ResumeData
from job_applicator.skills import NORMALIZATION_MAP, is_hard_negative
from job_applicator.utils.logging import get_logger

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

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._service = EmbeddingService(config)

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

    def match_resume_to_job(
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
        # Compute embeddings
        resume_emb = self.compute_resume_embedding(resume)
        job_emb = self.compute_job_embedding(job)

        # Compute semantic similarity
        semantic_score = self._service.similarity(resume_emb, job_emb)

        # Skill matching
        matched_skills, missing_skills = self._match_skills(
            resume.skills, job.requirements, resume.raw_text, job.description
        )

        # Compute skill coverage score
        skill_score = self._compute_skill_score(matched_skills, missing_skills)

        # Combined score: 60% semantic + 40% skill coverage
        score = (0.6 * semantic_score) + (0.4 * skill_score)

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

    def rank_jobs(
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

        # Compute resume embedding once
        resume_emb = self.compute_resume_embedding(resume)

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

        job_embs = self._service.embed_batch(job_texts)

        # Compute similarities with combined scoring
        matches = []
        for job, job_emb in zip(jobs, job_embs, strict=False):
            semantic_score = self._service.similarity(resume_emb, job_emb)
            matched, missing = self._match_skills(
                resume.skills, job.requirements, resume.raw_text, job.description
            )
            skill_score = self._compute_skill_score(matched, missing)

            # Combined score: 60% semantic + 40% skill coverage
            score = (0.6 * semantic_score) + (0.4 * skill_score)
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

    def _extract_requirements_from_description(self, description: str) -> list[str]:
        """Extract likely skill requirements from a job description.

        Uses the known skill-alias map conservatively: only terms that appear as
        whole words/phrases in the description are returned, and generic traits
        are filtered out. This is a fallback when a job listing has no explicit
        ``requirements`` list.
        """
        if not description:
            return []

        desc_lower = description.lower()
        found: set[str] = set()
        for term, canonical in NORMALIZATION_MAP.items():
            if is_hard_negative(canonical.lower()):
                continue
            # Check both the alias and the canonical form as whole words/phrases.
            for t in (term, canonical.lower()):
                pattern = r"(?<!\w)" + re.escape(t) + r"(?!\w)"
                if re.search(pattern, desc_lower):
                    found.add(canonical)
                    break
        return sorted(found)

    def _match_skills(
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
            job_requirements = self._extract_requirements_from_description(job_description)

        # Normalize and drop generic traits/hard negatives.
        norm_skills = [normalize_skill(s) for s in resume_skills]
        valid_skills = [
            s
            for s in norm_skills
            if len(s.strip()) > 2 and s.strip() != "•" and not is_hard_negative(s)
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

        # Compute embeddings on normalized texts
        skill_embs = self._service.embed_batch(valid_skills)
        req_texts = [n for n, _ in valid_reqs]
        req_embs = self._service.embed_batch(req_texts)

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
