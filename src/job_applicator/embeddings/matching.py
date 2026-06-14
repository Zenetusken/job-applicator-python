"""Job matching using embeddings - semantic similarity between resumes and jobs."""

from __future__ import annotations

from dataclasses import dataclass

from job_applicator.config import EmbeddingConfig
from job_applicator.embeddings.service import EmbeddingService, EmbeddingVector
from job_applicator.models import JobListing, ResumeData
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
                elif stripped not in ("•", "·", "-", "ANDREI PETROV", "andre.zen799@gmail.com"):
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
            resume.skills, job.requirements, resume.raw_text
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
            matched, missing = self._match_skills(resume.skills, job.requirements, resume.raw_text)
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

    def _match_skills(
        self,
        resume_skills: list[str],
        job_requirements: list[str],
        resume_text: str = "",
    ) -> tuple[list[str], list[str]]:
        """Match resume skills to job requirements using embeddings.

        Args:
            resume_skills: Extracted skills from resume
            job_requirements: Required skills from job listing
            resume_text: Full resume text for fallback matching

        Returns:
            Tuple of (matched_skills, missing_requirements)
        """
        if not job_requirements:
            return [], []

        # Filter out invalid skills (single chars, bullets, etc.)
        valid_skills = [s for s in resume_skills if len(s.strip()) > 2 and s.strip() != "•"]

        # If no valid skills, use resume text lines as skills
        if not valid_skills and resume_text:
            # Extract potential skills from resume text
            lines = resume_text.split("\n")
            valid_skills = [
                line.strip()
                for line in lines
                if 10 < len(line.strip()) < 80
                and not line.strip().startswith(("•", "·", "-", "AND", "Experienced"))
            ]

        if not valid_skills:
            return [], list(job_requirements)

        # Compute embeddings
        skill_embs = self._service.embed_batch(valid_skills)
        req_embs = self._service.embed_batch(job_requirements)

        # Find matches using similarity threshold
        matched = []
        missing = []
        threshold = 0.55  # Lower threshold for semantic matching

        used_skills: set[str] = set()
        for i, req in enumerate(job_requirements):
            best_score = 0.0
            best_skill = ""

            for j, skill in enumerate(valid_skills):
                sim = self._service.similarity(req_embs[i], skill_embs[j])
                if sim > best_score:
                    best_score = sim
                    best_skill = skill

            if best_score >= threshold and best_skill not in used_skills:
                matched.append(best_skill)
                used_skills.add(best_skill)
            else:
                missing.append(req)

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
