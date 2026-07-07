#!/usr/bin/env python3
"""Live tailor workflow test — tests the full tailor pipeline with real LLM."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()


async def test_live_tailor():
    """Run a real tailor operation and verify output metadata."""
    console.print("[bold cyan]LIVE TAILOR WORKFLOW TEST[/]\n")

    from job_applicator.config import EmbeddingConfig, LLMConfig
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.models import JobBoard, JobListing, ResumeData

    # 1. Create test resume
    resume_text = """John Doe
john@example.com
555-0123

Summary:
Senior Python developer with 6 years of experience in backend systems and cloud infrastructure.

Skills:
Python, FastAPI, Django, PostgreSQL, Docker, AWS, Redis, asyncio, pytest, Kubernetes

Experience:
Senior Python Developer | TechStart Inc. | 2021-Present
- Built microservices handling 10M+ daily requests using FastAPI and asyncio
- Reduced API response times by 40% through query optimization and caching
- Managed Kubernetes clusters on AWS EKS

Python Developer | CodeBase Corp | 2019-2021
- Developed RESTful APIs using Django REST Framework
- Implemented CI/CD pipelines with GitHub Actions and Docker
"""

    resume = ResumeData(
        raw_text=resume_text,
        name="John Doe",
        email="john@example.com",
        phone="555-0123",
        summary="Senior Python developer with 6 years of experience",
        skills=[
            "Python",
            "FastAPI",
            "Django",
            "PostgreSQL",
            "Docker",
            "AWS",
            "Redis",
            "asyncio",
            "pytest",
            "Kubernetes",
        ],
    )

    # 2. Create test job
    job = JobListing(
        title="Senior Backend Engineer",
        company="CloudScale Inc.",
        url="https://example.com/jobs/99999",
        description="""
        We're looking for a Senior Backend Engineer to build scalable APIs.

        Requirements:
        - 5+ years Python experience
        - FastAPI or Django expertise
        - PostgreSQL and Redis
        - Docker and Kubernetes
        - AWS cloud infrastructure
        - CI/CD pipeline experience

        Nice to have:
        - asyncio experience
        - Monitoring and observability
        """,
        location="Remote",
        board=JobBoard.LINKEDIN,
    )

    console.print(f"Resume: {resume.name} — {len(resume.skills)} skills")
    console.print(f"Job: {job.title} at {job.company}\n")

    # 3. Compute match scores
    console.print("[bold]Step 1: Computing match scores...[/]")
    embedding_config = EmbeddingConfig()
    matcher = JobMatcher(embedding_config)
    match_result = await matcher.match_resume_to_job(resume, job)
    console.print(f"  Match score: {match_result.score:.3f}")
    console.print(f"  Matched skills: {match_result.matched_skills}")
    console.print(f"  Missing skills: {match_result.missing_skills}")

    # 4. Run tailor
    console.print("\n[bold]Step 2: Running tailor with LLM...[/]")
    llm_config = LLMConfig()
    tailor = ResumeTailor(llm_config)

    result = await tailor.tailor(
        resume=resume,
        job=job,
        user_instructions="Emphasize cloud and scalability experience.",
        matcher=matcher,
        match_result=match_result,
    )

    # 5. Verify output
    console.print("\n[bold]Step 3: Verifying output...[/]")
    checks = []

    # A7: prompt_version
    checks.append(("prompt_version present", hasattr(result, "prompt_version")))
    checks.append(("prompt_version == '1.0'", getattr(result, "prompt_version", None) == "1.0"))

    # D7: semantic_score / skill_score
    checks.append(("semantic_score > 0", getattr(result, "semantic_score", 0) > 0))
    checks.append(("skill_score > 0", getattr(result, "skill_score", 0) > 0))
    checks.append(("match_score > 0", getattr(result, "match_score", 0) > 0))

    # Check that scores decompose correctly: match = 0.6*semantic + 0.4*skill
    # (the combined-scoring weights in embeddings/matching.py), NOT a raw sum.
    sem = getattr(result, "semantic_score", 0)
    sk = getattr(result, "skill_score", 0)
    total = getattr(result, "match_score", 0)
    checks.append(("weighted scores ≈ match_score", abs((0.6 * sem + 0.4 * sk) - total) < 0.01))

    # F4: seniority detection
    from job_applicator.models import detect_seniority

    seniority = detect_seniority(job.title)
    checks.append(("seniority detected for job", seniority is not None))

    # Tailored text checks
    tailored = result.tailored_text if hasattr(result, "tailored_text") else ""
    checks.append(("tailored text not empty", len(tailored) > 100))
    checks.append(("tailored text differs from original", tailored != resume_text))

    # Report
    all_pass = True
    for name, passed in checks:
        icon = "✓" if passed else "✗"
        detail = ""
        if "prompt_version" in name:
            detail = f" (got={getattr(result, 'prompt_version', 'MISSING')})"
        elif "semantic_score" in name:
            detail = f" (got={sem:.3f})"
        elif "skill_score" in name:
            detail = f" (got={sk:.3f})"
        elif "match_score" in name:
            detail = f" (got={total:.3f})"
        elif "sum" in name:
            detail = f" (sem={sem:.3f} + sk={sk:.3f} = {sem + sk:.3f} vs total={total:.3f})"
        elif "seniority" in name:
            detail = f" (got={seniority})"
        console.print(f"  {icon} {name}{detail}")
        if not passed:
            all_pass = False

    # Print sample of tailored text
    console.print("\n[bold]Tailored text sample (first 300 chars):[/]")
    console.print(tailored[:300] + "...")

    # Save output
    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "live_test_tailored.txt"
    out_path.write_text(tailored)
    console.print(f"\nSaved to: {out_path}")

    # Print full metadata
    console.print("\n[bold]Full metadata:[/]")
    meta = {
        "prompt_version": result.prompt_version,
        "match_score": result.match_score,
        "semantic_score": result.semantic_score,
        "skill_score": result.skill_score,
        "matched_skills": result.matched_skills,
        "missing_skills": result.missing_skills,
        "attempt": result.attempt,
    }
    console.print(json.dumps(meta, indent=2))

    return all_pass


if __name__ == "__main__":
    passed = asyncio.run(test_live_tailor())
    sys.exit(0 if passed else 1)
