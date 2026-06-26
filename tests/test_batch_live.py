#!/usr/bin/env python3
"""Live integration test for batch mode."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

console = Console()


async def test_batch_live() -> bool:
    """Run a real batch pipeline end-to-end."""
    console.print("[bold cyan]LIVE BATCH MODE TEST[/]\n")

    from job_applicator.config import EmbeddingConfig, LLMConfig
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.models import JobBoard, JobListing, ResumeData

    resume = ResumeData(
        raw_text=(
            "John Doe\njohn@example.com\n555-0123\n\n"
            "Summary:\nSenior Python developer with 6 years of experience.\n\n"
            "Skills:\nPython, FastAPI, Django, PostgreSQL, Docker, AWS, Redis, asyncio\n\n"
            "Experience:\n"
            "Senior Python Developer | TechStart Inc. | 2021-Present\n"
            "- Built microservices handling 10M+ daily requests\n\n"
            "Python Developer | CodeBase Corp | 2019-2021\n"
            "- Developed RESTful APIs using Django REST Framework\n"
        ),
        name="John Doe",
        email="john@example.com",
        phone="555-0123",
        summary="Senior Python developer",
        skills=[
            "Python",
            "FastAPI",
            "Django",
            "PostgreSQL",
            "Docker",
            "AWS",
            "Redis",
            "asyncio",
        ],
    )

    jobs = [
        JobListing(
            title="Senior Python Developer",
            company="TechCorp",
            url="https://example.com/1",
            description="Python, FastAPI, AWS",
            requirements=["Python", "FastAPI", "AWS"],
            location="Remote",
            board=JobBoard.LINKEDIN,
        ),
        JobListing(
            title="Backend Engineer",
            company="StartupXYZ",
            url="https://example.com/2",
            description="Django, PostgreSQL, Docker",
            requirements=["Django", "PostgreSQL", "Docker"],
            location="SF",
            board=JobBoard.LINKEDIN,
        ),
        JobListing(
            title="Marketing Manager",
            company="AdCo",
            url="https://example.com/3",
            description="SEO, social media",
            requirements=["SEO", "Social Media"],
            location="NYC",
            board=JobBoard.LINKEDIN,
        ),
    ]

    # Step 1: Match and rank
    console.print("[bold]Step 1: Match and rank[/]")
    matcher = JobMatcher(EmbeddingConfig())
    matches = await matcher.rank_jobs(resume, jobs, top_k=3)
    for m in matches:
        console.print(f"  {m.job.title} at {m.job.company}: {m.score:.3f}")

    # Step 2: Filter by min_score
    min_score = 0.4
    filtered = [m for m in matches if m.score >= min_score]
    console.print(f"\n[bold]Step 2: Filter (min_score={min_score})[/]")
    console.print(f"  {len(filtered)}/{len(matches)} jobs above threshold")

    # Step 3: Parallel tailoring
    console.print("\n[bold]Step 3: Parallel tailoring[/]")
    tailor_engine = ResumeTailor(LLMConfig())
    sem = asyncio.Semaphore(3)

    async def tailor_one(match_result: object) -> dict[str, object]:
        async with sem:
            try:
                result = await tailor_engine.tailor(
                    resume=resume,
                    job=match_result.job,  # type: ignore[union-attr]
                    matcher=matcher,
                    user_instructions="",
                )
                return {
                    "title": match_result.job.title,  # type: ignore[union-attr]
                    "company": match_result.job.company,  # type: ignore[union-attr]
                    "match_score": round(result.match_score, 4),
                    "semantic_score": round(result.semantic_score, 4),
                    "skill_score": round(result.skill_score, 4),
                    "tailored": True,
                    "tailored_length": len(result.tailored_text),
                }
            except Exception as e:
                return {
                    "title": match_result.job.title,  # type: ignore[union-attr]
                    "company": match_result.job.company,  # type: ignore[union-attr]
                    "tailored": False,
                    "error": str(e),
                }

    start = time.monotonic()
    results = await asyncio.gather(*(tailor_one(m) for m in filtered))
    elapsed = time.monotonic() - start

    # Step 4: Verify results
    console.print(f"\n[bold]Step 4: Verify ({elapsed:.1f}s)[/]")
    checks: list[bool] = []
    for r in results:
        ok = r.get("tailored", False)
        checks.append(ok)  # type: ignore[arg-type]
        icon = "✓" if ok else "✗"
        score = r.get("match_score", 0)
        console.print(
            f"  {icon} {r['title']} at {r['company']}: "
            f"score={score}, len={r.get('tailored_length', 0)}"
        )

    # Step 5: Summary JSON
    summary = {
        "timestamp": "live_test",
        "total_jobs": len(jobs),
        "matched": len(filtered),
        "results": results,
    }
    console.print("\n[bold]Step 5: Summary JSON[/]")
    console.print(json.dumps(summary, indent=2, default=str))

    all_ok = all(checks)
    console.print(f"\n{'✓ ALL PASS' if all_ok else '✗ SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    passed = asyncio.run(test_batch_live())
    sys.exit(0 if passed else 1)
