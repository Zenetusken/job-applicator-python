#!/usr/bin/env python3
"""Test style guide feature - read an example and mimic its style."""

import asyncio
import os
import tempfile
from pathlib import Path

# Example of a "professional but friendly" style resume/cover letter
EXAMPLE_STYLE_DOCUMENT = """John Smith
Senior Software Engineer
john.smith@email.com | (555) 987-6543

SUMMARY
Passionate software engineer with 8 years of experience building products that matter.
I love solving complex problems and mentoring the next generation of developers.

EXPERIENCE
Senior Software Engineer | TechLeader Inc. | 2020-Present
- Shipped a real-time analytics platform serving 50M+ events daily
- Led a team of 5 engineers through a successful product launch
- Reduced infrastructure costs by 35% through smart optimization

Software Engineer | StartupXYZ | 2017-2020
- Built the core API that powered our Series A growth
- Implemented automated testing that caught 90% of bugs pre-production

SKILLS
Python, Go, Kubernetes, AWS, PostgreSQL, Redis, Team Leadership

EDUCATION
B.S. Computer Science | Stanford University | 2016
"""


async def test_style_guide():
    """Test the style guide feature."""
    print("=" * 70)
    print("STYLE GUIDE TEST")
    print("=" * 70)

    # Save example document
    descriptor, temporary_path = tempfile.mkstemp(prefix="job-applicator-style-", suffix=".txt")
    os.close(descriptor)
    example_path = Path(temporary_path)
    await asyncio.to_thread(
        example_path.write_text,
        EXAMPLE_STYLE_DOCUMENT,
        encoding="utf-8",
    )

    print("\n[1] Example document saved")
    print(f"    Path: {example_path}")
    print(f"    Length: {len(EXAMPLE_STYLE_DOCUMENT)} chars")

    # Load config
    from job_applicator.config import EmbeddingConfig, LLMConfig

    config = LLMConfig()
    print(f"\n[2] LLM Config: {config.model}")

    # Analyze style
    print("\n[3] Analyzing writing style...")
    from job_applicator.documents.style_analyzer import StyleAnalyzer

    analyzer = StyleAnalyzer(config)
    style = await analyzer.analyze(EXAMPLE_STYLE_DOCUMENT)

    print(f"    Tone: {style.tone}")
    print(f"    Sentence structure: {style.sentence_structure}")
    print(f"    Vocabulary: {style.vocabulary_level}")
    print(f"    Key phrases: {style.key_phrases[:3]}")

    # Generate cover letter with style
    print("\n[4] Generating cover letter with style guide...")
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile

    matcher = JobMatcher(EmbeddingConfig(), config)
    generator = CoverLetterGenerator(config, matcher=matcher)

    job = JobListing(
        title="Backend Engineer",
        company="InnovateTech",
        url="https://example.com/jobs/999",
        description="Looking for a backend engineer to build scalable APIs.",
        location="Remote",
        board=JobBoard.LINKEDIN,
    )

    user = UserProfile(
        first_name="Jane",
        last_name="Doe",
        email="jane.doe@email.com",
        phone="555-0199",
    )

    resume = ResumeData(
        raw_text=(
            "Jane Doe\njane.doe@email.com\n\nSUMMARY\nBackend developer.\n\n"
            "EXPERIENCE\nBackend Engineer | Example Co | 2020-Present\n"
            "- Built Python APIs for internal services.\n"
            "- Maintained PostgreSQL data pipelines.\n\n"
            "PROJECTS\n- Containerized a FastAPI service.\n\n"
            "SKILLS\nPython, FastAPI, PostgreSQL\n\n"
            "EDUCATION\nComputer Science Certificate | Example College | 2020"
        ),
        skills=["Python", "FastAPI", "PostgreSQL"],
    )

    letter = await generator.generate(job, user, resume, style)

    print("\n" + "=" * 70)
    print("GENERATED COVER LETTER (with style guide)")
    print("=" * 70)
    print(letter)
    print("=" * 70)

    # Verify style elements are present
    checks = [
        ("Has 'InnovateTech'", "InnovateTech" in letter),
        ("Has 'Jane Doe'", "Jane Doe" in letter),
        ("Length > 100 chars", len(letter) > 100),
    ]

    print("\nVerification:")
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")

    # Cleanup
    await asyncio.to_thread(example_path.unlink, missing_ok=True)

    return all(ok for _, ok in checks)


if __name__ == "__main__":
    success = asyncio.run(test_style_guide())
    exit(0 if success else 1)
