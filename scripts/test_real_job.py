#!/usr/bin/env python3
"""Test cover letter generation with a real job description."""

import asyncio
from job_applicator.config import LLMConfig
from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile


# Real job description (simplified from a real posting)
REAL_JOB = JobListing(
    title="Senior Python Developer",
    company="TechCorp Solutions",
    url="https://example.com/jobs/12345",
    description="""
We are looking for a Senior Python Developer to join our growing team. 

Responsibilities:
- Design and implement scalable backend services using Python
- Build and maintain RESTful APIs using FastAPI or Django
- Write clean, testable, and well-documented code
- Collaborate with cross-functional teams to define technical requirements
- Mentor junior developers and participate in code reviews

Requirements:
- 5+ years of professional Python development experience
- Strong experience with FastAPI, Django, or Flask
- Experience with PostgreSQL or similar relational databases
- Familiarity with Docker and containerization
- Experience with AWS or GCP cloud services
- Strong understanding of software design patterns

Nice to have:
- Experience with async programming (asyncio)
- Knowledge of message queues (Redis, RabbitMQ)
- Experience with CI/CD pipelines
- Contributions to open source projects
""",
    location="Remote (US/EU)",
    board=JobBoard.LINKEDIN,
)


# Sample resume data
SAMPLE_RESUME = ResumeData(
    raw_text="""
John Doe
john.doe@email.com
555-0123

Summary:
Experienced Python developer with 6 years of expertise in building scalable backend systems. 
Passionate about clean code and mentorship.

Skills:
Python, FastAPI, Django, PostgreSQL, Docker, AWS, Git, Redis, asyncio, pytest

Experience:
Senior Python Developer | TechStart Inc. | 2021-Present
- Built microservices handling 10M+ daily requests using FastAPI
- Reduced API response times by 40% through optimization
- Mentored team of 3 junior developers

Python Developer | CodeBase Corp | 2019-2021
- Developed RESTful APIs using Django REST Framework
- Implemented CI/CD pipelines using GitHub Actions
- Wrote comprehensive test suites with 95% coverage
""",
    name="John Doe",
    email="john.doe@email.com",
    phone="555-0123",
    summary="Experienced Python developer with 6 years of expertise in building scalable backend systems.",
    skills=["Python", "FastAPI", "Django", "PostgreSQL", "Docker", "AWS", "Redis", "asyncio"],
)


SAMPLE_USER = UserProfile(
    first_name="John",
    last_name="Doe",
    email="john.doe@email.com",
    phone="555-0123",
)


async def test_cover_letter_with_llm():
    """Generate a cover letter using the LLM."""
    from litellm import acompletion

    config = LLMConfig()
    print(f"Using model: {config.model}")
    print(f"API base: {config.api_base}")

    # Build the prompt
    prompt = f"""Write a professional cover letter for this job application:

Job Title: {REAL_JOB.title}
Company: {REAL_JOB.company}
Location: {REAL_JOB.location}

Job Description:
{REAL_JOB.description}

Applicant Profile:
Name: {SAMPLE_USER.first_name} {SAMPLE_USER.last_name}
Email: {SAMPLE_USER.email}
Summary: {SAMPLE_RESUME.summary}
Skills: {", ".join(SAMPLE_RESUME.skills)}

Generate a compelling cover letter (3-4 paragraphs) that:
1. Opens with enthusiasm for the role
2. Highlights relevant experience matching the job requirements
3. Shows knowledge of the company/role
4. Closes with a call to action

Keep it professional but personable."""

    print("\nGenerating cover letter...")
    response = await acompletion(
        model=f"openai/{config.model}",
        api_base=config.api_base,
        api_key=config.api_key,
        messages=[
            {
                "role": "system",
                "content": "You are a professional cover letter writer. Write concise, compelling cover letters. Do not include thinking process or reasoning steps in your output.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
    )

    cover_letter = response.choices[0].message.content

    # Strip thinking process if present
    from job_applicator.documents.cover_letter import strip_thinking_process

    cover_letter = strip_thinking_process(cover_letter)

    return cover_letter


async def main():
    """Run the test."""
    print("=" * 70)
    print("Cover Letter Generation Test - Real Job Description")
    print("=" * 70)

    print(f"\nJob: {REAL_JOB.title} at {REAL_JOB.company}")
    print(f"Location: {REAL_JOB.location}")

    print(f"\nApplicant: {SAMPLE_USER.first_name} {SAMPLE_USER.last_name}")
    print(f"Skills: {', '.join(SAMPLE_RESUME.skills[:5])}...")

    try:
        cover_letter = await test_cover_letter_with_llm()

        print("\n" + "=" * 70)
        print("GENERATED COVER LETTER")
        print("=" * 70)
        print(cover_letter)
        print("=" * 70)

        # Verify it mentions key elements
        checks = [
            ("Company name", REAL_JOB.company in cover_letter),
            ("Job title", "Python" in cover_letter),
            ("Applicant name", SAMPLE_USER.first_name in cover_letter),
            (
                "Skills mentioned",
                any(skill in cover_letter for skill in ["FastAPI", "Python", "backend"]),
            ),
        ]

        print("\nVerification:")
        for name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")

        all_passed = all(ok for _, ok in checks)
        print(f"\n{'All checks passed!' if all_passed else 'Some checks failed'}")

        return all_passed

    except Exception as e:
        print(f"\nError: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
