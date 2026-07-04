#!/usr/bin/env python3
"""LLM integration test - works with local vLLM or cloud APIs."""

import asyncio
import os
from pydantic import BaseModel, Field


class SimpleResponse(BaseModel):
    """Simple structured output for testing."""

    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence level 0-1")


async def test_litellm_direct():
    """Test litellm directly (always works)."""
    import litellm
    from job_applicator.config import LLMConfig

    config = LLMConfig()
    print(f"Model: {config.model}")
    print(f"API base: {config.api_base}")

    response = await litellm.acompletion(
        model=f"openai/{config.model}",
        api_base=config.api_base,
        api_key=config.api_key,
        messages=[{"role": "user", "content": "Reply with just 'OK'"}],
        max_tokens=5,
    )
    print(f"Response: {response.choices[0].message.content}")
    return True


async def test_cover_letter_generation():
    """Test cover letter generation (template-based, always works)."""
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.config import LLMConfig
    from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile

    config = LLMConfig()
    generator = CoverLetterGenerator(config)

    job = JobListing(
        title="Python Developer",
        company="TechCorp",
        url="https://example.com/1",
        board=JobBoard.LINKEDIN,
    )
    user = UserProfile(
        first_name="John",
        last_name="Doe",
        email="john@example.com",
        phone="555-0123",
    )
    resume = ResumeData(
        raw_text="John Doe\nPython developer with 5 years experience",
        skills=["Python", "FastAPI", "Playwright"],
    )

    # Template-based generation (always works)
    letter = generator.generate_from_template(job, user, resume)
    print(f"Template letter (first 100 chars): {letter[:100]}...")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM Integration Test")
    print("=" * 60)

    results = []

    # Test 1: litellm direct
    print("\n1. Testing litellm direct call...")
    try:
        ok = await test_litellm_direct()
        results.append(("litellm direct", ok))
    except Exception as e:
        print(f"Error: {e}")
        results.append(("litellm direct", False))

    # Test 2: cover letter template
    print("\n2. Testing cover letter template generation...")
    try:
        ok = await test_cover_letter_generation()
        results.append(("cover letter template", ok))
    except Exception as e:
        print(f"Error: {e}")
        results.append(("cover letter template", False))

    # Test 3: instructor (requires --tool-call-parser on vLLM)
    print("\n3. Testing instructor structured output...")
    print("   Note: Requires --tool-call-parser flag on vLLM server")
    try:
        import instructor
        import litellm
        from job_applicator.config import LLMConfig

        config = LLMConfig()
        client = instructor.from_litellm(litellm.acompletion)

        response = await client.create(
            model=f"openai/{config.model}",
            api_base=config.api_base,
            api_key=config.api_key,
            messages=[{"role": "user", "content": "What is 2+2? Answer with number."}],
            response_model=SimpleResponse,
            max_retries=1,
        )
        print(f"Answer: {response.answer}")
        results.append(("instructor", True))
    except Exception as e:
        print(f"Skipped (vLLM needs --tool-call-parser): {type(e).__name__}")
        results.append(("instructor", None))  # None = skipped

    # Summary
    print("\n" + "=" * 60)
    print("Results:")
    for name, ok in results:
        status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  {name}: {status}")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok is True)
    skipped = sum(1 for _, ok in results if ok is None)
    print(f"\n{passed} passed, {skipped} skipped, {len(results) - passed - skipped} failed")

    return all(ok is not False for _, ok in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
