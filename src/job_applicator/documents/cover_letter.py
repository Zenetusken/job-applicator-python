"""AI-powered cover letter generator using litellm + instructor."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from job_applicator.config import LLMConfig
from job_applicator.exceptions import LLMError
from job_applicator.models import JobListing, ResumeData, StyleGuide, UserProfile
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

logger = get_logger("documents.cover_letter")


def strip_thinking_process(text: str) -> str:
    """Remove thinking process blocks from LLM output.

    Some models (like Qwen) output their reasoning before the final answer.
    This function strips that out, leaving only the clean response.
    """
    # Strategy: Find where the actual letter content starts
    # Letters typically start with "Dear", "Hello", "To [Company]", etc.

    # First, check if there's a thinking block
    if "Thinking Process:" in text or re.match(r"^\s*\d+\.\s+\*{2}", text):
        # Look for "Final Polish:", "Final version:", or similar markers
        final_marker_pattern = r"(?:Final\s+(?:Polish|version|draft|letter)[:\s]*\n)"
        final_match = re.search(final_marker_pattern, text, re.IGNORECASE)

        if final_match:
            # Extract from after the final marker
            text = text[final_match.end() :]
        else:
            # Find the FIRST letter opening (the actual letter)
            letter_pattern = r"(?:^|\n)\s*(Dear\s|Hello\s|To\s)"
            letter_match = re.search(letter_pattern, text, re.IGNORECASE)

            if letter_match:
                text = text[letter_match.start() :]

    # Clean up
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


class CoverLetterOutput(BaseModel):
    """Structured output from LLM for cover letter generation."""

    cover_letter: str = Field(description="The generated cover letter text")
    key_points: list[str] = Field(
        description="Key points highlighted in the letter", default_factory=list
    )


SYSTEM_PROMPT = """You are a professional cover letter writer. Generate a tailored cover letter.

Guidelines:
- Be specific to the job and company
- Highlight relevant experience and skills
- Keep it concise (3-4 paragraphs)
- Use a professional but personable tone
- Do not use placeholder text like [Company Name]
- Do not repeat the entire resume
- End with a clear call to action"""


class CoverLetterGenerator:
    """Generate AI-powered cover letters via litellm + instructor."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = None
        self._style_cache: StyleGuide | None = None

    def _get_client(self) -> object:
        """Lazy-load instructor client."""
        if self._client is None:
            try:
                import instructor
                from litellm import acompletion

                self._client = instructor.from_litellm(acompletion)
            except ImportError as exc:
                raise LLMError("litellm or instructor not installed") from exc
        return self._client

    async def load_style_guide(self, style_guide_path: str) -> StyleGuide:
        """Load and analyze a style guide from a file.

        This reads an example resume/cover letter and extracts writing patterns
        using the LLM. The result is cached for subsequent calls.
        """
        if self._style_cache is not None:
            return self._style_cache

        from pathlib import Path

        path = Path(style_guide_path)
        if not path.exists():  # noqa: ASYNC240
            raise LLMError(f"Style guide not found: {path}")

        # Load the text content
        if path.suffix.lower() == ".pdf":
            from job_applicator.documents.resume import ResumeLoader

            loader = ResumeLoader()
            resume_data = loader.load(path)
            text = resume_data.raw_text
        else:
            text = path.read_text(encoding="utf-8")  # noqa: ASYNC240

        # Analyze the style
        from job_applicator.documents.style_analyzer import StyleAnalyzer

        analyzer = StyleAnalyzer(self._config)
        style = await analyzer.analyze(text)
        self._style_cache = style

        logger.info("Loaded style guide from %s: tone=%s", path.name, style.tone)
        return style

    @async_retry(max_attempts=2, base_delay=1.0, exceptions=(LLMError,))
    async def generate(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
    ) -> str:
        """Generate a cover letter for a job application.

        Args:
            job: The job listing to apply for
            user: User profile information
            resume: Parsed resume data
            style_guide: Optional style guide to mimic writing patterns
        """
        user_message = self._build_prompt(job, user, resume, style_guide)

        # For local vLLM, need "openai/" prefix
        model = f"openai/{self._config.model}" if self._config.api_base else self._config.model

        # Try instructor first (structured output)
        try:
            client = self._get_client()
            response = await client.create(  # type: ignore[attr-defined]
                model=model,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_model=CoverLetterOutput,
                max_retries=1,
            )
            letter = strip_thinking_process(response.cover_letter)
        except Exception:
            # Fallback to direct litellm call
            logger.info("Instructor failed, falling back to direct litellm call")
            try:
                from litellm import acompletion

                response = await acompletion(
                    model=model,
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                )
                letter = strip_thinking_process(response.choices[0].message.content)
            except Exception as exc:
                raise LLMError(f"LLM API call failed: {exc}") from exc

        logger.info(
            "Generated cover letter for %s at %s (%d chars)",
            job.title,
            job.company,
            len(letter),
        )
        return letter

    def _build_prompt(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
    ) -> str:
        """Build the prompt for cover letter generation."""
        parts = [
            "Write a cover letter for the following position:",
            "",
            f"Job Title: {job.title}",
            f"Company: {job.company}",
            f"Location: {job.location}",
        ]

        if job.description:
            parts.extend(["", "Job Description:", job.description])

        parts.extend(
            [
                "",
                "Applicant Profile:",
                f"Name: {user.first_name} {user.last_name}",
                f"Email: {user.email}",
            ]
        )

        if resume.summary:
            parts.extend(["", f"Summary: {resume.summary}"])

        if resume.skills:
            parts.extend(["", f"Key Skills: {', '.join(resume.skills)}"])

        # Add style guide if provided
        if style_guide:
            from job_applicator.documents.style_analyzer import StyleAnalyzer

            analyzer = StyleAnalyzer(self._config)
            style_section = analyzer.format_style_for_prompt(style_guide)
            parts.extend(["", style_section])

        parts.extend(["", "Generate a professional cover letter with key points highlighted."])

        return "\n".join(parts)

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
