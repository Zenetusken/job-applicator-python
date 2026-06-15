"""AI-powered cover letter generator using litellm + instructor."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from job_applicator.config import LLMConfig
from job_applicator.exceptions import LLMError
from job_applicator.models import JobListing, ResumeData, StyleGuide, UserProfile
from job_applicator.utils.llm import strip_thinking_process
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

logger = get_logger("documents.cover_letter")


class CoverLetterOutput(BaseModel):
    """Structured output from LLM for cover letter generation."""

    cover_letter: str = Field(description="The generated cover letter text")
    key_points: list[str] = Field(
        description="Key points highlighted in the letter", default_factory=list
    )


SYSTEM_PROMPT = """You are a professional cover letter writer. Generate a tailored cover letter.

Guidelines:
- Be specific to the job and company
- Highlight relevant experience and skills from the resume
- Keep it concise (3-4 paragraphs, 300-400 words)
- Do not use placeholder text like [Company Name]
- Do not repeat the entire resume
- Do not invent experience, skills, or qualifications not in the resume
- Mirror the job posting's language and terminology
- End with a clear call to action
- Write in first person ("I am excited to apply...")
- When a TONE directive is provided, follow it precisely:
  - Use the specified action verbs naturally in your writing
  - Emphasize the listed themes where relevant
  - Avoid the listed patterns entirely
  - Match the vocabulary and sentence style of the tone
- When no tone directive is provided, use a professional but personable tone

EXAMPLE — strong opening paragraph:
"I am writing to express my interest in the Help Desk Analyst position at
Acme Corp. With over five years of experience in IT support and a proven
track record of resolving complex technical issues, I am confident in my
ability to contribute to your team's commitment to exceptional customer
service."

EXAMPLE — strong closing paragraph:
"I would welcome the opportunity to discuss how my technical support
experience and ServiceNow expertise align with Acme Corp's needs. I am
available for an interview at your convenience and look forward to hearing
from you." """


class CoverLetterGenerator:
    """Generate AI-powered cover letters via litellm + instructor."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any = None
        self._style_cache: StyleGuide | None = None

    def _get_client(self) -> Any:
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
        if not await asyncio.to_thread(path.exists):
            raise LLMError(f"Style guide not found: {path}")

        # Load the text content
        if path.suffix.lower() == ".pdf":
            from job_applicator.documents.resume import ResumeLoader

            loader = ResumeLoader()
            resume_data = loader.load(path)
            text = resume_data.raw_text
        else:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")

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
        tone_section: str = "",
        tailored_resume_text: str = "",
    ) -> str:
        """Generate a cover letter for a job application.

        Args:
            job: The job listing to apply for
            user: User profile information
            resume: Parsed resume data
            style_guide: Optional style guide to mimic writing patterns
            tone_section: Optional tone profile section to inject into the prompt
            tailored_resume_text: Optional tailored resume text as primary content source
        """
        user_message = self._build_prompt(
            job,
            user,
            resume,
            style_guide,
            tone_section=tone_section,
            tailored_resume_text=tailored_resume_text,
        )

        # For local vLLM, need "openai/" prefix
        model = f"openai/{self._config.model}" if self._config.api_base else self._config.model

        # Try instructor first (structured output)
        try:
            client = self._get_client()
            response = await client.create(
                model=model,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_model=CoverLetterOutput,
                max_retries=1,
                max_tokens=self._config.max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
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
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
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

    @async_retry(max_attempts=2, base_delay=1.0, exceptions=(LLMError,))
    async def refine(
        self,
        job: JobListing,
        resume: ResumeData,
        current_text: str,
        user_feedback: str,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
    ) -> str:
        """Refine a cover letter based on user feedback.

        Uses the same structured generation pipeline as generate() —
        system prompt, style guide, tone section, instructor fallback.
        """
        parts = [
            f"Job: {job.title} at {job.company}",
            f"Location: {job.location}",
        ]
        if job.description:
            parts.extend(["", "Job Description:", job.description[:800]])
        if resume.skills:
            parts.extend(["", f"Candidate Skills: {', '.join(resume.skills)}"])
        if tone_section:
            parts.extend(["", tone_section])
        if style_guide:
            from job_applicator.documents.style_analyzer import StyleAnalyzer

            analyzer = StyleAnalyzer(self._config)
            parts.extend(["", analyzer.format_style_for_prompt(style_guide)])
        parts.extend(
            [
                "",
                "Current cover letter:",
                current_text,
                "",
                f"User feedback: {user_feedback}",
                "",
                "Apply the user's feedback and return the complete updated cover letter.",
            ]
        )
        user_message = "\n".join(parts)

        model = f"openai/{self._config.model}" if self._config.api_base else self._config.model

        try:
            client = self._get_client()
            response = await client.create(
                model=model,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_model=CoverLetterOutput,
                max_retries=1,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return strip_thinking_process(response.cover_letter)
        except Exception:
            logger.info("Instructor failed for refine, falling back to direct litellm")
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
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return strip_thinking_process(response.choices[0].message.content)

    def _build_prompt(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
        tailored_resume_text: str = "",
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

        if tone_section:
            parts.extend(["", tone_section])

        # Add style guide if provided
        if style_guide:
            from job_applicator.documents.style_analyzer import StyleAnalyzer

            analyzer = StyleAnalyzer(self._config)
            style_section = analyzer.format_style_for_prompt(style_guide)
            parts.extend(["", style_section])

        if tailored_resume_text:
            from datetime import datetime as dt

            today = dt.now().strftime("%B %d, %Y")
            parts.extend(
                [
                    "",
                    f"Today's date: {today}",
                    "",
                    "Use tailored resume as primary source for experience and skills:",
                    tailored_resume_text,
                    "",
                    "Ensure the cover letter is consistent with the tailored resume — "
                    "do not mention skills, tools, or experience absent from it.",
                    "",
                    "IMPORTANT: Use the actual date provided above. "
                    "Do NOT write '[Date]' or any placeholder — use the real date.",
                ]
            )

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
