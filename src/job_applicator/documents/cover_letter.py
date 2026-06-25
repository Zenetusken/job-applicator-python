"""AI-powered cover letter generator using litellm + instructor."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from pydantic import BaseModel, Field

from job_applicator.config import LLMConfig
from job_applicator.documents.resume import ResumeLoader
from job_applicator.documents.sign_off import extract_sign_off, validate_sign_off
from job_applicator.documents.style_analyzer import StyleAnalyzer
from job_applicator.exceptions import LLMError
from job_applicator.models import JobListing, ResumeData, StyleGuide, UserProfile
from job_applicator.utils.llm import (
    CircuitOpenError,
    LLMRuntime,
    ValidatedOutput,
    llm_call_error,
    quiet_litellm,
    strip_thinking_process,
)
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry
from job_applicator.utils.text import contains_word

logger = get_logger("documents.cover_letter")


class CoverLetterOutput(BaseModel):
    """Structured output from LLM for cover letter generation."""

    model_config = {"extra": "forbid"}

    cover_letter: str = Field(description="The generated cover letter text")
    key_points: list[str] = Field(
        description="Key points highlighted in the letter", default_factory=list
    )


# Curated, high-precision AI clichés. Kept narrow on purpose: a blocklist that
# over-reaches just flags ordinary prose. SINGLE SOURCE OF TRUTH — the SYSTEM_PROMPT
# ban list is derived from this, and _voice_tells scores against it, so the
# instruction the model gets and the detector that grades it can never drift apart.
_CLICHES = (
    "proven track record",
    "i am excited to apply",
    "i am writing to express my interest",
    "drive your",
    "more than just",
    "wealth of experience",
    "passionate about",
    "team player",
    "hit the ground running",
    "look forward to hearing from you",
    "fast-paced environment",
    "do not hesitate",
    "perfect fit",
)
_BANNED_PHRASES = "; ".join(f'"{c}"' for c in _CLICHES)

SYSTEM_PROMPT = f"""You are a cover letter writer. Write a tailored cover letter that reads \
like a specific person wrote it for this specific role — not like AI-generated boilerplate.

Content rules:
- Use only experience, skills, and facts present in the resume and job description. Do not \
invent metrics, employers, problem domains, or qualifications not in the resume.
- Highlight the most relevant experience; do not restate the whole resume.
- Include exactly ONE concrete, specific detail that ties this applicant to THIS company or role.
- Keep it to 3-4 paragraphs (250-350 words), first person.
- Do not use placeholder text like [Company Name] or [Date]; use the real values provided.
- Sign the letter using the applicant's name exactly as provided in the Applicant Profile. Do not \
invent or abbreviate a different name in the signature.
- Before the sign-off, include a brief, direct call to action.

Voice rules — these are what separate human writing from AI writing, follow them strictly:
- Vary sentence length. Include at least one short sentence (under eight words). Never write \
paragraph after paragraph of uniform ~30-word sentences.
- Write plain prose only. No markdown, no bullet points, no bold, and no backticks around \
terms like asyncio or mypy.
- State accomplishments directly. Do NOT stack trailing "-ing" clauses such as \
"..., demonstrating my ability to..." or "..., ensuring...".
- Avoid cliché filler. Do NOT use any of these phrases: {_BANNED_PHRASES}.
- Be specific instead of grand: one real detail beats three superlatives.

Tone directive:
- When a TONE directive is provided, follow it precisely: use the specified action verbs \
naturally, emphasize the listed themes, avoid the listed patterns, and match its vocabulary \
and sentence style.
- When no tone directive is provided, use a professional but personable tone."""


class CoverLetterGenerator:
    """Generate AI-powered cover letters via litellm + instructor."""

    def __init__(self, config: LLMConfig, runtime: LLMRuntime | None = None) -> None:
        self._config = config
        self._client: Any = None
        # The breaker lives on a per-command runtime (built from config when not
        # injected) — shared across all cover-letter calls in this command (e.g.
        # every job in a batch run), with no module-global mutable state.
        self._runtime = runtime or LLMRuntime.defaults(name="cover-letter")
        self._breaker = self._runtime.breaker

    def _get_client(self) -> Any:
        """Lazy-load instructor client."""
        if self._client is None:
            try:
                quiet_litellm()
                import instructor
                from litellm import acompletion

                self._client = instructor.from_litellm(acompletion)
            except ImportError as exc:
                raise LLMError("litellm or instructor not installed") from exc
        return self._client

    async def load_style_guide(self, style_guide_path: str, ocr_mode: str = "auto") -> StyleGuide:
        """Load and analyze one or more style-guide files into a single StyleGuide.

        ``style_guide_path`` may be a single file or a comma-separated list. All
        files are parsed through ``ResumeLoader``, so PDFs, text files, DOCX, and
        images are supported with the same OCR fallback used for résumés. A single
        file is analyzed directly; multiple files are analyzed individually and
        merged. Per-text caching lives in ``StyleAnalyzer``, so repeated calls for
        the same path are cheap.
        """
        from pathlib import Path

        from job_applicator.exceptions import DocumentError

        paths = [p.strip() for p in style_guide_path.split(",") if p.strip()]
        if not paths:
            raise DocumentError("No style guide paths provided")

        loader = ResumeLoader()
        texts: list[str] = []
        for path_str in paths:
            path = Path(path_str)
            try:
                resume_data = loader.load(path, ocr_mode=ocr_mode)
            except DocumentError as exc:
                # Re-raise with a style-guide-specific prefix so callers know
                # which path failed without leaking raw ResumeLoader internals.
                raise DocumentError(f"Could not load style guide {path}: {exc}") from exc
            texts.append(resume_data.raw_text)

        analyzer = StyleAnalyzer(self._config)
        if len(texts) == 1:
            style = await analyzer.analyze(texts[0])
        else:
            style = await analyzer.analyze_multiple(texts)

        logger.info("Loaded style guide from %s: tone=%s", style_guide_path, style.tone)
        return style

    def _validate_output(self, text: str, user: UserProfile) -> None:
        """Validate a generated cover letter.

        Checks for empty output, placeholder text, and a valid sign-off signed
        with the applicant's name. Raises ``LLMError`` if unusable.
        """
        stripped = text.strip()
        if not stripped:
            raise LLMError("Generated cover letter is empty")
        if len(stripped) < 50:
            raise LLMError("Generated cover letter is too short")
        placeholders = r"company\s*name|hiring\s*manager|position\s*title|your\s*name|date|address"
        placeholder_pattern = rf"\[\s*(?:{placeholders})\s*\]"
        if re.search(placeholder_pattern, text, re.IGNORECASE):
            raise LLMError("Generated cover letter contains placeholder text")
        validate_sign_off(text, user)

    @staticmethod
    def _humanize(text: str) -> str:
        """Deterministically strip markdown the LLM leaks into prose.

        Conservative on purpose. Removes only formatting an applicant would never
        type in a letter: code backticks, and markdown headings/bullets anchored at
        the start of a line. Inline ``*``/``**`` are left ALONE — stripping them
        would mis-pair on a literal asterisk (``2*3`` -> ``23``) and silently
        corrupt real prose; stray emphasis is handled by the ``markdown`` voice-tell
        and re-prompt instead. Underscores survive so identifiers like
        ``get_user_id`` are untouched.

        May return an empty string for all-markdown input; callers keep the
        validated original in that case.
        """
        text = text.replace("`", "")
        text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*", "", text)  # markdown headings
        text = re.sub(r"(?m)^[ \t]*[-*+][ \t]+", "", text)  # markdown bullets (line-anchored)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _voice_tells(text: str) -> list[str]:
        """Detect robotic-writing tells. A measurement instrument (and re-prompt
        signal): the more tells, the more the draft reads as AI-generated.

        High-precision by design — conservative thresholds so short or mocked text
        does not false-positive. The sign-off block is excluded from sentence-length
        analysis so a valid closing like ``Sincerely,\\nJ D`` does not suppress the
        short-sentence tell.
        """
        tells: list[str] = []
        if "`" in text or "**" in text:
            tells.append("markdown")
        low = text.lower()
        tells.extend(f"cliche:{c}" for c in _CLICHES if c in low)

        # Exclude the trailing sign-off block from sentence analysis.
        text_for_sentences = text
        lines = text.splitlines()
        sign_off = extract_sign_off(text)
        if sign_off:
            closing_word, _signature = sign_off
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().lower().rstrip(",") == closing_word:
                    text_for_sentences = "\n".join(lines[:i])
                    break

        # Split on sentence-ending punctuation followed by whitespace or end-of-text,
        # so a decimal ("3.5 years", "99.9%") is NOT treated as a sentence boundary —
        # which would inject a phantom short fragment and suppress no_short_sentences.
        sentences = [s for s in re.split(r"[.!?]+(?:\s|$)", text_for_sentences) if s.strip()]
        if len(sentences) >= 4 and not any(len(s.split()) < 8 for s in sentences):
            tells.append("no_short_sentences")
        if len(re.findall(r",\s+\w+ing\b", text)) >= 3:
            tells.append("participial_tails")
        return tells

    @staticmethod
    def _company_in_resume(company: str, resume: ResumeData) -> bool:
        """True if the target company also appears as an employer on the resume.

        Drives an entity-collision note so the letter doesn't say "my former role
        at X" while applying to X. Normalizes case and legal suffixes; falls back
        to a normalized scan of the raw resume text when experience isn't structured.
        """

        def norm(s: str) -> str:
            s = re.sub(r"[.,]", "", s.lower())
            stripped = re.sub(r"\b(inc|llc|ltd|corp|corporation|co|company|gmbh|plc)\b", "", s)
            stripped = re.sub(r"\s+", " ", stripped).strip()
            # Keep the un-stripped form if suffix removal emptied it (a company
            # literally named "Co"/"Inc"/"Company" must not normalize to "").
            return stripped or re.sub(r"\s+", " ", s).strip()

        target = norm(company)
        if not target:
            return False
        if any(exp.company and norm(exp.company) == target for exp in resume.experience):
            return True
        # Whole-token match (utils.text.contains_word), NOT bare substring: "Ace"
        # must not match inside "marketplace" and inject a false returning-candidate note.
        return contains_word(norm(resume.raw_text), target)

    @classmethod
    def _voice_correction(cls, tells: list[str]) -> str:
        """Targeted re-prompt suffix naming the exact tells to fix.

        Specific corrective feedback ("you did X — fix X") moves a small model more
        than a general instruction it already ignored on the first pass.
        """
        issues: list[str] = []
        if "no_short_sentences" in tells:
            issues.append(
                "vary sentence length — include at least two short sentences (under 8 words)"
            )
        if "participial_tails" in tells:
            issues.append("remove trailing '-ing' clauses; state each point as its own sentence")
        if "markdown" in tells:
            issues.append("remove all markdown and backticks")
        cliches = [t.split(":", 1)[1] for t in tells if t.startswith("cliche:")]
        if cliches:
            issues.append("delete these cliché phrases entirely: " + "; ".join(cliches))
        return (
            "\n\nRevise the previous draft to fix: "
            + "; ".join(issues)
            + ". Keep the same facts and structure; change only the wording."
        )

    async def _devoice(
        self, letter: str, regen: Callable[[str], Awaitable[str]], user: UserProfile
    ) -> str:
        """Graceful, single-shot voice backstop.

        If the (already hard-validated, humanized) draft still trips ``_voice_tells``,
        re-prompt ONCE with targeted feedback and keep whichever draft has fewer
        tells. Never raises and never returns a worse draft — the original always
        stands if the retry is unusable, errors, or isn't an improvement.
        """
        tells = self._voice_tells(letter)
        if not tells:
            return letter
        try:
            retry = self._humanize(await regen(self._voice_correction(tells)))
            self._validate_output(retry, user)
        except (LLMError, CircuitOpenError):
            return letter
        return retry if len(self._voice_tells(retry)) < len(tells) else letter

    async def _complete(self, user_message: str) -> str:
        """Run ONE cover-letter completion, hardened in a single place.

        Instructor (structured) with a direct-litellm fallback, wrapped by the
        circuit breaker and a single transport-retry tier. A circuit-open
        rejection (``CircuitOpenError``) is NOT retried — retrying only re-hits
        the same open breaker. Content validation is the caller's concern
        (``ValidatedOutput``), so transport and content retries never multiply.
        """
        # For local vLLM, need "openai/" prefix
        model = f"openai/{self._config.model}" if self._config.api_base else self._config.model

        async def _one_call() -> str:
            # Try instructor first (structured output); fall back to direct litellm.
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
                    temperature=self._config.temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return strip_thinking_process(response.cover_letter)
            except Exception as exc:
                logger.info("Instructor failed (%s), falling back to direct litellm", exc)
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
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    return strip_thinking_process(response.choices[0].message.content)
                except Exception as exc2:
                    raise llm_call_error(exc2, self._config.api_base) from exc2

        @async_retry(
            max_attempts=2,
            base_delay=1.0,
            exceptions=(LLMError,),
            exclude=(CircuitOpenError,),
        )
        async def _guarded() -> str:
            return await self._breaker.call(_one_call)

        return await _guarded()

    async def _generate_raw(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None,
        tone_section: str,
        tailored_resume_text: str,
        correction: str = "",
    ) -> str:
        """Build the generation prompt and run one completion (validated by caller).

        ``correction`` is a short suffix derived from a prior validation failure,
        appended so a retry re-prompts with the rejection.
        """
        user_message = self._build_prompt(
            job,
            user,
            resume,
            style_guide,
            tone_section=tone_section,
            tailored_resume_text=tailored_resume_text,
        )
        return await self._complete(user_message + correction)

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

        async def _call(prev: LLMError | None) -> str:
            correction = (
                f"\n\nThe previous draft was rejected ({prev}). Return a corrected version."
                if prev
                else ""
            )
            return await self._generate_raw(
                job,
                user,
                resume,
                style_guide,
                tone_section,
                tailored_resume_text,
                correction=correction,
            )

        letter = await ValidatedOutput(max_retries=self._runtime.validation_max_retries).call(
            _call, partial(self._validate_output, user=user)
        )
        # Keep the validated draft if _humanize strips it to nothing (all-markdown
        # input) — never return empty past the ValidatedOutput empty-letter guard.
        letter = self._humanize(letter) or letter

        async def _regen(correction: str) -> str:
            return await self._generate_raw(
                job,
                user,
                resume,
                style_guide,
                tone_section,
                tailored_resume_text,
                correction=correction,
            )

        letter = await self._devoice(letter, _regen, user)

        logger.info(
            "Generated cover letter for %s at %s (%d chars)",
            job.title,
            job.company,
            len(letter),
        )
        return letter

    async def refine(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        current_text: str,
        user_feedback: str,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
    ) -> str:
        """Refine a cover letter based on user feedback.

        Routes through the SAME hardened pipeline as generate() — circuit
        breaker, transport retry, and output validation — via ``_complete``.
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
            parts.extend(["", StyleAnalyzer.format_style_for_prompt(style_guide)])
        parts.extend(
            [
                "",
                "Applicant Profile:",
                f"Name: {user.first_name} {user.last_name}",
                f"Email: {user.email}",
                "",
                f"Sign the refined letter as: {user.first_name} {user.last_name}",
                "",
                "The updated letter must end with a recognized sign-off line followed by "
                f"the applicant's name (e.g. 'Sincerely,\\n{user.first_name} {user.last_name}'). "
                "The sign-off must be the very last text in the letter.",
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

        async def _call(prev: LLMError | None) -> str:
            msg = user_message
            if prev:
                msg += f"\n\nThe previous draft was rejected ({prev}). Return a corrected version."
            return await self._complete(msg)

        validated = await ValidatedOutput(max_retries=self._runtime.validation_max_retries).call(
            _call, partial(self._validate_output, user=user)
        )
        letter = self._humanize(validated) or validated

        async def _regen(correction: str) -> str:
            return await self._complete(user_message + correction)

        return await self._devoice(letter, _regen, user)

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

        if self._company_in_resume(job.company, resume):
            parts.extend(
                [
                    "",
                    f"NOTE: The applicant previously worked at {job.company} (it appears on "
                    f"their resume). If relevant, acknowledge this naturally as a returning "
                    f"candidate. Do NOT describe it as 'my former role at {job.company}' as if "
                    f"it were a different employer.",
                ]
            )

        parts.extend(
            [
                "",
                "Applicant Profile:",
                f"Name: {user.first_name} {user.last_name}",
                f"Email: {user.email}",
                "",
                f"Sign the letter as: {user.first_name} {user.last_name}",
                "",
                "End the letter with a recognized sign-off line followed by the applicant's name. "
                "The sign-off must be the very last text in the letter, for example:",
                "",
                f"Sincerely,\n{user.first_name} {user.last_name}",
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
            style_section = StyleAnalyzer.format_style_for_prompt(style_guide)
            closing = style_guide.closing_style or "Sincerely"
            parts.extend(["", style_section])
            parts.extend(
                [
                    "",
                    "Follow the style guide's voice. The letter MUST still end with a "
                    f"recognized sign-off line (preferably '{closing}') followed by the "
                    "applicant's name.",
                ]
            )

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
