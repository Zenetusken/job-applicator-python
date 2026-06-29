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
    litellm_model,
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
    "unique blend",
)
_BANNED_PHRASES = "; ".join(f'"{c}"' for c in _CLICHES)

# Showy verbs a small model reaches for that read as AI-generated in a plain cover letter. Stems,
# so 'crafted'/'crafting' match. A pile-up (>=2 distinct) is the tell — one alone can be fine.
_ORNATE_VERBS = ("conceptualiz", "operationaliz", "orchestrat", "craft", "curat", "envision")

SYSTEM_PROMPT = f"""You are a cover letter writer. Write a tailored cover letter in a natural, \
specific, personal voice — the way a real person writes for one specific role.

Content rules:
- The RESUME is the only source of the applicant's experience, skills, and facts. The job \
description tells you what the ROLE needs — it is the target to address, NOT a source of things \
to claim about the applicant. Do not invent metrics, employers, problem domains, or \
qualifications not in the resume.
- The applicant's claimed tools come from the resume only. Never name a specific vendor or \
commercial product (a named SIEM, EDR, SOAR, IDS/IPS, vulnerability scanner, or ticketing product) \
unless that exact name appears in the resume. When the role needs such a tool the resume lacks, \
name the general capability instead (the category, e.g. "SIEM monitoring", "EDR", "incident \
response") or speak to the applicant's transferable skills — never imply hands-on use of a named \
product they have not used.
- Highlight the most relevant experience; do not restate the whole resume.
- Include exactly ONE concrete detail DRAWN FROM THE RESUME — a real role, task, or project, even \
if it transfers to this role only by analogy — never a fabricated role-specific accomplishment (a \
control you tuned, a strategy you authored, an operation you ran) the resume does not contain. \
Coursework and a personal home lab are LEARNING, not professional operations: describe them as \
such ("in my coursework / home lab"), never as "I operationalized SIEM" or "I designed a security \
strategy". Do NOT claim the applicant previously worked for the company unless the resume \
explicitly lists that company as an employer.
- HUMILITY — write at the seniority the resume actually evidences. Never claim authority over, \
ownership of, or a plan to change the EMPLOYER's systems: do not say the applicant will audit, \
overhaul, manage, fix, secure, or "begin by …-ing" their environment, stack, network, or \
configurations. Never claim "mastery", "comprehensive coverage", or that the applicant is \
"uniquely positioned". For an entry or junior role the applicant is a capable beginner — eager \
and grounded, not a drop-in expert. If the role is more senior than the resume shows, offer \
transferable strengths honestly, never as equivalent experience.
- Write ONLY the letter's own content. The reader sees the letter, NOT the resume, the job \
posting, or these instructions — so never refer to any of them inside the letter. Phrases like \
"my resume", "a resume that listed", "without relying on a resume", "as the posting requires", or \
"per your requirements" must never appear; state the qualification directly instead.
- Keep it SHORT: exactly 3 paragraphs, 250-300 words MAXIMUM, first person. A long letter reads \
as padding — cut every sentence that does not earn its place; never repeat a point.
- Do not use placeholder text like [Company Name] or [Date]; use the real values provided.
- Sign the letter using the applicant's name exactly as provided in the Applicant Profile. Do not \
invent or abbreviate a different name in the signature.
- The sign-off (e.g. "Sincerely, <name>") must be the VERY LAST text in the letter, with a SINGLE \
comma after the closing word. Do not place a sign-off at the beginning or middle of the letter.
- End with a brief, modest ask — usually a request to discuss the role or the applicant's fit. Do \
NOT promise actions on the employer's systems, list tasks the applicant would perform on their \
environment, or propose a plan or timeline for their infrastructure. Avoid vague enthusiasm and \
generic "blend of qualities" summaries.

Voice rules — keep the voice natural and specific; follow them strictly:
- Vary sentence length. Include at least one short sentence (under eight words). Never write \
paragraph after paragraph of uniform ~30-word sentences.
- Vary how sentences begin AND which verbs you use. Never start more than one sentence with the \
same opening words (do not write "I envision…" / "I conceptualized…" again and again), and do not \
reuse a distinctive verb more than twice — repetition reads as canned. Avoid ornate, \
showy verbs ("craft", "curate", "conceptualize", "operationalize", "orchestrate", "architect"); \
plain verbs read naturally.
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


# Named security PRODUCTS a small model commonly pulls from a JD and wrongly attributes to the
# applicant. Deterministic REGRESSION CATCH for the SOC domain (the user's reality), NOT a general
# guard: named tools are unbounded, so this only catches known products — and is kept HIGH-PRECISION
# (no common-English words like 'soar'/'snort'/'tenable' that would wrongly reject a clean letter;
# the prompt covers those). The general signal is the matcher's `missing_skills` (tools the
# candidate lacks) — thread that through if the generator ever receives the match result (the
# `apply --from` funnel path has it; the standalone `--description` path does not). Human review
# is the backstop.
_NAMED_TOOLS: frozenset[str] = frozenset(
    {
        "arcsight",
        "cortex xdr",
        "cortex xsoar",
        "crowdstrike",
        "cylance",
        "defender for endpoint",
        "demisto",
        "exabeam",
        "fortinet",
        "jira",
        "logrhythm",
        "metasploit",
        "nessus",
        "qradar",
        "qualys",
        "rapid7",
        "securonix",
        "sentinelone",
        "servicenow",
        "sourcefire",
        "splunk",
        "suricata",
    }
)


def _word_present(term: str, haystack_lower: str) -> bool:
    """Whether ``term`` (already lowercase) appears in ``haystack_lower`` not flanked by
    alphanumerics — so 'snort' wouldn't match inside 'snorting' and a multi-word product name
    matches only as a whole phrase."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack_lower) is not None


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
        # None = unknown, True = usable, False = skip instructor (e.g. vLLM has no
        # tool-call-parser). Cached per generator instance so a failed probe is not
        # retried on every cover letter in a batch run.
        self._instructor_usable: bool | None = None

    def _get_client(self) -> Any:
        """Lazy-load instructor client.

        TOOLS mode is intentionally used: it relies on vLLM's tool-call parser
        (qwen3_xml for Qwen3.5) and is faster and more schema-accurate for this
        model than json_object or guided_json in our vLLM 0.23 tests.
        """
        if self._client is None:
            try:
                quiet_litellm()
                import instructor
                from litellm import acompletion

                # instructor defaults to TOOLS mode, which uses vLLM's tool-call
                # parser (qwen3_xml for Qwen3.5). In our vLLM 0.23 tests this is
                # faster and more schema-accurate than json_object or guided_json.
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

    def _validate_output(
        self, text: str, user: UserProfile, job: JobListing, resume: ResumeData
    ) -> None:
        """Validate a generated cover letter.

        Checks for empty output, placeholder text, hallucinated employment
        claims, and a valid sign-off signed with the applicant's name. Raises
        ``LLMError`` if unusable.
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

        # Reject invented prior-employment claims when the company is not on the
        # resume. Small models frequently hallucinate "I previously worked at X".
        if not self._company_in_resume(job.company, resume):
            self._reject_invented_company_employment(text, job.company)
        self._reject_unlisted_named_tools(text, resume, job.company)

        validate_sign_off(text, user)

    @staticmethod
    def _strip_early_sign_off(text: str) -> str:
        """Remove a sign-off line that appears before the body of the letter.

        Looks for a recognized closing word followed by the applicant's name
        (two-line or single-line) before the final sign-off block, and removes
        that stray block. Returns the text unchanged if no stray sign-off is found.
        """
        from job_applicator.documents.sign_off import _SIGN_OFF_RE, _SINGLE_LINE_SIGN_OFF_RE

        lines = text.splitlines()
        # Find the last recognized sign-off line to know where the real closing is.
        last_sign_off_idx: int | None = None
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if not stripped:
                continue
            if _SIGN_OFF_RE.match(stripped) or _SINGLE_LINE_SIGN_OFF_RE.match(stripped):
                last_sign_off_idx = i
                break
        if last_sign_off_idx is None or last_sign_off_idx == 0:
            return text

        # Scan for a stray sign-off block strictly before the real closing.
        stray_start: int | None = None
        stray_end: int | None = None
        i = 0
        while i < last_sign_off_idx - 1:
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if _SIGN_OFF_RE.match(stripped):
                # Two-line closing: sign-off line + signature line.
                if i + 1 < last_sign_off_idx and lines[i + 1].strip():
                    stray_start = i
                    stray_end = i + 2
                    break
            single = _SINGLE_LINE_SIGN_OFF_RE.match(stripped)
            if single:
                stray_start = i
                stray_end = i + 1
                break
            i += 1

        if stray_start is None:
            return text

        # Remove the stray block and collapse any excess blank lines.
        cleaned = lines[:stray_start] + lines[stray_end:]
        cleaned_text = "\n".join(cleaned)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
        return cleaned_text

    @staticmethod
    def _reject_invented_company_employment(text: str, company: str) -> None:
        """Raise ``LLMError`` if ``text`` claims employment at ``company``.

        Only run when the company does not appear on the resume; catches common
        small-model hallucinations like "I previously worked at Acme".
        """
        norm_company = re.sub(r"[.,]", "", company.lower())
        norm_text = re.sub(r"[.,]", "", text.lower())
        # Phrases that strongly imply a past employment relationship.
        employment_patterns = [
            rf"\b(worked|employed|tenure|time) (at|for) {re.escape(norm_company)}\b",
            rf"\b(former|previous|past|returning|my) .{{0,40}}? {re.escape(norm_company)}\b",
            rf"\b{re.escape(norm_company)} .{{0,40}}? (employee|colleague|team member|tenure)\b",
            rf"\b(i|my) .{{0,30}}? (built|led|worked|spent|was) .{{0,30}}? "
            rf"{re.escape(norm_company)}\b",
        ]
        for pattern in employment_patterns:
            if re.search(pattern, norm_text):
                raise LLMError(f"Generated cover letter falsely claims employment at {company}")

    @staticmethod
    def _reject_unlisted_named_tools(text: str, resume: ResumeData, company: str) -> None:
        """Raise ``LLMError`` if the letter names a known security PRODUCT not in the résumé.

        A small model often pulls a JD's named tool (Splunk, SourceFire, ServiceNow…) and
        attributes hands-on experience to the applicant; the résumé is the only valid source of a
        claimed tool. Presence is checked against the résumé's FULL text (a tool can live in an
        experience bullet, not the skills list). A tool that matches the TARGET COMPANY's name is
        allowed — naming the employer you are applying TO (e.g. a letter addressed to CrowdStrike)
        is not a skill claim; many security products ARE companies a SOC analyst applies to. SOC-
        scoped regression catch, not exhaustive — the prompt is the primary defence and human
        review the backstop. Raising feeds the existing validate→retry→fail-closed loop, so the
        model regenerates without the overclaim."""
        resume_lower = resume.raw_text.lower()
        company_lower = company.lower()
        text_lower = text.lower()
        for tool in sorted(_NAMED_TOOLS):
            if _word_present(tool, resume_lower) or _word_present(tool, company_lower):
                continue  # the candidate lists it, or it is the employer being addressed
            if _word_present(tool, text_lower):
                raise LLMError(f"Generated cover letter names a tool not in the résumé: {tool!r}")

    @classmethod
    def _humanize(cls, text: str) -> str:
        """Deterministically strip artifacts the LLM leaks into prose.

        Conservative on purpose. Removes only formatting an applicant would never
        type in a letter: code backticks, markdown headings/bullets anchored at
        the start of a line, and stray sign-off blocks that appear before the
        body. Inline ``*``/``**`` are left ALONE — stripping them would mis-pair
        on a literal asterisk (``2*3`` -> ``23``) and silently corrupt real prose;
        stray emphasis is handled by the ``markdown`` voice-tell and re-prompt
        instead. Underscores survive so identifiers like ``get_user_id`` are
        untouched.

        May return an empty string for all-markdown input; callers keep the
        validated original in that case.
        """
        text = text.replace("`", "")
        text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*", "", text)  # markdown headings
        text = re.sub(r"(?m)^[ \t]*[-*+][ \t]+", "", text)  # markdown bullets (line-anchored)
        text = cls._strip_early_sign_off(text)
        text = re.sub(
            r",{2,}", ",", text
        )  # collapse a doubled comma the model leaks ("Sincerely,,")
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
        # Verbosity: a cover letter over ~350 words reads as padded (target is 250-300). Threshold
        # well above any short/mocked test text, so it stays high-precision.
        if len(text_for_sentences.split()) > 350:
            tells.append("too_long")
        # Repetitive openings: 3+ sentences starting with the same two words ("I envision …").
        openers: dict[str, int] = {}
        for s in sentences:
            parts = s.split()
            if len(parts) >= 2:
                key = f"{parts[0]} {parts[1]}".lower()
                openers[key] = openers.get(key, 0) + 1
        if openers and max(openers.values()) >= 3:
            tells.append("repeated_openings")
        if sum(1 for stem in _ORNATE_VERBS if re.search(rf"\b{stem}", low)) >= 2:
            tells.append("ornate_verbs")
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
        # Use the full normalized company name for the raw-text scan so a bare stem
        # (e.g. "example" from "Example Corp") is not mistaken for an email domain
        # or school name. The raw text is only lower-cased and de-punctuated, not
        # stripped of legal suffixes, so "Example Corp" can still be found.
        raw_text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " ", resume.raw_text)
        full_target = re.sub(r"\s+", " ", re.sub(r"[.,]", "", company.lower())).strip()
        raw_normalized = re.sub(r"\s+", " ", re.sub(r"[.,]", "", raw_text.lower())).strip()
        return contains_word(raw_normalized, full_target)

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
        if "too_long" in tells:
            issues.append("cut it to 3 short paragraphs, 250-300 words — remove padding, not facts")
        if "repeated_openings" in tells:
            issues.append(
                "vary how sentences begin; never start several sentences with the same words"
            )
        if "ornate_verbs" in tells:
            issues.append(
                "replace showy verbs (craft, curate, conceptualize, orchestrate, envision) with "
                "plain ones (build, run, design, lead)"
            )
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
        self,
        letter: str,
        regen: Callable[[str], Awaitable[str]],
        user: UserProfile,
        job: JobListing,
        resume: ResumeData,
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
            self._validate_output(retry, user, job=job, resume=resume)
        except (LLMError, CircuitOpenError):
            return letter
        return retry if len(self._voice_tells(retry)) < len(tells) else letter

    @staticmethod
    def _is_tool_call_config_error(exc: Exception) -> bool:
        """True when the backend cannot parse tool/function calls.

        vLLM without ``--tool-call-parser`` rejects instructor's function-calling
        request with this message; treating it as a known config issue keeps the
        logs clean and lets us skip instructor on subsequent calls.
        """
        msg = str(exc).lower()
        return "tool-call-parser" in msg or "tool_call_parser" in msg

    async def _complete(self, user_message: str) -> str:
        """Run ONE cover-letter completion, hardened in a single place.

        Instructor (structured) with a direct-litellm fallback, wrapped by the
        circuit breaker and a single transport-retry tier. A circuit-open
        rejection (``CircuitOpenError``) is NOT retried — retrying only re-hits
        the same open breaker. Content validation is the caller's concern
        (``ValidatedOutput``), so transport and content retries never multiply.

        If instructor fails because the backend lacks a tool-call parser, we
        remember that and skip it for the rest of this generator's lifetime,
        falling back to direct completion without noisy logs.
        """
        model = litellm_model(self._config)

        async def _direct_litellm() -> str:
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

        async def _one_call() -> str:
            # Try instructor first (structured output); fall back to direct litellm.
            if self._instructor_usable is not False:
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
                    self._instructor_usable = True
                    return strip_thinking_process(response.cover_letter)
                except Exception as exc:
                    if self._is_tool_call_config_error(exc):
                        self._instructor_usable = False
                        logger.debug(
                            "Instructor unavailable: backend lacks a tool-call parser; "
                            "set --tool-call-parser on vLLM or use a compatible endpoint. "
                            "Falling back to direct litellm."
                        )
                    else:
                        logger.debug("Instructor failed (%s), falling back to direct litellm", exc)
            try:
                return await _direct_litellm()
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
            _call, partial(self._validate_output, user=user, job=job, resume=resume)
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

        letter = await self._devoice(letter, _regen, user, job=job, resume=resume)

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
            _call, partial(self._validate_output, user=user, job=job, resume=resume)
        )
        letter = self._humanize(validated) or validated

        async def _regen(correction: str) -> str:
            return await self._complete(user_message + correction)

        return await self._devoice(letter, _regen, user, job=job, resume=resume)

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
        else:
            parts.extend(
                [
                    "",
                    f"IMPORTANT: The applicant's resume does NOT list {job.company} as a "
                    f"previous employer. Do NOT state or imply that the applicant previously "
                    f"worked at {job.company}.",
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
