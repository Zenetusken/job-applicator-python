"""AI-powered cover letter generator using litellm + instructor."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from pydantic import BaseModel, Field

from job_applicator.config import LLMConfig
from job_applicator.documents.grounding_verifier import GroundingVerifier
from job_applicator.documents.resume import ResumeLoader
from job_applicator.documents.sign_off import extract_sign_off, validate_sign_off
from job_applicator.documents.style_analyzer import StyleAnalyzer
from job_applicator.exceptions import CoverLetterGroundingError, GroundingUnavailableError, LLMError
from job_applicator.models import GroundingReport, JobListing, ResumeData, StyleGuide, UserProfile
from job_applicator.utils.language import resolve_output_language
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


class CoverLetterDraft(BaseModel):
    """A cover letter as THREE connected paragraphs. Assembled with blank lines between them so the
    3-paragraph structure is deterministic — a 4B will not reliably produce paragraph breaks in
    free prose (measured: 11/12 came out as a single wall-of-text paragraph)."""

    model_config = {"extra": "forbid"}

    opening: str = Field(
        description=(
            "First paragraph, 2-3 sentences, first person: who the applicant is, why they are "
            "moving into this field, and a specific hook tying their REAL background to this role. "
            "Plain and grounded — no literary metaphors or flourishes."
        )
    )
    body: str = Field(
        description=(
            "Second paragraph, 3-5 CONNECTED sentences (use transitions, not a list): develop ONE "
            "concrete experience or project drawn from the résumé and show how it transfers to "
            "this role. Every claim must come from the résumé. NEVER present the employer's stack, "
            "tools, or environment (a named cloud like AWS, 'cloud-native', 'Mac-first') as the "
            "applicant's own experience. Do not dump lists of acronyms."
        )
    )
    closing: str = Field(
        description=(
            "Third paragraph, 1-2 sentences: a brief, modest request to discuss the role. Do NOT "
            "promise actions on the employer's systems. Do NOT include a 'Sincerely,' sign-off or "
            "the applicant's name here — the sign-off is appended automatically."
        )
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

SYSTEM_PROMPT = f"""You are writing a cover letter as one specific real person — a thoughtful \
career-changer applying for one specific role. Write a single, connected letter that tells a \
coherent story, not a list of facts.

HOW TO WRITE IT (this matters most):
- Give the letter a through-line: who the applicant is, why they are moving into this field, and \
how their real background prepares them. Write exactly three distinct paragraphs, separated by \
blank lines, that build on each other — an opening that sets up the move, a body that develops \
the strongest relevant experience, and a short close with a genuine, modest ask to talk.
- Connect each sentence to the one before it with transitions and cause-and-effect ("which is \
why…", "that same instinct…", "after years of…", "while doing that, I learned…"). Never write a \
run of disconnected "I did X. I know Y. I can Z." statements.
- Keep each paragraph to one idea; do not grab-bag unrelated facts. Vary how sentences begin and \
run — they should not all open with "I", nor all be the same clipped length. It should sound like \
a person talking, not a résumé read back.
- Do not dump lists of tools or acronyms. Name at most one or two that matter and say what the \
applicant actually did with them.
- Use plain, professional language — direct and specific, the way a capable person actually \
talks. No literary metaphors, no scene-setting, no flowery abstractions: say what the applicant \
did, not how elegant or dramatic it felt. Plain does NOT mean choppy — keep the sentences \
connected and flowing.
- Write a focused letter — long enough to develop the experience, short enough to stay tight \
(roughly 200 to 300 words is plenty). Never pad to hit a number; every sentence earns its place. \
Plain prose only — no markdown, bullets, bold, or backticks. Avoid tired filler: {_BANNED_PHRASES}.

STAY TRUTHFUL — the résumé is the only source of facts:
- Every claim comes from the résumé. The job description is the target to address, not a list of \
things to claim about the applicant. Invent nothing — no experience, tools, metrics, problem \
domains, or qualifications the résumé does not show — and never inflate coursework or a personal \
home lab into professional operations or a strategy the applicant authored.
- Be honest about level. The applicant is early in this field: write with quiet confidence, not \
as a seasoned expert. Do not claim to have run, mastered, or be positioned to manage the \
employer's systems, and do not offer to audit, overhaul, or fix their environment — offer what \
the applicant can genuinely contribute, and do not claim they previously worked for the company.
- Do not name a specific product or vendor, or claim a credential ("certified", "accredited"), \
that the résumé does not list; name the general capability instead.
- Never mention the résumé, the job posting, or these instructions inside the letter.

FORMAT:
- Use the real company and details provided; never placeholders like [Company Name] or [Date].
- ALWAYS end with a sign-off — this is required and easy to forget. After the closing paragraph, \
write a closing word with one comma ("Sincerely,") and then the applicant's exact name. Never \
stop at the closing paragraph without this sign-off; it is the very last thing in the letter.

TONE:
- When a TONE directive is given, follow it: use its action verbs naturally, emphasize its \
themes, and match its vocabulary and sentence style.
- Otherwise, use a warm, professional, grounded voice."""


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

# Credential/status words the letter must not claim for the applicant unless the résumé states
# them (a 4B inflates in-progress coursework into "accredited cybersecurity coursework"). Mirror of
# the résumé tailor's `_strip_unearned_credentials` term list (PR #101); kept local because that
# guard is on a separate unmerged branch — consolidate to one shared constant once both land.
_CREDENTIAL_TERMS = (
    "accredited",
    "certified",
    "licensed",
    "chartered",
    "credentialed",
    "accrédité",
    "accréditée",
    "accredite",
    "accreditee",
    "certifié",
    "certifiée",
    "certifie",
    "certifiee",
    "agréé",
    "agréée",
    "agree",
    "agreee",
    "licencié",
    "licenciée",
    "licencie",
    "licenciee",
)

# Phrases that almost always describe the EMPLOYER's environment or an inflated level — never the
# applicant's grounded experience. Rejected (→ retry) when present in the letter but NOT the
# résumé. The structured body field-description reduces these, but a 4B grounds only WEAKLY against
# a JD that leads with a stack (measured: "cloud-native"/"Mac-first" persisted ~half the time), so
# this is the deterministic floor. High-precision: each phrase is an overclaim unless the résumé
# itself uses it.
_OVERCLAIM_PHRASES = (
    "cloud-native",
    "cloud native",
    "home lab",
    "mac-first",
    "self-built home lab",
    "uniquely positioned",
    "mastered",
    "compréhension approfondie",
)
_JOB_SIDE_OVERCLAIM_TERMS = (
    "workstation deployment",
    "it infrastructure maintenance",
    "asset inventory",
    "itil",
)
_SOURCE_MERGE_OVERCLAIM_PHRASES = (
    "critical systems",
    "isolate the issue",
    "listen first",
    "technical support coursework",
    "it support coursework",
)
_INSTITUTION_MARKERS = (
    "college",
    "collège",
    "institute",
    "institut",
    "school",
    "university",
    "université",
)
_CYBER_EDUCATION_TERMS = (
    "analysis & operational cybersecurity",
    "cybersecurity",
    "cybersécurité",
    "operational cybersecurity",
    "cybersécurité opérationnelle",
)


def _education_institution_names(raw_text: str) -> list[str]:
    names: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip(" •-\t")
        if not line:
            continue
        low = line.casefold()
        if not any(marker in low for marker in _INSTITUTION_MARKERS):
            continue
        parts = [part.strip() for part in re.split(r"\s+[—-]\s+", line) if part.strip()]
        candidate = parts[-1] if parts else line
        if len(candidate.split()) >= 2 and candidate not in names:
            names.append(candidate)
    return names


def _institution_fragment_supports_cyber(raw_text: str, institution: str) -> bool:
    lines = raw_text.splitlines()
    institution_low = institution.casefold()
    for index, line in enumerate(lines):
        if institution_low not in line.casefold():
            continue
        current = line.strip()
        if current.casefold().strip(".,;:") == institution_low.strip(".,;:"):
            previous = lines[index - 1] if index > 0 else ""
            fragment = f"{previous}\n{current}"
        else:
            fragment = current
        fragment_low = fragment.casefold()
        if any(term in fragment_low for term in _CYBER_EDUCATION_TERMS):
            return True
    return False


def _word_present(term: str, haystack_lower: str) -> bool:
    """Whether ``term`` (already lowercase) appears in ``haystack_lower`` not flanked by
    alphanumerics — so 'snort' wouldn't match inside 'snorting' and a multi-word product name
    matches only as a whole phrase."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack_lower) is not None


def _canonical_sign_off(language: str) -> str:
    """The sign-off word the app appends and expects, per resolved output language."""
    return "Cordialement" if language == "French" else "Sincerely"


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
        # Its OWN runtime/breaker (NOT this generator's): a flaky verifier endpoint must never
        # record failures on the breaker that guards real generation — repeated verify failures on
        # a shared breaker could trip the circuit and block the letter itself, inverting the
        # fail-safe intent (#4: a verifier problem never blocks generation). Used ONLY by
        # generate_verified() — generate() stays the pure primitive (no verifier call), so the fast
        # unit gate never fires a live LLM.
        self._verifier = GroundingVerifier(config)

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

        analyzer = StyleAnalyzer(self._config, runtime=self._runtime)
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
        self._reject_unearned_credentials(text, resume)
        self._reject_overclaim_phrases(text, resume)
        self._reject_job_side_overclaim_terms(text, resume)
        self._reject_source_merge_overclaims(text, resume)

        # A PRESENT sign-off must use the applicant's real name (catch a wrong/invented name). A
        # MISSING sign-off does NOT fail the draft — a 4B forgets the closing ~1 in 4 times, and
        # failing the whole generation over a forgotten FORMAT element is wasteful; it is appended
        # deterministically after generation (`_ensure_sign_off`).
        if extract_sign_off(text) is not None:
            validate_sign_off(text, user)

    @staticmethod
    def _ensure_sign_off(text: str, user: UserProfile, language: str = "English") -> str:
        """Guarantee exactly ONE clean sign-off in the resolved output language. Strip any closing
        the model emitted — the structured ``closing`` field often ends with an INLINE
        "Sincerely, NAME" that ``extract_sign_off`` misses, which used to double the sign-off —
        then append the canonical one ("Sincerely," in English, "Cordialement," in French). The
        applicant's name is known, so this invents nothing; anchored to the end so an adverbial
        "sincerely" mid-sentence is untouched."""
        name = f"{user.first_name} {user.last_name}".strip()
        # Bilingual: strip a model-emitted English OR French closing so the French path doesn't
        # double-sign (the English strip-list missed "Cordialement").
        closings = (
            r"sincerely|regards|best regards|kind regards|warm regards|"
            r"respectfully|yours sincerely|yours truly|"
            r"cordialement|bien cordialement|salutations distinguées|sincères salutations|"
            r"meilleures salutations|salutations|respectueusement|bien à vous"
        )
        cleaned = text.rstrip()
        # A trailing "<closing>, <the applicant's name>" (inline or on its own line), then a bare
        # trailing closing word. Both anchored to $ so only a real sign-off block is removed.
        cleaned = re.sub(
            rf"(?is)\s*\b(?:{closings})\b[,\s]+{re.escape(name)}\s*$", "", cleaned
        ).rstrip()
        cleaned = re.sub(rf"(?is)\s*\b(?:{closings})\b[,\s]*$", "", cleaned).rstrip()
        return f"{cleaned}\n\n{_canonical_sign_off(language)},\n{name}"

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
    def _split_leading_salutation(text: str) -> str:
        """Put a leading salutation on its own paragraph when the model merges it with prose."""
        stripped = text.strip()
        if "\n\n" not in stripped:
            return text
        patterns = (
            r"^(Dear\s+[^,\n]{1,80},)\s+(\S.*)$",
            r"^(Bonjour\s+[^,\n]{1,80},)\s+(\S.*)$",
            r"^(Bonjour\s+[^.\n]{1,120}\.)\s+(Je\s+\S.*)$",
        )
        for pattern in patterns:
            match = re.match(pattern, stripped, flags=re.IGNORECASE | re.DOTALL)
            if match and "\n\n" not in match.group(1):
                return f"{match.group(1)}\n\n{match.group(2).lstrip()}"
        return text

    @staticmethod
    def _reject_invented_company_employment(text: str, company: str) -> None:
        """Raise ``LLMError`` if ``text`` claims employment at ``company``.

        Only run when the company does not appear on the resume; catches common
        small-model hallucinations like "I previously worked at Acme".
        """
        norm_company = re.sub(r"[.,]", "", company.lower())
        norm_text = re.sub(r"[.,]", "", text.lower())
        # Phrases that strongly imply a past employment relationship. NOTE: a bare "my" was
        # deliberately removed from the second pattern — "my .{0,40}? <company>" false-fired on
        # ordinary cover-letter phrasing ("my interest in <company>", "my passion for <company>"),
        # which the 8B writes naturally (measured: ~45% false-reject). Real claims stay caught by
        # the "<verb> at/for <company>" and "(i|my) <verb> <company>" patterns below.
        employment_patterns = [
            rf"\b(worked|employed|tenure|time) (at|for) {re.escape(norm_company)}\b",
            rf"\b(former|previous|past|returning) .{{0,40}}? {re.escape(norm_company)}\b",
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

    @staticmethod
    def _reject_unearned_credentials(text: str, resume: ResumeData) -> None:
        """Raise ``LLMError`` if the letter describes the applicant with a credential/status word
        (accredited, certified, licensed…) the résumé does not contain.

        A small model inflates in-progress coursework into "accredited cybersecurity coursework";
        the résumé is the only valid source of a credential. Shares ``_CREDENTIAL_TERMS`` with the
        résumé tailor's guard (single source) and checks the résumé's FULL text. Feeds the existing
        validate→retry→fail-closed loop. Human review remains the backstop."""
        resume_lower = resume.raw_text.lower()
        text_lower = text.lower()
        for term in _CREDENTIAL_TERMS:
            if _word_present(term, text_lower) and not _word_present(term, resume_lower):
                raise LLMError(f"Generated cover letter claims an unearned credential: {term!r}")

    @staticmethod
    def _reject_overclaim_phrases(text: str, resume: ResumeData) -> None:
        """Raise ``LLMError`` if the letter claims an employer-environment or inflated-level phrase
        (e.g. 'cloud-native', 'Mac-first', 'mastered', 'uniquely positioned') the résumé does not
        contain. Structured generation grounds the body only weakly against a JD that leads with a
        stack, so this is the deterministic floor; feeds the validate→retry→fail-closed loop."""
        resume_lower = resume.raw_text.lower()
        text_lower = text.lower()
        for phrase in _OVERCLAIM_PHRASES:
            if _word_present(phrase, text_lower) and not _word_present(phrase, resume_lower):
                raise LLMError(
                    f"Generated cover letter makes an unearned/environment claim: {phrase!r}"
                )

    @staticmethod
    def _reject_job_side_overclaim_terms(text: str, resume: ResumeData) -> None:
        """Raise when a generic job-side requirement is written as applicant capability.

        These are not named products, but the model can still pull them from the JD and phrase them
        as what the applicant is prepared to do. If the CV is silent, retry without the term.
        """
        resume_lower = resume.raw_text.lower()
        text_lower = text.lower()
        for term in _JOB_SIDE_OVERCLAIM_TERMS:
            if _word_present(term, text_lower) and not _word_present(term, resume_lower):
                raise LLMError(f"Generated cover letter claims an unlisted job-side term: {term!r}")

    @staticmethod
    def _reject_source_merge_overclaims(text: str, resume: ResumeData) -> None:
        """Raise when separate source facts are merged into a false credential/coursework claim."""
        resume_lower = resume.raw_text.lower()
        text_lower = text.casefold()
        for phrase in _SOURCE_MERGE_OVERCLAIM_PHRASES:
            if _word_present(phrase, text_lower) and not _word_present(phrase, resume_lower):
                raise LLMError(f"Generated cover letter merges source facts into: {phrase!r}")
        if any(term in text_lower for term in _CYBER_EDUCATION_TERMS):
            for institution in _education_institution_names(resume.raw_text):
                if institution.casefold() not in text_lower:
                    continue
                if not _institution_fragment_supports_cyber(resume.raw_text, institution):
                    raise LLMError(
                        "Generated cover letter merges source facts into: "
                        f"'cybersecurity coursework at {institution}'"
                    )

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
        text = cls._split_leading_salutation(text)
        text = re.sub(
            r"\bJe suis impatient(?:e)? de discuter\b",
            "Je serais disponible pour discuter",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bdiscuter de comment\b",
            "discuter de la façon dont",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bcompréhension approfondie\b",
            "compréhension pratique",
            text,
            flags=re.IGNORECASE,
        )
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
        # Verbosity: the target is 320-380 words (a coherent letter needs room for subordinate
        # clauses — clipping it short is what made letters choppy). This sits just past the 380
        # cap, so it catches a runaway letter while leaving the coherent target range clear, and
        # stays well above any short/mocked test text.
        wc = len(text_for_sentences.split())
        if wc > 390:
            tells.append("too_long")
        # Thinness floor: catch GENUINELY thin drafts (measured: a 93-word letter) WITHOUT
        # over-firing on the 8B's natural ~200-word substance-complete length (its sweet spot —
        # forcing longer just pads). Lowered 200 -> 150 when the base model became the 8B; still
        # well above any short/mocked test text.
        if 60 < wc < 150:
            tells.append("too_short")
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
        if "too_short" in tells:
            issues.append(
                "the letter reads underdeveloped — add one more concrete, grounded example from "
                "the résumé (do not pad to hit a word count)"
            )
        if "no_short_sentences" in tells:
            issues.append(
                "vary sentence length — include at least two short sentences (under 8 words)"
            )
        if "participial_tails" in tells:
            issues.append("remove trailing '-ing' clauses; state each point as its own sentence")
        if "too_long" in tells:
            issues.append(
                "tighten toward ~350 words across three paragraphs — cut padding, not facts"
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
                        response_model=CoverLetterDraft,
                        max_retries=1,
                        max_tokens=self._config.max_tokens,
                        temperature=self._config.temperature,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    self._instructor_usable = True
                    # Assemble the three fields into paragraphs — blank lines make the 3-paragraph
                    # structure deterministic (the fix for the wall-of-text). The sign-off is
                    # appended later by `_ensure_sign_off`.
                    parts = [
                        strip_thinking_process(p).strip()
                        for p in (response.opening, response.body, response.closing)
                    ]
                    return "\n\n".join(p for p in parts if p)
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
        raw = await self._complete(user_message + correction)
        return self._humanize(raw) or raw

    async def generate(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
        tailored_resume_text: str = "",
        grounding_correction: str = "",
    ) -> str:
        """Generate a cover letter for a job application.

        Args:
            job: The job listing to apply for
            user: User profile information
            resume: Parsed resume data
            style_guide: Optional style guide to mimic writing patterns
            tone_section: Optional tone profile section to inject into the prompt
            tailored_resume_text: Optional tailored resume text as primary content source
            grounding_correction: Optional instruction (from a grounding-verifier retry) appended to
                every generation attempt, so the redraft drops the unsupported claims. Empty by
                default — ``generate`` stays the pure primitive; ``generate_verified`` supplies it.
        """
        language = resolve_output_language(self._config.language, job.description)
        logger.info(
            "Generating cover letter in %s (language setting=%s) for %s at %s",
            language,
            self._config.language,
            job.title,
            job.company,
        )

        async def _call(prev: LLMError | None) -> str:
            correction = grounding_correction + (
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
                correction=grounding_correction + correction,
            )

        letter = await self._devoice(letter, _regen, user, job=job, resume=resume)
        letter = self._ensure_sign_off(letter, user, language)

        logger.info(
            "Generated cover letter for %s at %s (%d chars)",
            job.title,
            job.company,
            len(letter),
        )
        return letter

    async def generate_verified(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
        tailored_resume_text: str = "",
    ) -> str:
        """``generate`` plus a language-agnostic grounding pass (spec §6).

        Generate the letter, verify it against the BASE résumé, and — if it claims anything the
        résumé does not support — regenerate ONCE with those claims as a correction, keeping the
        draft with STRICTLY fewer problems. A verifier false positive (its measured precision
        residual) therefore can never strip a grounded claim: a persistent flag keeps the original.
        Fail-safe (#4): a verifier failure raises a typed cover-letter error rather than returning
        an unverified draft. Callers can catch it as a requested-but-failed cover letter; real
        submission paths must not silently send it.
        """
        letter = await self.generate(
            job, user, resume, style_guide, tone_section, tailored_resume_text
        )
        try:
            report = await self._verifier.verify(letter, resume)
        except GroundingUnavailableError as exc:
            logger.info("Grounding verification skipped (verifier unavailable): %s", exc)
            raise CoverLetterGroundingError(
                f"Cover letter grounding verification unavailable: {exc}"
            ) from exc
        if report.clean:
            return letter

        def score(r: GroundingReport) -> tuple[int, int]:
            # Lexicographic: a confirmed UNSUPPORTED claim outweighs a softer coverage gap, so a
            # retry that drops a fabrication but rewords into one extra uncovered sentence still
            # wins. Equal-weighting them would discard that honesty gain (tuple compare, not sum).
            return (len(r.unsupported), len(r.coverage_gaps))

        logger.info(
            "Grounding flagged %d unsupported + %d coverage gap(s); regenerating once",
            len(report.unsupported),
            len(report.coverage_gaps),
        )
        retry = await self.generate(
            job,
            user,
            resume,
            style_guide,
            tone_section,
            tailored_resume_text,
            grounding_correction=self._grounding_correction(report),
        )
        try:
            retry_report = await self._verifier.verify(retry, resume)
        except GroundingUnavailableError:
            raise CoverLetterGroundingError(
                "Cover letter grounding verification unavailable after retry"
            ) from None
        kept, kept_report = (
            (retry, retry_report) if score(retry_report) < score(report) else (letter, report)
        )
        if not kept_report.clean:
            # F5 (fail-safe visibility): the single best-effort retry did not fully ground the
            # letter. The letter is disposable, so we still return the cleaner draft — but never
            # SILENTLY: a residual flag is logged so a shipped-but-flagged letter is observable
            # (the letter path surfaces no report to the user, unlike the résumé path).
            logger.warning(
                "Cover letter still has %d unsupported claim(s) + %d coverage gap(s) after retry "
                "(kept the cleaner draft); review before sending",
                len(kept_report.unsupported),
                len(kept_report.coverage_gaps),
            )
            raise CoverLetterGroundingError(self._grounding_failure_message(kept_report))
        return kept

    @staticmethod
    def _grounding_failure_message(report: GroundingReport) -> str:
        """Human-actionable failure summary for a non-clean grounding report."""
        message = (
            "Cover letter grounding verification found "
            f"{len(report.unsupported)} unsupported claim(s) and "
            f"{len(report.coverage_gaps)} unchecked claim(s) after retry"
        )
        unsupported = "; ".join(check.claim for check in report.unsupported[:3])
        if unsupported:
            message += f"; unsupported: {unsupported}"
        gaps = "; ".join(report.coverage_gaps[:3])
        if gaps:
            message += f"; unchecked: {gaps}"
        return message

    @staticmethod
    def _grounding_correction(report: GroundingReport) -> str:
        """A correction naming the ungrounded material, for one regeneration."""
        details: list[str] = []
        claims = "; ".join(c.claim for c in report.unsupported[:6])
        if claims:
            details.append(f"Remove or fix unsupported claims: {claims}.")
        gaps = "; ".join(report.coverage_gaps[:6])
        if gaps:
            details.append(
                "Rephrase or remove unchecked sentences that were not covered by grounded "
                f"résumé claims: {gaps}."
            )
        detail = " " + " ".join(details) if details else ""
        return (
            "\n\nIMPORTANT: some statements were not supported by the résumé. Keep ONLY what the "
            "résumé supports — invent nothing, and do not inflate any number or credential."
            + detail
        )

    async def refine_verified(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        current_text: str,
        user_feedback: str,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
    ) -> tuple[str, GroundingReport | None]:
        """``refine`` plus a grounding pass on the refined letter (spec §6). Returns the letter and
        its report (``None`` on fail-safe) so the interactive refine surfaces the same honesty check
        as the primary generate_verified path. No auto-retry — a refine is the user's explicit edit;
        the report is surfaced for the user to act on, never used to silently regenerate."""
        letter = await self.refine(
            job, user, resume, current_text, user_feedback, style_guide, tone_section
        )
        try:
            report = await self._verifier.verify(letter, resume)
        except GroundingUnavailableError as exc:
            logger.info("Cover-letter refine grounding skipped (verifier unavailable): %s", exc)
            report = None
        return letter, report

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

        refined = await self._devoice(letter, _regen, user, job=job, resume=resume)
        language = resolve_output_language(self._config.language, job.description)
        return self._ensure_sign_off(refined, user, language)

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
        language = resolve_output_language(self._config.language, job.description)
        parts = [
            f"Write a cover letter for the following position. Write the ENTIRE letter in "
            f"{language} — every sentence, including the sign-off.",
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
                "Name the target company and role naturally once as application context. "
                "Do not turn that target mention into a prior-employment claim.",
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
                f"{_canonical_sign_off(language)},\n{user.first_name} {user.last_name}",
            ]
        )

        if resume.summary:
            parts.extend(["", f"Summary: {resume.summary}"])

        if resume.skills:
            parts.extend(["", f"Key Skills: {', '.join(resume.skills)}"])

        education_context = resume.raw_text.casefold()
        if len(_education_institution_names(resume.raw_text)) >= 2 and any(
            term in education_context for term in _CYBER_EDUCATION_TERMS
        ):
            parts.extend(
                [
                    "",
                    "Education fact boundary: keep each program, credential, and coursework area "
                    "attached to the exact institution shown in the résumé. Do NOT move "
                    "cybersecurity coursework onto another degree or school.",
                ]
            )

        absent_job_side_terms = self._absent_job_side_terms(job, resume)
        if absent_job_side_terms:
            parts.extend(
                [
                    "",
                    "Do NOT mention these job-description terms as applicant capabilities because "
                    "they are absent from the résumé:",
                    ", ".join(absent_job_side_terms),
                ]
            )

        absent_source_merge_phrases = self._absent_source_merge_phrases(resume)
        if absent_source_merge_phrases:
            parts.extend(
                [
                    "",
                    "Do NOT use these phrases because they merge separate résumé facts or imply "
                    "details the résumé does not state:",
                    ", ".join(absent_source_merge_phrases),
                ]
            )

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

    @staticmethod
    def _absent_job_side_terms(job: JobListing, resume: ResumeData) -> list[str]:
        resume_lower = resume.raw_text.lower()
        job_text = f"{job.description} {' '.join(job.requirements)}".lower()
        return [
            term
            for term in _JOB_SIDE_OVERCLAIM_TERMS
            if _word_present(term, job_text) and not _word_present(term, resume_lower)
        ]

    @staticmethod
    def _absent_source_merge_phrases(resume: ResumeData) -> list[str]:
        resume_lower = resume.raw_text.lower()
        return [
            phrase
            for phrase in _SOURCE_MERGE_OVERCLAIM_PHRASES
            if not _word_present(phrase, resume_lower)
        ]
