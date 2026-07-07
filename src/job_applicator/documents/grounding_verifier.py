"""Deterministic core of the grounding verifier — the pure, unit-tested honesty floor.

The LLM verifier (added in a later slice) enumerates every claim in a generated document and
quotes the SOURCE line it believes grounds each one. This module does NOT trust that judgement: it
verifies the EVIDENCE. ``audit_claim`` overrides a "grounded" verdict when the cited quote is not
really in the source (token-overlap) or when a number in the claim is absent from its quote;
``coverage_gaps`` catches the structural miss-direction — a sentence the verifier never enumerated
(a fabrication that would otherwise pass silently).

Everything here is pure + deterministic, so it runs on the FAST unit gate (what actually protects
honesty in CI), and it is language-agnostic: the SOURCE is the base résumé, so a French claim is
grounded by quoting the English source line, and the overlap checks are on shared content tokens.
"""

from __future__ import annotations

import re
from typing import Any, cast

from job_applicator.config import LLMConfig
from job_applicator.exceptions import GroundingUnavailableError, LLMError
from job_applicator.models import ClaimCheck, GroundingReport, ResumeData, VerificationReport
from job_applicator.utils.llm import LLMRuntime, litellm_model, quiet_litellm
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.grounding_verifier")

# A grounded claim's cited quote must really come from the source: at least this fraction of the
# quote's content words must appear in the source (robust to the model lightly reformatting a
# quote; a fabricated quote shares few words).
_QUOTE_OVERLAP = 0.7
# A generated sentence is "covered" when at least this fraction of its content words appear across
# the enumerated claims; below it, the verifier skipped the sentence (incomplete enumeration).
_COVERAGE_OVERLAP = 0.5

_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_PCT_RE = re.compile(r"(\d+)\s*%")
# Standalone integers (years, counts, team sizes) — NOT a digit glued to letters ('BIND9',
# 'SHA256'), which is a proper noun, not a metric. Case-insensitive boundary so 'BIND9' is excluded
# regardless of case.
_NUM_RE = re.compile(r"(?<![a-zA-Z0-9])\d+(?![a-zA-Z0-9])")
_SENT_SPLIT_RE = re.compile(r"[.!?\n]+")


def _tokens(text: str) -> set[str]:
    """Content tokens (>=3 chars, lowercased) — the unit for every overlap check here."""
    return set(_WORD_RE.findall(text.lower()))


def _normalized_text(text: str) -> str:
    """Normalize lightweight formatting so source-verbatim headings compare reliably."""
    normalized = text.lower()
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[*_`]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _pcts(text: str) -> set[str]:
    """Percentages only (e.g. '95%')."""
    return set(_PCT_RE.findall(text))


def _nums(text: str) -> set[str]:
    """Standalone integers — years, counts, team sizes (e.g. '15', '200', '50'). Excludes a digit
    glued to letters ('BIND9', 'SHA256'): a proper noun, not a metric."""
    return set(_NUM_RE.findall(text))


def _overlap(part: set[str], whole: set[str]) -> float:
    return len(part & whole) / len(part) if part else 1.0


def audit_claim(check: ClaimCheck, source: str) -> str | None:
    """Deterministically override a *grounded* verdict — does NOT trust the model's judgement.

    Returns a failure reason (so the claim is reclassified UNSUPPORTED) or ``None`` if the cited
    evidence holds up. A non-grounded check is the model's call to keep, so it returns ``None``.
    """
    if not check.grounded:
        return None
    quote_tokens = _tokens(check.source_quote)
    if quote_tokens and _overlap(quote_tokens, _tokens(source)) < _QUOTE_OVERLAP:
        return "cited quote is not in the résumé (hallucinated grounding)"
    # Numeric backstop: a number in the claim must appear in its cited quote, so '100%' cannot be
    # grounded by a '95%' line and '15 years' cannot be grounded by a '10+ years' line, even if the
    # model mis-judges. Percentages are checked as percentages AND as standalone integers — the
    # integer pass is STRICTLY ADDITIONAL (it never relaxes the percentage check; a percentage whose
    # value also appears as a bare number in the quote is still caught by the percentage pass).
    if not _pcts(check.claim) <= _pcts(check.source_quote):
        return f"claim percentage {_pcts(check.claim) or '∅'} is not in the cited quote"
    if not _nums(check.claim) <= _nums(check.source_quote):
        missing = _nums(check.claim) - _nums(check.source_quote)
        return f"claim number(s) {missing} not in the cited quote"
    return None


def _sentences(text: str) -> list[str]:
    """Substantive sentences/bullets (>=3 content tokens) of a generated document."""
    return [
        s.strip()
        for s in _SENT_SPLIT_RE.split(text)
        if len(_tokens(s)) >= 3 and not _looks_like_contact_fragment(s)
    ]


def _looks_like_contact_fragment(text: str) -> bool:
    """Contact header fragments are identity data, not factual claims to enumerate.

    Sentence splitting can chop ``name | phone | email | linkedin`` lines at ``.`` and create
    pseudo-sentences like ``com/in/name``. The verifier may reasonably skip those, so the
    deterministic coverage backstop should not report them as claim coverage gaps.
    """
    low = text.lower()
    digits = sum(ch.isdigit() for ch in text)
    return (
        "@" in text
        or "linkedin" in low
        or "/in/" in low
        or (digits >= 7 and ("|" in text or "+" in text or "(" in text))
    )


def coverage_gaps(generated: str, claims: list[ClaimCheck], source: str = "") -> list[str]:
    """Sentences of *generated* that no enumerated claim covers (token-overlap).

    The structural miss-direction: a fabrication the verifier never enumerated is neither grounded
    nor flagged, so it would pass silently. An uncovered sentence means the enumeration was
    incomplete, and the caller routes to the fail-safe path.
    """
    claim_tokens: set[str] = set()
    for check in claims:
        claim_tokens |= _tokens(check.claim)
    normalized_source = _normalized_text(source)
    return [
        s
        for s in _sentences(generated)
        if _overlap(_tokens(s), claim_tokens) < _COVERAGE_OVERLAP
        and _normalized_text(s) not in normalized_source
    ]


def audit_report(report: VerificationReport, generated: str, source: str) -> GroundingReport:
    """Apply the deterministic audit to the verifier's raw report -> the surfaced GroundingReport.

    A claim is UNSUPPORTED when the model flagged it OR when ``audit_claim`` overrides a grounded
    verdict (the override reason replaces the note). Coverage gaps are attached separately.
    """
    unsupported: list[ClaimCheck] = []
    for check in report.claims:
        if not check.grounded:
            unsupported.append(check)
        elif reason := audit_claim(check, source):
            unsupported.append(check.model_copy(update={"grounded": False, "note": reason}))
    return GroundingReport(
        unsupported=unsupported,
        coverage_gaps=coverage_gaps(generated, report.claims, source),
    )


# Grounds faithful paraphrases/translations (MEANING, not shared words) so a cross-language or
# low-overlap restatement of a real fact is not falsely flagged — paired with an explicit inflation
# + SCOPE guard, because measurement showed ANY loosening (even a translation-only clause) lets a
# scope inflation ("...the entire company" from a "...the sales team" source) slip; the guard closes
# it. Validated EN+FR against adversarial inflations (scope/number/role/tool/credential). The
# deterministic English floor (resume_tailor/cover_letter _reject_*) still double-covers the
# tool/credential vectors independently of this prompt.
VERIFIER_SYSTEM_PROMPT = (
    "You are a strict résumé fact-checker. The SOURCE is the candidate's ORIGINAL résumé — the "
    "ONLY source of truth about the candidate. The GENERATED text is a tailored résumé or cover "
    "letter to check. Enumerate EVERY substantive factual claim in GENERATED (metrics and numbers, "
    "skills, job duties, credentials, tools, experience). For each claim set grounded=true if the "
    "SOURCE supports its MEANING, and copy the EXACT VERBATIM SOURCE text that supports it into "
    "source_quote (verbatim, so it can be checked). Grounding is about MEANING, not shared words: "
    "a faithful PARAPHRASE or TRANSLATION of a SOURCE fact is grounded even if it shares FEW WORDS "
    "with the source or is written in another language — quote the SOURCE line it restates. Set "
    "grounded=false when the claim says MORE than the SOURCE: it ADDS a fact, tool, employer, or "
    "metric the source lacks; INFLATES a number; asserts a credential or experience the source "
    "does not state; or BROADENS THE SCOPE of a source fact — if the SOURCE limits something to a "
    "specific team, source, system, or period, a claim that applies it to a whole company, all "
    "departments, all systems, or everything is NOT grounded (e.g. SOURCE says a metric is for "
    "'the sales team' but the claim says 'the entire company'). A claim that restates the same "
    "fact at the same scope, or claims LESS, is grounded. When grounded=false put the reason in "
    "note and leave source_quote empty. A number is grounded only if the SAME number appears in "
    "the SOURCE for the SAME fact at the SAME scope. Coursework, an in-progress certificate, or an "
    "'exam pending' status is NOT a held credential: a claim of HOLDING a certification or "
    "qualification is grounded=false unless the SOURCE states it as completed. Do NOT set "
    "grounded=false merely because the wording or language differs. But you MUST set "
    "grounded=false for any tool, technology, employer, certification, role seniority, quantity, "
    "or SCOPE that is broader than the SOURCE. Enumerate every sentence so nothing is skipped."
)


class GroundingVerifier:
    """Language-agnostic semantic honesty layer (spec §2): the model enumerates every claim in a
    generated document and quotes the SOURCE line grounding each; the deterministic ``audit_report``
    above then overrides any grounding whose evidence does not hold up.

    Augments the deterministic English floor, never replaces it. **Fail-safe (#4):** any verifier
    failure raises ``GroundingUnavailableError`` — it never returns a clean report on failure, so a
    down endpoint can never be mistaken for an honesty-verified document.
    """

    def __init__(self, config: LLMConfig, runtime: LLMRuntime | None = None) -> None:
        self._config = config
        self._client: Any = None
        self._runtime = runtime or LLMRuntime.defaults(name="grounding-verifier")

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                quiet_litellm()
                import instructor
                from litellm import acompletion

                self._client = instructor.from_litellm(acompletion)
            except ImportError as exc:
                raise GroundingUnavailableError("litellm or instructor not installed") from exc
        return self._client

    async def verify(self, generated: str, resume: ResumeData) -> GroundingReport:
        """Verify ``generated`` against the BASE résumé (``resume.raw_text``) — never the JD or the
        tailored intermediate (spec §2). Returns the audited ``GroundingReport``, or raises
        ``GroundingUnavailableError`` (the fail-safe) when the verifier cannot run."""
        source = resume.raw_text
        model = litellm_model(self._config)
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCE (résumé):\n{source}\n\nGENERATED:\n{generated}"},
        ]

        async def _call(_prev: LLMError | None) -> VerificationReport:
            client = self._get_client()
            return cast(
                VerificationReport,
                await client.create(
                    model=model,
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    messages=messages,
                    response_model=VerificationReport,
                    max_retries=1,
                    max_tokens=self._config.max_tokens,
                    temperature=0.1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ),
            )

        try:
            report = await self._runtime.run(_call)
        except Exception as exc:
            # Fail-safe (#4): ANY failure must not pass as clean — re-raise as the typed error.
            raise GroundingUnavailableError(f"grounding verification failed: {exc}") from exc
        return audit_report(report, generated, source)
