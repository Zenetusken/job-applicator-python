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
import unicodedata
from typing import Any, cast

from job_applicator.config import LLMConfig
from job_applicator.exceptions import GroundingUnavailableError, LLMError
from job_applicator.models import ClaimCheck, GroundingReport, ResumeData, VerificationReport
from job_applicator.utils.llm import (
    LLMRuntime,
    litellm_completion_kwargs,
    litellm_model,
    quiet_litellm,
)
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
_RESUME_HEADING_LABELS = frozenset(
    {
        "summary",
        "professional summary",
        "profile",
        "resume",
        "résumé",
        "profil",
        "skills",
        "technical skills",
        "core skills",
        "competences",
        "compétences",
        "experience",
        "professional experience",
        "work experience",
        "experience professionnelle",
        "expérience",
        "expérience professionnelle",
        "education",
        "education & certifications",
        "education and certifications",
        "éducation",
        "éducation & certifications",
        "formation",
        "formation et certifications",
        "certifications",
        "languages",
        "langues",
        "projects",
        "projets",
        "volunteer",
        "volunteer experience",
        "benevolat",
        "bénévolat",
        "references",
        "références",
    }
)


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


def _ascii_fold(text: str) -> str:
    """Lower-friction phrase matching across French accents."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _pcts(text: str) -> set[str]:
    """Percentages only (e.g. '95%')."""
    return set(_PCT_RE.findall(text))


def _nums(text: str) -> set[str]:
    """Standalone integers — years, counts, team sizes (e.g. '15', '200', '50'). Excludes a digit
    glued to letters ('BIND9', 'SHA256'): a proper noun, not a metric."""
    return set(_NUM_RE.findall(text))


def _source_fragments(text: str) -> list[str]:
    """Sentence/bullet-sized source fragments for supplemental evidence lookup."""
    fragments: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fragments.extend(s.strip() for s in _SENT_SPLIT_RE.split(stripped) if s.strip())
    return fragments


def _source_inventory_blocks(text: str) -> list[str]:
    """Contiguous source blocks that are clearly list/inventory sections.

    This keeps faithful multi-line skills/tool inventories from being flagged as unsupported while
    still avoiding whole-résumé token pooling across unrelated education or employment entries.
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append("\n".join(current))

    inventory_blocks: list[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        inventory_like = sum(
            1
            for line in lines
            if ":" in line or "," in line or "·" in line or ";" in line or "/" in line
        )
        if inventory_like == len(lines):
            inventory_blocks.append(block)
    return inventory_blocks


def _supplemental_percentage_evidence(check: ClaimCheck, source: str, missing: set[str]) -> str:
    """Find source fragments that carry a missing percentage for the same claim.

    The verifier can cite a real but incomplete source quote when a generated claim faithfully
    combines adjacent résumé bullets. We only supplement percentages when the source fragment has
    the exact missing percentage and overlaps the claim's content terms, so a different metric
    elsewhere in the résumé cannot rescue an inflated claim.
    """
    claim_tokens = _tokens(check.claim)
    evidence: list[str] = []
    found: set[str] = set()
    for fragment in _source_fragments(source):
        fragment_pcts = _pcts(fragment)
        if not missing & fragment_pcts:
            continue
        fragment_tokens = _tokens(fragment)
        if _overlap(fragment_tokens, claim_tokens) < 0.35:
            continue
        evidence.append(fragment)
        found |= missing & fragment_pcts
        if missing <= found:
            break
    return " ".join(evidence)


def _technical_number_supported(number: str, claim: str, source: str) -> bool:
    contexts = {
        "365": (r"\bmicrosoft\s+365\b", r"\boffice\s+365\b"),
        "802": (r"\b802\.1x\b",),
    }
    patterns = contexts.get(number, ())
    return any(
        re.search(pattern, claim, re.IGNORECASE) and re.search(pattern, source, re.IGNORECASE)
        for pattern in patterns
    )


def _source_year_supports_claim(number: str, claim: str, source: str) -> bool:
    if not (number.isdigit() and 1900 <= int(number) <= 2100):
        return False
    claim_tokens = _tokens(claim)
    for fragment in _source_fragments(source):
        if number not in _nums(fragment):
            continue
        if _overlap(_tokens(fragment), claim_tokens) >= 0.35:
            return True
    return False


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
    evidence_text = check.source_quote
    missing_percentages = _pcts(check.claim) - _pcts(evidence_text)
    if missing_percentages:
        supplemental = _supplemental_percentage_evidence(check, source, missing_percentages)
        if supplemental:
            evidence_text = f"{evidence_text} {supplemental}"
    if not _pcts(check.claim) <= _pcts(evidence_text):
        return f"claim percentage {_pcts(check.claim) or '∅'} is not in the cited quote"
    missing = _nums(check.claim) - _nums(evidence_text)
    supported_missing = {
        number
        for number in missing
        if _technical_number_supported(number, check.claim, source)
        or _source_year_supports_claim(number, check.claim, source)
    }
    if missing - supported_missing:
        missing = missing - supported_missing
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


def _looks_like_resume_heading_fragment(text: str) -> bool:
    """Known résumé labels are formatting, not factual claims to enumerate."""
    cleaned = _normalized_text(text)
    cleaned = re.sub(r"^#+\s*", "", cleaned).strip(" :")
    return cleaned in _RESUME_HEADING_LABELS


def _looks_like_application_framing(text: str) -> bool:
    """Cover-letter courtesy/target-role framing, not a candidate résumé fact."""
    low = text.lower().replace("\u2019", "'")
    return any(
        phrase in low
        for phrase in (
            "thank you for considering",
            "thank you for your consideration",
            "welcome the opportunity to discuss",
            "opportunity to discuss how my background",
            "i am applying for",
            "i am eager to bring",
            "i would welcome",
            "bonjour a l'equipe de recrutement",
            "bonjour à l'équipe de recrutement",
            "bonjour l'equipe de recrutement",
            "bonjour equipe de recrutement",
            "bonjour équipe de recrutement",
            "je vous propose ma candidature",
            "je presente ma candidature",
            "je présente ma candidature",
            "je souhaite postuler",
            "je m'appelle",
            "je serais heureux de discuter",
            "je serais ravi de discuter",
            "je serais heureux d'echanger",
            "je serais heureux d'échanger",
            "je serais ravi d'echanger",
            "je serais ravi d'échanger",
            "j'aimerais en discuter",
            "j'aimerais echanger",
            "j'aimerais échanger",
            "ce poste correspond bien",
            "correspond bien a mon objectif professionnel",
            "correspond bien à mon objectif professionnel",
            "ce poste correspond bien a mon objectif professionnel",
            "ce poste correspond bien à mon objectif professionnel",
            "mon profil correspond bien au poste",
            "je serais disponible pour discuter",
            "je suis impatient de discuter",
            "je suis impatiente de discuter",
            "ces experiences m'ont prepare a contribuer",
            "ces expériences m'ont préparé à contribuer",
            "je suis enthousiaste a l'idee de contribuer",
            "je suis enthousiaste à l'idée de contribuer",
            "mettre mes competences a profit dans ce role",
            "mettre mes compétences à profit dans ce rôle",
            "mon objectif est de mettre a profit",
            "mon objectif est de mettre à profit",
            "mettre a profit ces competences",
            "mettre à profit ces compétences",
            "je vous remercie pour votre consideration",
            "je vous remercie pour votre considération",
        )
    )


_AUDITABLE_FRAMING_TERMS = frozenset(
    {
        "aws",
        "azure",
        "cissp",
        "cloud-native",
        "cloud native",
        "edr",
        "gcp",
        "ids",
        "incident response",
        "kubernetes",
        "linux",
        "production",
        "python",
        "qradar",
        "servicenow",
        "siem",
        "splunk",
        "windows",
    }
)


def _has_auditable_framing_content(text: str) -> bool:
    folded = _ascii_fold(text).lower()
    if _pcts(text) or _nums(text):
        return True
    credential_markers = (
        "accredited",
        "accredite",
        "certified",
        "certifie",
        "chartered",
        "credentialed",
        "licensed",
        "licencie",
        "agree",
    )
    if any(marker in folded for marker in credential_markers):
        return True
    return any(term in folded for term in _AUDITABLE_FRAMING_TERMS)


def _is_pure_application_framing(text: str) -> bool:
    return _looks_like_application_framing(text) and not _has_auditable_framing_content(text)


def _unsupported_is_application_framing(check: ClaimCheck) -> bool:
    low_note = check.note.lower()
    if _is_pure_application_framing(check.claim):
        return True
    target_only_note = (
        "job title" in low_note or "company" in low_note
    ) and "not mentioned in the source" in low_note
    return target_only_note and any(
        phrase in check.claim.lower()
        for phrase in (" role at ", " position at ", "role with", "position with")
    )


def _unsupported_is_source_backed_french_security_bridge(check: ClaimCheck, source: str) -> bool:
    """Accept a measured French cover-letter bridge only when the source has the facts.

    This is narrower than application framing: it still requires source support for the security
    concepts being summarized, so a fabricated incident-response bridge cannot slip through.
    """
    claim = _ascii_fold(check.claim).lower()
    if "comprehension pratique" not in claim:
        return False
    if not all(term in claim for term in ("triage", "escalade", "incident")):
        return False

    source_norm = _ascii_fold(source).lower()
    has_security_evidence = (
        "triage" in source_norm
        and "escalat" in source_norm
        and ("incident response" in source_norm or "reponse aux incidents" in source_norm)
    )
    if not has_security_evidence:
        return False
    if "documentation" in claim and "document" not in source_norm:
        return False
    if "communication" in claim and not any(
        term in source_norm for term in ("communication", "client", "customer")
    ):
        return False
    return True


def _unsupported_is_source_verbatim(check: ClaimCheck, source: str) -> bool:
    claim = _normalized_text(check.claim).strip(" .;:")
    normalized_source = _normalized_text(source)
    return bool(claim and claim in normalized_source)


def _unsupported_is_source_token_inventory(check: ClaimCheck, source: str) -> bool:
    """Ignore a model false-negative for a long list from one source fragment.

    This deliberately does not assemble tokens across the whole résumé: that can hide cross-entry
    source merges such as attaching cybersecurity coursework to an unrelated degree.
    """
    claim_tokens = _tokens(check.claim)
    if len(claim_tokens) < 8:
        return False
    separator_count = sum(check.claim.count(sep) for sep in (",", ";", "·", "/"))
    if separator_count < 4:
        return False
    return any(
        claim_tokens <= _tokens(fragment)
        for fragment in [*_source_fragments(source), *_source_inventory_blocks(source)]
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
        and not _looks_like_resume_heading_fragment(s)
        and not _is_pure_application_framing(s)
    ]


def audit_report(report: VerificationReport, generated: str, source: str) -> GroundingReport:
    """Apply the deterministic audit to the verifier's raw report -> the surfaced GroundingReport.

    A claim is UNSUPPORTED when the model flagged it OR when ``audit_claim`` overrides a grounded
    verdict (the override reason replaces the note). Coverage gaps are attached separately.
    """
    unsupported: list[ClaimCheck] = []
    for check in report.claims:
        if _unsupported_is_application_framing(check):
            continue
        if _unsupported_is_source_backed_french_security_bridge(check, source):
            continue
        if not check.grounded and (
            _unsupported_is_source_verbatim(check, source)
            or _unsupported_is_source_token_inventory(check, source)
        ):
            continue
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
                    **litellm_completion_kwargs(self._config, temperature=0.1),
                ),
            )

        try:
            report = await self._runtime.run(_call)
        except Exception as exc:
            # Fail-safe (#4): ANY failure must not pass as clean — re-raise as the typed error.
            raise GroundingUnavailableError(f"grounding verification failed: {exc}") from exc
        return audit_report(report, generated, source)
