"""Deterministic core of the grounding verifier — the pure, unit-tested honesty floor.

The LLM verifier (added in a later slice) enumerates every claim in a generated document and
quotes the SOURCE line it believes grounds each one. This module does NOT trust that judgement: it
verifies the EVIDENCE. ``audit_claim`` overrides a "grounded" verdict when the cited quote is not
really in the source (token-overlap) or when a percentage in the claim is absent from its quote;
``coverage_gaps`` catches the structural miss-direction — a sentence the verifier never enumerated
(a fabrication that would otherwise pass silently).

Everything here is pure + deterministic, so it runs on the FAST unit gate (what actually protects
honesty in CI), and it is language-agnostic: the SOURCE is the base résumé, so a French claim is
grounded by quoting the English source line, and the overlap checks are on shared content tokens.
"""

from __future__ import annotations

import re

from job_applicator.models import ClaimCheck, GroundingReport, VerificationReport

# A grounded claim's cited quote must really come from the source: at least this fraction of the
# quote's content words must appear in the source (robust to the model lightly reformatting a
# quote; a fabricated quote shares few words).
_QUOTE_OVERLAP = 0.7
# A generated sentence is "covered" when at least this fraction of its content words appear across
# the enumerated claims; below it, the verifier skipped the sentence (incomplete enumeration).
_COVERAGE_OVERLAP = 0.5

_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_PCT_RE = re.compile(r"(\d+)\s*%")
_SENT_SPLIT_RE = re.compile(r"[.!?\n]+")


def _tokens(text: str) -> set[str]:
    """Content tokens (>=3 chars, lowercased) — the unit for every overlap check here."""
    return set(_WORD_RE.findall(text.lower()))


def _pcts(text: str) -> set[str]:
    """Percentages only (a digit inside a proper noun like 'BIND9' is NOT a metric)."""
    return set(_PCT_RE.findall(text))


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
    # Numeric backstop: a percentage in the claim must appear in its cited quote, so '100%' cannot
    # be grounded by a '95%' line even if the model mis-judges.
    if not _pcts(check.claim) <= _pcts(check.source_quote):
        return f"claim percentage {_pcts(check.claim) or '∅'} is not in the cited quote"
    return None


def _sentences(text: str) -> list[str]:
    """Substantive sentences/bullets (>=3 content tokens) of a generated document."""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(_tokens(s)) >= 3]


def coverage_gaps(generated: str, claims: list[ClaimCheck]) -> list[str]:
    """Sentences of *generated* that no enumerated claim covers (token-overlap).

    The structural miss-direction: a fabrication the verifier never enumerated is neither grounded
    nor flagged, so it would pass silently. An uncovered sentence means the enumeration was
    incomplete, and the caller routes to the fail-safe path.
    """
    claim_tokens: set[str] = set()
    for check in claims:
        claim_tokens |= _tokens(check.claim)
    return [
        s for s in _sentences(generated) if _overlap(_tokens(s), claim_tokens) < _COVERAGE_OVERLAP
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
        coverage_gaps=coverage_gaps(generated, report.claims),
    )
