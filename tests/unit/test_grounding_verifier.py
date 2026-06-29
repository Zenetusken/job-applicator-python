"""Unit tests for the deterministic grounding-audit core (the fast-gate honesty floor).

These are PURE — no LLM. They pin the evidence-check (`audit_claim`), the structural
miss-direction (`coverage_gaps`), and the combination (`audit_report`). The keystone cases are the
user's real pair: "100% first-call" (fabricated — CV says 95%) vs "100% inbound" (grounded)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.grounding_verifier import (
    GroundingVerifier,
    audit_claim,
    audit_report,
    coverage_gaps,
)
from job_applicator.exceptions import GroundingUnavailableError
from job_applicator.models import ClaimCheck, GroundingReport, ResumeData, VerificationReport

SOURCE = (
    "Took over 100% of inbound email service requests from the sales team. "
    "Maintained roughly 95% first-contact resolution without Tier 2 escalation. "
    "Technical Skills: SIEM, Wireshark, Nmap, BIND9."
)


def gc(claim: str, grounded: bool = True, quote: str = "", note: str = "") -> ClaimCheck:
    return ClaimCheck(claim=claim, grounded=grounded, source_quote=quote, note=note)


# ---- audit_claim: verifies the EVIDENCE, does not trust the model's grounded=True ----


def test_audit_catches_hallucinated_quote() -> None:
    c = gc("Managed a team of 50 engineers", quote="Led a team of 50 engineers")
    assert audit_claim(c, SOURCE) is not None


def test_audit_catches_numeric_mismatch() -> None:
    # the user's real fabrication: a 100% claim grounded by the 95% source line
    c = gc("Maintained 100% first-contact resolution", quote="roughly 95% first-contact resolution")
    reason = audit_claim(c, SOURCE)
    assert reason is not None and "percentage" in reason


def test_audit_passes_real_grounded_percentage() -> None:
    # the user's real grounded claim: 100% inbound, quoting the real line
    c = gc(
        "Took over 100% of inbound email", quote="Took over 100% of inbound email service requests"
    )
    assert audit_claim(c, SOURCE) is None


def test_audit_passes_real_short_quote() -> None:
    # a short proper-noun quote is real; BIND9's '9' is NOT treated as a percentage
    c = gc("Skilled in SIEM and BIND9", quote="Technical Skills: SIEM, Wireshark, Nmap, BIND9")
    assert audit_claim(c, SOURCE) is None


def test_audit_passes_reformatted_quote() -> None:
    # the model lightly reordered the quote — token-overlap still holds
    c = gc("95% first-contact resolution", quote="first-contact resolution roughly 95%")
    assert audit_claim(c, SOURCE) is None


def test_audit_leaves_not_grounded_to_the_model() -> None:
    # a check the model already flagged is not the audit's to override
    c = gc("Holds a CISSP", grounded=False, note="source is silent")
    assert audit_claim(c, SOURCE) is None


# ---- coverage_gaps: a fabrication the verifier never enumerated must not pass silently ----


def test_coverage_flags_unenumerated_sentence() -> None:
    generated = (
        "Maintained 95% first-contact resolution. "
        "Single-handedly architected the entire enterprise cloud security program."
    )
    claims = [gc("Maintained 95% first-contact resolution")]
    gaps = coverage_gaps(generated, claims)
    assert any("architected" in g for g in gaps)


def test_coverage_clean_when_every_sentence_enumerated() -> None:
    generated = "Maintained 95% first-contact resolution. Skilled in SIEM and Wireshark."
    claims = [
        gc("Maintained 95% first-contact resolution"),
        gc("Skilled in SIEM and Wireshark"),
    ]
    assert coverage_gaps(generated, claims) == []


# ---- audit_report: combine model flags + audit overrides + coverage ----


def test_audit_report_combines_all_signals() -> None:
    generated = (
        "Took over 100% of inbound email. "  # grounded (real)
        "Maintained 100% first-contact resolution. "  # override: grounded-by-95%
        "Holds a CISSP. "  # model-flagged unsupported
        "Ran a 200-node Kubernetes production fleet."  # never enumerated -> coverage gap
    )
    report = VerificationReport(
        claims=[
            gc(
                "Took over 100% of inbound email",
                quote="Took over 100% of inbound email service requests",
            ),
            gc(
                "Maintained 100% first-contact resolution",
                quote="roughly 95% first-contact resolution",
            ),
            gc("Holds a CISSP", grounded=False, note="source is silent"),
        ]
    )
    result = audit_report(report, generated, SOURCE)
    flagged = {u.claim for u in result.unsupported}
    assert "Maintained 100% first-contact resolution" in flagged  # audit override
    assert "Holds a CISSP" in flagged  # model flag
    assert "Took over 100% of inbound email" not in flagged  # grounded survives the audit
    assert any("Kubernetes" in g for g in result.coverage_gaps)  # coverage gap
    assert not result.complete and not result.clean


def test_grounding_report_clean_and_complete() -> None:
    assert GroundingReport().clean and GroundingReport().complete
    r = GroundingReport(coverage_gaps=["something un-enumerated"])
    assert not r.complete and not r.clean


# ---- GroundingVerifier (Slice 2): the LLM verify path is mocked; the audit is real ----

_RESUME = ResumeData(raw_text=SOURCE, skills=["SIEM", "Wireshark"])


async def test_verify_applies_the_real_audit_over_the_mocked_llm() -> None:
    verifier = GroundingVerifier(LLMConfig(model="m"))
    mocked = VerificationReport(
        claims=[
            gc(
                "Maintained 100% first-contact resolution",
                quote="roughly 95% first-contact resolution",
            ),
            gc("Skilled in SIEM", quote="Technical Skills: SIEM, Wireshark, Nmap, BIND9"),
        ]
    )
    client = MagicMock()
    client.create = AsyncMock(return_value=mocked)
    with patch.object(verifier, "_get_client", return_value=client):
        result = await verifier.verify(
            "Maintained 100% first-contact resolution. Skilled in SIEM.", _RESUME
        )
    flagged = {u.claim for u in result.unsupported}
    assert "Maintained 100% first-contact resolution" in flagged  # audit override (100 vs 95)
    assert "Skilled in SIEM" not in flagged  # grounded survives the audit


async def test_verify_failsafe_raises_never_returns_clean() -> None:
    # A verifier failure must NOT be masked as a clean document (spec §3 #4).
    verifier = GroundingVerifier(LLMConfig(model="m"))
    client = MagicMock()
    client.create = AsyncMock(side_effect=RuntimeError("endpoint down"))
    with patch.object(verifier, "_get_client", return_value=client):
        with pytest.raises(GroundingUnavailableError):
            await verifier.verify("Anything at all.", _RESUME)


async def test_verify_sources_the_base_resume_only() -> None:
    # The SOURCE handed to the model is the base résumé, never the JD or a tailored intermediate.
    verifier = GroundingVerifier(LLMConfig(model="m"))
    client = MagicMock()
    client.create = AsyncMock(return_value=VerificationReport(claims=[]))
    with patch.object(verifier, "_get_client", return_value=client):
        await verifier.verify("some generated document", _RESUME)
    sent = client.create.call_args.kwargs["messages"][1]["content"]
    assert SOURCE in sent
