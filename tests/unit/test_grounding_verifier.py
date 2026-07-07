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


def test_audit_repairs_incomplete_metric_quote_from_source_fragment() -> None:
    # The verifier sometimes cites the non-metric half of a faithful combined claim. The audit may
    # supplement ONLY with the source fragment carrying the same percentage and overlapping terms.
    source = (
        "Delivered Tier 1 technical support by phone, chat, and email, diagnosing and remotely "
        "resolving signal, receiver, connectivity, and website issues.\n"
        "Maintained roughly 95% first-contact resolution without Tier 2 escalation."
    )
    c = gc(
        "Delivered Tier 1 technical support by phone, chat, and email, diagnosing and resolving "
        "signal, receiver, connectivity, and website issues with a 95% first-contact resolution "
        "rate.",
        quote=(
            "Delivered Tier 1 technical support by phone, chat, and email, diagnosing and remotely "
            "resolving signal, receiver, connectivity, and website issues"
        ),
    )
    assert audit_claim(c, source) is None


def test_audit_does_not_repair_percentage_from_unrelated_source_fragment() -> None:
    source = (
        "Took over 100% of inbound email service requests from the sales team. "
        "Maintained roughly 95% first-contact resolution without Tier 2 escalation."
    )
    c = gc(
        "Maintained 100% first-contact resolution",
        quote="Maintained roughly 95% first-contact resolution",
    )
    reason = audit_claim(c, source)
    assert reason is not None and "percentage" in reason


def test_audit_leaves_not_grounded_to_the_model() -> None:
    # a check the model already flagged is not the audit's to override
    c = gc("Holds a CISSP", grounded=False, note="source is silent")
    assert audit_claim(c, SOURCE) is None


def test_audit_catches_fabricated_non_percentage_number() -> None:
    # F3: the numeric backstop covers standalone integers (years/counts/team sizes), not just
    # percentages — '10+ years' grounds '10', not the fabricated '15'. The user's claims are
    # metric-heavy, so a number absent from its quote must be caught even without a '%'.
    src = "10+ years of experience in IT support and operations."
    c = gc("15 years of experience in IT support", quote="10+ years of experience in IT support")
    reason = audit_claim(c, src)
    assert reason is not None and "number" in reason


def test_audit_excludes_digit_glued_to_letters() -> None:
    # F3 boundary: 'BIND9'/'SHA256' are proper nouns, not metrics — their digits never trip the
    # numeric backstop, so a real tool name is not mistaken for a fabricated count.
    src = "Technical Skills: BIND9, SHA256 hashing, Nmap."
    c = gc("Configured BIND9 and SHA256 hashing", quote="Technical Skills: BIND9, SHA256 hashing")
    assert audit_claim(c, src) is None


def test_audit_passes_faithful_integer_match() -> None:
    # F3 precision: a faithful claim whose integer DOES appear in the quote stays grounded (the
    # backstop only fires on a number ABSENT from its quote).
    src = "Resolved 200 tickets across a team of 5 analysts."
    c = gc("Resolved 200 tickets", quote="Resolved 200 tickets across a team of 5 analysts")
    assert audit_claim(c, src) is None


def test_audit_allows_supported_numeric_skill_names_when_quote_is_broad() -> None:
    src = "TECHNICAL SKILLS\nMicrosoft 365 · 802.1X · IPv6"
    c = gc(
        "Skilled in Microsoft 365 and 802.1X",
        quote="TECHNICAL SKILLS",
    )

    assert audit_claim(c, src) is None


def test_audit_allows_source_backed_heading_years_when_quote_omits_heading() -> None:
    src = (
        "Customer Service Manager 2022 – 2025\n"
        "• Took over 100% of inbound email service requests from the sales team"
    )
    c = gc(
        "Customer Service Manager 2022 – 2025: Took over 100% of inbound email service requests",
        quote="Took over 100% of inbound email service requests from the sales team",
    )

    assert audit_claim(c, src) is None


# ---- coverage_gaps: a fabrication the verifier never enumerated must not pass silently ----


def test_coverage_flags_unenumerated_sentence() -> None:
    generated = (
        "Maintained 95% first-contact resolution. "
        "Single-handedly architected the entire enterprise cloud security program."
    )
    claims = [gc("Maintained 95% first-contact resolution")]
    gaps = coverage_gaps(generated, claims)
    assert any("architected" in g for g in gaps)


def test_coverage_ignores_contact_header_fragments() -> None:
    generated = (
        "Jane Doe\n"
        "Montreal, QC | (514) 555-0199 | jane@example.com | linkedin.com/in/jane-doe\n"
        "Single-handedly architected the entire enterprise cloud security program."
    )
    gaps = coverage_gaps(generated, [])

    assert not any("514" in g or "linkedin" in g for g in gaps)
    assert any("architected" in g for g in gaps)


def test_coverage_clean_when_every_sentence_enumerated() -> None:
    generated = "Maintained 95% first-contact resolution. Skilled in SIEM and Wireshark."
    claims = [
        gc("Maintained 95% first-contact resolution"),
        gc("Skilled in SIEM and Wireshark"),
    ]
    assert coverage_gaps(generated, claims) == []


def test_coverage_ignores_source_verbatim_role_heading() -> None:
    source = "Staff Engineer, Acme Data (2021-Present)"
    generated = "**Staff Engineer, Acme Data** (2021–Present)"

    assert coverage_gaps(generated, [], source) == []


def test_coverage_ignores_cover_letter_courtesy_sentence() -> None:
    generated = "Thank you for considering my application."

    assert coverage_gaps(generated, [], SOURCE) == []


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


def test_audit_report_ignores_cover_letter_application_framing() -> None:
    generated = (
        "I would welcome the opportunity to discuss how my background and skills align with "
        "the IT On-site Support Technician role at WSP. Holds a CISSP."
    )
    report = VerificationReport(
        claims=[
            gc(
                "I would welcome the opportunity to discuss how my background and skills align "
                "with the IT On-site Support Technician role at WSP.",
                grounded=False,
                note="cited quote is not in the résumé",
            ),
            gc("Holds a CISSP", grounded=False, note="source is silent"),
        ]
    )

    result = audit_report(report, generated, SOURCE)

    assert [claim.claim for claim in result.unsupported] == ["Holds a CISSP"]


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


async def test_verify_returns_clean_on_grounded_and_covered_doc() -> None:
    # I2: pin the SUCCESS path end-to-end — a fully grounded + fully covered report passes the REAL
    # audit cleanly. The other verify tests exercise overrides/failure; without this, no unit test
    # ever runs verify()->audit_report->clean, so a bug in that wiring would stay green.
    verifier = GroundingVerifier(LLMConfig(model="m"))
    generated = "Took over 100% of inbound email. Skilled in SIEM."
    mocked = VerificationReport(
        claims=[
            gc(
                "Took over 100% of inbound email",
                quote="Took over 100% of inbound email service requests",
            ),
            gc("Skilled in SIEM", quote="Technical Skills: SIEM, Wireshark, Nmap, BIND9"),
        ]
    )
    client = MagicMock()
    client.create = AsyncMock(return_value=mocked)
    with patch.object(verifier, "_get_client", return_value=client):
        result = await verifier.verify(generated, _RESUME)
    assert result.clean and result.complete
    assert result.unsupported == [] and result.coverage_gaps == []


async def test_verify_honors_configured_max_tokens() -> None:
    verifier = GroundingVerifier(LLMConfig(model="m", max_tokens=4096))
    client = MagicMock()
    client.create = AsyncMock(return_value=VerificationReport(claims=[]))

    with patch.object(verifier, "_get_client", return_value=client):
        await verifier.verify("Generated text.", _RESUME)

    assert client.create.call_args.kwargs["max_tokens"] == 4096


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
