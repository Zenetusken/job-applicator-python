"""Tests for resume tailoring engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.config import LLMConfig
from job_applicator.documents.resume_tailor import (
    CHANGES_PROMPT_TEMPLATE,
    TAILOR_PROMPT_TEMPLATE,
    TAILOR_SYSTEM_PROMPT,
    ResumeTailor,
    parse_sections,
)
from job_applicator.embeddings.matching import MatchResult
from job_applicator.exceptions import ConfigError, GroundingUnavailableError
from job_applicator.models import (
    ClaimCheck,
    GroundingReport,
    JobBoard,
    JobListing,
    ResumeData,
    TailoredResume,
)


def _tr(text: str) -> TailoredResume:
    return TailoredResume(
        original_path="",
        tailored_text=text,
        job_title="T",
        job_company="C",
        match_score=0.5,
        semantic_score=0.5,
        skill_score=0.5,
        changes_summary="",
    )


def _mock_matcher(job: JobListing) -> MagicMock:
    matcher = MagicMock()
    matcher.match_resume_to_job = AsyncMock(
        return_value=MatchResult(
            job=job,
            score=0.72,
            semantic_score=0.5,
            skill_score=0.3,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            summary="Good match",
        )
    )
    return matcher


async def test_verify_tailored_attaches_report_for_review() -> None:
    # The résumé path SURFACES the grounding report; it never auto-strips (spec §6).
    tailor = ResumeTailor(LLMConfig(model="m"))
    tailor._verifier.verify = AsyncMock(  # type: ignore[method-assign]
        return_value=GroundingReport(unsupported=[ClaimCheck(claim="x", grounded=False)])
    )
    out = await tailor.verify_tailored(
        _tr("Maintained 100%."), ResumeData(raw_text="src", skills=[])
    )
    assert out.grounding_report is not None
    assert len(out.grounding_report.unsupported) == 1
    assert out.tailored_text == "Maintained 100%."  # text untouched — flagged, not stripped


async def test_verify_tailored_failsafe_leaves_none() -> None:
    # fail-safe (#4): a verifier failure leaves grounding_report=None — never blocked, never a
    # false "verified clean".
    tailor = ResumeTailor(LLMConfig(model="m"))
    tailor._verifier.verify = AsyncMock(  # type: ignore[method-assign]
        side_effect=GroundingUnavailableError("down")
    )
    out = await tailor.verify_tailored(_tr("anything"), ResumeData(raw_text="src", skills=[]))
    assert out.grounding_report is None


async def test_tailor_requires_matcher_or_match_result() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    resume = ResumeData(raw_text="src", skills=[])
    job = MagicMock()
    with pytest.raises(ConfigError, match="configured JobMatcher"):
        await tailor.tailor(resume, job)


async def test_refine_verified_reverifies_the_refined_result() -> None:
    # #4: an interactively refined résumé gets the SAME grounding pass as the primary — refine()
    # then verify_tailored(), the report attached for review (never auto-stripped).
    tailor = ResumeTailor(LLMConfig(model="m"))
    tailor.refine = AsyncMock(return_value=_tr("REFINED text"))  # type: ignore[method-assign]
    tailor._verifier.verify = AsyncMock(  # type: ignore[method-assign]
        return_value=GroundingReport(unsupported=[ClaimCheck(claim="y", grounded=False)])
    )
    resume = ResumeData(raw_text="src", skills=[])
    out = await tailor.refine_verified(resume, _tr("CURRENT"), "feedback", MagicMock())
    assert out.tailored_text == "REFINED text"
    assert out.grounding_report is not None and len(out.grounding_report.unsupported) == 1
    tailor.refine.assert_awaited_once()  # type: ignore[attr-defined]


def test_strip_unsupported_metric_claims_removes_invented_percentages() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "Built async ingestion handling 2B events/day with asyncio.\n"
        "Led a Pydantic v2 + mypy-strict migration across 40 services.\n"
        "Designed PostgreSQL schemas and Redis caching for a 5M-user app."
    )
    tailored = (
        "• Built async ingestion pipelines handling 2 billion events per day using asyncio, "
        "improving throughput by 30%.\n"
        "• Led the migration of 40+ services to Pydantic v2, reducing bugs by 45%.\n"
        "• Designed PostgreSQL schemas for a 5 million user application.\n"
        "References\n"
        "Available upon request."
    )

    cleaned = tailor._strip_unsupported_metric_claims(tailored, original)
    cleaned = tailor._strip_unbacked_references(cleaned, original)

    assert "30%" not in cleaned
    assert "45%" not in cleaned
    assert "2B events" in cleaned
    assert "40 services" in cleaned
    assert "5M user" in cleaned
    assert "References" not in cleaned


def test_strip_unsupported_metric_claims_preserves_source_percentages() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = "Maintained a 95% first-call resolution rate."
    tailored = "• Maintained a 95% first-call resolution rate."

    assert tailor._strip_unsupported_metric_claims(tailored, original) == tailored


def test_extract_education_entries_handles_compound_source_header() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "SUMMARY\nAnalyst.\n\n"
        "EDUCATION & CERTIFICATIONS\n"
        "Undergraduate Certificate — Analysis & Operational Cybersecurity 2024 – Present\n"
        "Northbridge Technical Institute\n"
        "Cisco CCNA & CompTIA Linux+ Coursework 2023 – 2024\n"
        "Riverbend College\n"
        "B.A., Accounting — Metro City University 2012 – 2015\n\n"
        "LANGUAGES\nFluent in French and English, plus Spanish.\n"
    )

    extracted = tailor._extract_education_entries(original)

    assert "EDUCATION & CERTIFICATIONS" in extracted
    assert "Northbridge Technical Institute" in extracted
    assert "Metro City University" in extracted


def test_preserve_source_required_sections_restores_education_and_languages() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "ALEX MORGAN\n\n"
        "EDUCATION & CERTIFICATIONS\n"
        "Undergraduate Certificate — Analysis & Operational Cybersecurity 2024 – Present\n"
        "Northbridge Technical Institute\n"
        "B.A., Accounting — Metro City University 2012 – 2015\n\n"
        "LANGUAGES\n"
        "Fluent in French and English, plus Spanish.\n"
    )
    tailored = (
        "ALEX MORGAN\n\n"
        "**Skills**\nSIEM, Linux\n\n"
        "**Experience**\nSupport Analyst\n\n"
        "**Certifications**\n"
        "Undergraduate Certificate — Analysis & Operational Cybersecurity\n"
        "Northbridge Technical Institute\n\n"
        "**Languages**\n"
        "English (Professional), French (Conversational)"
    )

    result = tailor._preserve_source_required_sections(tailored, original)

    assert "**Education & Certifications**" in result
    assert "**Certifications**" not in result
    assert "B.A., Accounting" in result
    assert "Fluent in French and English, plus Spanish." in result
    assert "English (Professional), French (Conversational)" not in result


def test_localize_standard_labels_for_french_output() -> None:
    tailored = (
        "ALEX MORGAN\n\n"
        "**RÉSUMÉ**\nOperations professional.\n\n"
        "**COMPÉTENCES**\nSIEM, Linux\n\n"
        "**EXPÉRIENCE PROFESSIONNELLE**\nSupport Analyst\n\n"
        "Éducation & Certifications\nCoursework\n\n"
        "Languages\nFluent in French and English, plus Spanish."
    )

    result = ResumeTailor._localize_standard_labels_for_language(tailored, "French")

    assert "RÉSUMÉ" not in result
    assert "Skills" not in result
    assert "EXPÉRIENCE PROFESSIONNELLE" not in result
    assert "Languages" not in result
    assert "**Profil**" in result
    assert "**Compétences**" in result
    assert "**Expérience**" in result
    assert "Formation et certifications" in result
    assert "Langues" in result
    assert "Français et anglais courants; espagnol." in result


def test_strip_non_source_bold_entry_headings() -> None:
    original = "Technical Support Specialist (L1) & Sales\nBeacon Satellite\n2015 - 2018"
    tailored = (
        "**Experience**\n"
        "**Spécialiste de support technique (Niveau 1) & Ventes**\n"
        "Beacon Satellite\n"
        "2015 - 2018\n\n"
        "**Technical Support Specialist (L1) & Sales**\n"
        "Beacon Satellite"
    )

    result = ResumeTailor._strip_non_source_bold_entry_headings(tailored, original)

    assert "**Experience**" in result
    assert "Spécialiste de support technique" not in result
    assert "**Technical Support Specialist (L1) & Sales**" in result


async def test_refine_localizes_standard_labels_for_french_output() -> None:
    job = JobListing(
        title="Analyste",
        company="Acme",
        location="Montréal",
        url="https://example.com/job",
        description="Poste en français.",
        requirements=["Windows"],
        board=JobBoard.LINKEDIN,
    )
    resume = ResumeData(
        raw_text=(
            "ALEX MORGAN\n"
            "Operations professional.\n\n"
            "Education\nCourse\n\n"
            "Languages\nFluent in French and English, plus Spanish."
        ),
        skills=["Windows"],
    )
    tailor = ResumeTailor(LLMConfig(model="m", language="fr"))
    tailor._call_llm = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            "ALEX MORGAN\n\n"
            "SUMMARY\nOperations professional.\n\n"
            "Skills\nWindows\n\n"
            "Experience\nTechnical Support Specialist\n\n"
            "Education\nCourse\n\n"
            "Languages\nFluent in French and English, plus Spanish."
        )
    )
    tailor._summarize_changes = AsyncMock(return_value="localized")  # type: ignore[method-assign]

    result = await tailor.refine(
        resume,
        _tr("old"),
        "Keep source-backed facts.",
        job,
        matcher=_mock_matcher(job),
    )

    assert "Profil" in result.tailored_text
    assert "Compétences" in result.tailored_text
    assert "Expérience" in result.tailored_text
    assert "Formation" in result.tailored_text
    assert "Langues" in result.tailored_text
    assert "Français et anglais courants; espagnol." in result.tailored_text


def test_validate_skills_keeps_parenthetical_skill_as_one_token(llm_config) -> None:
    tailor = ResumeTailor(llm_config)
    original_skills = ["Linux (Fedora, CLI, Bash)", "SIEM"]
    tailored = "**Skills**\n• Linux (Fedora, CLI, Bash), SIEM, Bash)\n\n**Experience**\n"

    result = tailor._validate_skills(tailored, original_skills)

    assert "Linux (Fedora, CLI, Bash)" in result
    assert "SIEM" in result
    assert result.count("Bash)") == 1


def test_strip_malformed_tool_removal_sentence_keeps_clean_summary_sentence() -> None:
    tailored = (
        "SUMMARY\n"
        "Experienced operations professional with 10+ years of support experience. "
        "Skilled in Microsoft 365, ticketing systems, and , with a strong foundation in , "
        "and ticketing systems.\n\n"
        "Skills\nMicrosoft 365, ticketing & escalation"
    )

    result = ResumeTailor._strip_malformed_tool_removal_sentences(tailored)

    assert "Experienced operations professional" in result
    assert "Skilled in Microsoft 365" not in result
    assert "and ," not in result
    assert "Skills\nMicrosoft 365" in result


def test_ground_summary_phrases_rewrites_to_source_backed_terms() -> None:
    original = (
        "SUMMARY\n"
        "Career-changer with 10+ years of operations management and high-stakes "
        "client problem-solving in client-facing roles. University coursework spans "
        "cybersecurity topics. Calm under pressure, with a track record of triage, "
        "escalation, and dispute resolution.\n\n"
        "EXPERIENCE\n"
        "Technical Support Specialist\n"
        "• Delivered Tier 1 technical support by phone, chat, and email.\n"
    )
    tailored = (
        "SUMMARY\n"
        "Experienced operations professional with 10+ years of incident resolution, "
        "client support, and process improvement, now transitioning into IT support "
        "through hands-on technical training and coursework, stakeholder coordination, "
        "and technical troubleshooting experience.\n\n"
        "Skills\nPython"
    )

    result = ResumeTailor._ground_summary_phrases(tailored, original)

    assert "incident resolution" not in result
    assert "process improvement" not in result
    assert "client support" not in result
    assert "hands-on cybersecurity training" not in result
    assert "hands-on technical training" not in result
    assert "stakeholder coordination" not in result
    assert "now transitioning into IT support" not in result
    assert "technical troubleshooting experience" not in result
    assert "high-stakes client problem-solving" in result
    assert "operations management" in result
    assert "client-facing work" in result
    assert "cybersecurity coursework" in result
    assert "triage and escalation" in result
    assert "front-line technical support experience" in result


def test_ensure_source_backed_summary_replaces_generated_summary() -> None:
    original = (
        "SUMMARY\n"
        "Career-changer with 10+ years of operations management and high-stakes "
        "client problem-solving. University coursework spans cybersecurity and networking. "
        "Calm under pressure, with a track record of triage, escalation, and dispute "
        "resolution.\n\n"
        "TECHNICAL SKILLS\nSIEM · SOC operations · incident response · threat intelligence\n\n"
        "EXPERIENCE\nTechnical Support Specialist\n"
        "• Delivered Tier 1 technical support by phone, chat, and email.\n"
        "EDUCATION\nCybersecurity coursework\n"
    )
    tailored = (
        "SUMMARY\n"
        "Experienced candidate transitioning into IT support through certification.\n\n"
        "**Skills**\nSIEM\n\n**Experience**\nSupport"
    )

    result = ResumeTailor._ensure_source_backed_summary(tailored, original)

    assert "Operations professional with 10+ years" in result
    assert "front-line technical support experience" in result
    assert "transitioning into IT support" not in result
    assert "\n\n**Skills**" in result


def test_ensure_source_backed_summary_localizes_french_fallback() -> None:
    original = (
        "SUMMARY\n"
        "Career-changer with 10+ years of operations management and high-stakes "
        "client problem-solving. University coursework spans cybersecurity and networking. "
        "Calm under pressure, with a track record of triage, escalation, and dispute "
        "resolution.\n\n"
        "TECHNICAL SKILLS\nSIEM · SOC operations · incident response · threat intelligence\n\n"
        "EXPERIENCE\nTechnical Support Specialist\n"
        "• Delivered Tier 1 technical support by phone, chat, and email.\n"
        "EDUCATION\nCybersecurity coursework\n"
    )
    tailored = (
        "Profil\n"
        "Candidat expérimenté en transition vers le support TI.\n\n"
        "Compétences\nSIEM\n\nExpérience\nSupport"
    )

    result = ResumeTailor._ensure_source_backed_summary(tailored, original, "French")

    assert "Professionnel des opérations avec plus de 10 ans" in result
    assert "Operations professional with 10+ years" not in result
    assert "transition vers le support TI" not in result


def test_preserve_source_required_sections_localizes_french_source_body() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "ALEX MORGAN\n\n"
        "EDUCATION & CERTIFICATIONS\n"
        "Undergraduate Certificate — Analysis & Operational Cybersecurity 2024 – Present\n"
        "Northbridge Technical Institute\n"
        "Completed: Intro to Cybersecurity, Attack & Defense Methods, Server Security, and "
        "Networking & Security — incl. a hands-on SIEM lab, SOC operations & monitoring, IDS/IPS, "
        "EDR, incident response, and threat intelligence.\n"
        "Cisco CCNA & CompTIA Linux+ Coursework 2023 – 2024\n"
        "Riverbend College\n"
        "Full CCNA networking curriculum — network components, VLSM/subnetting, and routing & "
        "switching (certification exam pending) — plus Linux administration in Fedora "
        "(CLI, scripting).\n\n"
        "LANGUAGES\n"
        "Fluent in French and English, plus Spanish.\n"
    )
    tailored = "ALEX MORGAN\n\nCompétences\nSIEM\n\nExpérience\nSupport"

    result = tailor._preserve_source_required_sections(tailored, original, "French")

    assert "Cours complétés :" in result
    assert "Programme complet de réseautique CCNA" in result
    assert "réponse aux incidents et renseignement sur les menaces" in result
    assert "administration Linux sous Fedora" in result
    assert "Français et anglais courants; espagnol." in result
    assert "Completed:" not in result
    assert "Full CCNA networking curriculum" not in result
    assert " and renseignement" not in result
    assert " in Fedora" not in result
    assert "Fluent in French and English" not in result


def test_preserve_source_required_sections_localizes_projects_home_lab() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "ALEX MORGAN\n\n"
        "PROJECTS & HOME LAB\n"
        "Home cybersecurity lab & pen-test sandbox — a multi-VM environment (Kali attacker "
        "against target hosts) for hands-on attack/defense and detection practice\n"
        "Self-hosted BIND9 DNS server — configured, secured, and administered end-to-end\n"
        "Hands-on security tooling — packet analysis (Wireshark), network scanning (Nmap), "
        "network monitoring (Zabbix), and Kali Linux, through coursework labs and TryHackMe\n\n"
        "LANGUAGES\n"
        "Fluent in French and English, plus Spanish.\n"
    )
    tailored = "ALEX MORGAN\n\nCompétences\nSIEM\n\nExpérience\nSupport"

    result = tailor._preserve_source_required_sections(tailored, original, "French")

    assert "Projets & laboratoire à domicile" in result
    assert "Laboratoire de cybersécurité à domicile" in result
    assert "Serveur DNS BIND9 autohébergé" in result
    assert "Outils de sécurité en pratique" in result
    assert "Projects & Home Lab" not in result
    assert "Home cybersecurity lab" not in result
    assert "Self-hosted BIND9" not in result


def test_polish_french_output_fixes_measured_literal_phrases() -> None:
    tailored = (
        "Expérience\n"
        "- Prendre en charge 100 % des demandes de service client provenant de l'équipe de vente\n"
        "- Géré plus de 100 % des demandes de service entrantes du service commercial\n"
        "- Réalisé les contrats et assigné les opérateurs aux plannings quotidiens\n"
        "- Géré les relations clients VIP dans une opération rapide et à faible délai\n"
        "- Géré les relations clients VIP dans une opération rapide et exigeante en temps\n"
        "- Géré les relations clients VIP dans une opération rapide et à haute priorité\n"
        "- Protéger la rétention client dans une entreprise à contrats bloqués\n"
        "- Protéger la rétention client dans une entreprise à contrat verrouillé\n"
        "- A été le point de contact technique du département, coordonnant avec le fournisseur "
        "externe d'IT responsable du réseau\n"
        "- A été le point de contact technique du département, coordonnant avec le fournisseur "
        "externe d\u2019IT responsable du réseau\n"
        "- Résolu les demandes de réclamation de bénéfices par téléphone et courriel en "
        "résolution à la première tentative\n"
        "- A résolu les demandes de bénéfices en téléphone et par e-mail pour une résolution "
        "en première instance\n"
        "- Résolu les demandes de prestations par téléphone et courriel en résolution en premier "
        "appel\n"
        "- A résolu des demandes de prestations en téléphone et par courriel pour une résolution "
        "au premier appel\n"
        "- Résolu les problèmes de signal en diagnosticant et en résolvant les problèmes\n"
        "- Troublé les problèmes de site web et de CRM pour Salesforce, SAP et Microsoft 365\n"
        "- Apporté un support technique de niveau 1 par téléphone, chat et courriel, diagnostic "
        "et résolution à distance des problèmes de signal\n"
        "- Trié et escalada les problèmes complexes vers les niveaux supérieurs"
    )

    result = ResumeTailor._polish_french_output(tailored, "French")

    assert "Pris en charge 100 %" in result
    assert "Géré plus de 100 %" not in result
    assert "Planifié les contrats" in result
    assert "environnement rapide à délais serrés" in result
    assert "demandes de prestations" in result
    assert "Dépanné les problèmes de site web" in result
    assert "Fourni un support technique de niveau 1" in result
    assert "diagnostiquant et résolvant à distance" in result
    assert "diagnosticant" not in result
    assert "fournisseur TI externe" in result
    assert "contrats fixes" in result
    assert "fidélisation des clients" in result
    assert "haute priorité" not in result
    assert "rétention client" not in result
    assert "contrat verrouillé" not in result
    assert "Trié et escaladé" in result
    assert "première tentative" not in result
    assert "demandes de prestations par téléphone et par courriel" in result
    assert "première instance" not in result
    assert "e-mail" not in result
    assert "en téléphone" not in result


def test_strip_duplicate_bullets_keeps_last_occurrence() -> None:
    tailored = (
        "Expérience\n"
        "Canada Life\n"
        "• Résolu les demandes de prestations.\n"
        "• Trié et escaladé les problèmes complexes vers les niveaux supérieurs.\n\n"
        "Beacon Satellite\n"
        "• Fourni un support technique de niveau 1.\n"
        "• Trié et escaladé les problèmes complexes vers les niveaux supérieurs.\n"
    )

    result = ResumeTailor._strip_duplicate_bullets(tailored)

    assert result.count("Trié et escaladé") == 1
    assert "Beacon Satellite\n• Fourni" in result
    assert "Beacon Satellite\n• Fourni un support technique de niveau 1.\n• Trié" in result


def test_strip_unverifiable_aspirations_and_unbacked_responsibility_bullets() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "Senior Python engineer, 8 years building async data pipelines and ML services.\n"
        "Built async ingestion handling 2B events/day with asyncio."
    )
    tailored = (
        "Experienced Python engineer. Seeking to leverage expertise as a Senior Python Engineer.\n"
        "• Built async ingestion pipelines using asyncio.\n"
        "• Collaborated with data scientists to design scalable ML service architectures."
        "\n• Supported internal teams with technical guidance and escalation procedures."
    )

    cleaned = tailor._strip_unverifiable_aspirations(tailored)
    cleaned = tailor._strip_unbacked_responsibility_bullets(cleaned, original)

    assert "Seeking to leverage" not in cleaned
    assert "Collaborated with data scientists" not in cleaned
    assert "Supported internal teams with technical guidance" not in cleaned
    assert "Built async ingestion" in cleaned


def test_strip_unverifiable_aspirations_removes_french_target_role_sentence() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    tailored = (
        "Professionnel des opérations. Recherche un rôle d'analyste junior en cybersécurité "
        "opérationnelle et gestion des risques, en alignement avec ses compétences."
    )

    cleaned = tailor._strip_unverifiable_aspirations(tailored)

    assert "Recherche un rôle" not in cleaned
    assert cleaned == "Professionnel des opérations."


def test_strip_unbacked_responsibility_bullets_removes_deployment_and_testing_claims() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "Senior Python engineer. Skills: Python, Pydantic, PostgreSQL, Docker, AWS. "
        "Built async ingestion handling 2B events/day with asyncio."
    )
    tailored = (
        "• Built async ingestion handling 2B events/day with asyncio.\n"
        "• Integrated Docker and AWS for deployment and monitoring of backend systems.\n"
        "• Automated testing and CI/CD pipelines for Python-based services."
    )

    cleaned = tailor._strip_unbacked_responsibility_bullets(tailored, original)

    assert "Built async ingestion" in cleaned
    assert "deployment and monitoring" not in cleaned
    assert "CI/CD pipelines" not in cleaned


def test_strip_unbacked_responsibility_bullets_removes_french_continuity_overclaim() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "Operations Coordinator\n"
        "Booked contracts and assigned operators across daily, weekly, and monthly schedules.\n"
        "Managed VIP client relationships in a fast-paced, time-critical operation.\n"
        "Coordinated drivers and the mechanical department to keep the fleet operational."
    )
    tailored = (
        "• Réalisé les contrats et assigné les opérateurs aux plannings quotidiens, "
        "hebdomadaires et mensuels\n"
        "• Participé à la gestion des urgences et des incidents techniques, en assurant la "
        "continuité des opérations"
    )

    cleaned = tailor._strip_unbacked_responsibility_bullets(tailored, original)

    assert "Réalisé les contrats" in cleaned
    assert "continuité des opérations" not in cleaned


def test_normalize_date_range_dashes_for_grounding() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))

    assert (
        tailor._normalize_date_range_dashes("Staff Engineer (2021–Present)")
        == "Staff Engineer (2021-Present)"
    )


def test_strip_low_evidence_bullets_keeps_source_backed_paraphrases() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "Built async ingestion handling 2B events/day with asyncio.\n"
        "Led a Pydantic v2 + mypy-strict migration across 40 services."
    )
    tailored = (
        "• Built asynchronous ingestion pipelines handling 2B events per day using asyncio.\n"
        "• Led a Pydantic v2 migration across 40 services with mypy-strict enforcement.\n"
        "• Designed scalable Python workflows using FastAPI."
    )

    cleaned = tailor._strip_low_evidence_bullets(tailored, original)

    assert "Built asynchronous ingestion" in cleaned
    assert "Pydantic v2 migration" in cleaned
    assert "Designed scalable Python workflows" not in cleaned


def test_strip_low_evidence_bullets_keeps_source_backed_french_translations() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "Managed daily delivery operations and drivers as the single escalation point for all "
        "operational issues — on-site client disputes, missing-item claims, and scheduling and "
        "driver coordination.\n"
        "Booked contracts and assigned operators across daily, weekly, and monthly schedules.\n"
        "Triaged and escalated complex issues to higher tiers per documented procedures."
    )
    tailored = (
        "• Géré les opérations quotidiennes de livraison et les conducteurs en tant que point "
        "d'escalade unique pour tous les problèmes opérationnels.\n"
        "• Réalisé les contrats et assigné les opérateurs aux plannings quotidiens, "
        "hebdomadaires et mensuels.\n"
        "• Trié et escaladé les problèmes complexes vers les niveaux supérieurs selon les "
        "procédures documentées.\n"
        "• Conçu des tableaux de bord Power BI pour automatiser les rapports exécutifs."
    )

    cleaned = tailor._strip_low_evidence_bullets(tailored, original)

    assert "Géré les opérations quotidiennes" in cleaned
    assert "Réalisé les contrats" in cleaned
    assert "Trié et escaladé" in cleaned
    assert "Power BI" not in cleaned


def test_strip_low_evidence_bullets_keeps_source_backed_french_home_lab() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "PROJECTS & HOME LAB\n"
        "Home cybersecurity lab & pen-test sandbox — a multi-VM environment (Kali attacker "
        "against target hosts) for hands-on attack/defense and detection practice\n"
        "Self-hosted BIND9 DNS server — configured, secured, and administered end-to-end\n"
        "Hands-on security tooling — packet analysis (Wireshark), network scanning (Nmap), "
        "network monitoring (Zabbix), and Kali Linux, through coursework labs and TryHackMe"
    )
    tailored = (
        "• Laboratoire de cybersécurité à domicile et sandbox de test de pénétration — "
        "environnement multi-VM (Kali attaquant contre des hôtes cibles) pour la pratique "
        "attaque/défense et détection.\n"
        "• Serveur DNS BIND9 auto-hébergé — configuré, sécurisé et administré de bout en bout.\n"
        "• Outils de sécurité en pratique — analyse de paquets (Wireshark), scan de réseau "
        "(Nmap), surveillance réseau (Zabbix), Kali Linux, laboratoires de cours et TryHackMe.\n"
        "• Déployé Splunk SOAR pour automatiser les incidents en production."
    )

    cleaned = tailor._strip_low_evidence_bullets(tailored, original)

    assert "Laboratoire de cybersécurité" in cleaned
    assert "Serveur DNS BIND9" in cleaned
    assert "Outils de sécurité" in cleaned
    assert "Splunk SOAR" not in cleaned


def test_strip_unbacked_responsibility_bullets_drops_french_real_time_overclaim() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = "Managed daily delivery operations and drivers. Booked contracts."
    tailored = (
        "• Participé à la planification et à la gestion des opérations en temps réel.\n"
        "• Géré les opérations quotidiennes de livraison et les conducteurs."
    )

    cleaned = tailor._strip_unbacked_responsibility_bullets(tailored, original)

    assert "temps réel" not in cleaned
    assert "opérations quotidiennes" in cleaned


def test_strip_unbacked_responsibility_bullets_drops_french_training_seminar_noise() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = "Ran technical training and learning seminars for fellow agents."
    tailored = (
        "• A organisé des séminaires de formation et d'apprentissage pour les agents.\n"
        "A organisé des séminaires de formation et d'apprentissage pour les agents.\n"
        "• A fourni un support technique de niveau 1 par téléphone."
    )

    cleaned = tailor._strip_unbacked_responsibility_bullets(tailored, original)

    assert "séminaires" not in cleaned
    assert "support technique" in cleaned


def test_strip_unbacked_responsibility_bullets_drops_french_operations_flow_overclaim() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = "Booked contracts and assigned operators across schedules."
    tailored = (
        "• Participé à la gestion des plannings et à la coordination des équipes pour assurer "
        "la fluidité opérationnelle.\n"
        "• Participé à la gestion des opérations et à la coordination des équipes pour assurer "
        "la fluidité des processus.\n"
        "• Planifié les contrats et assigné les opérateurs aux horaires."
    )

    cleaned = tailor._strip_unbacked_responsibility_bullets(tailored, original)

    assert "fluidité opérationnelle" not in cleaned
    assert "fluidité des processus" not in cleaned
    assert "Planifié les contrats" in cleaned


def test_strip_misplaced_support_domain_bullets_keeps_shaw_only() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "Beacon Satellite Support\n"
        "Delivered Tier 1 technical support by phone, chat, and email, diagnosing signal, "
        "receiver, connectivity, and website issues.\n\n"
        "Benefits Operations\n"
        "Resolved benefits requests by phone and email."
    )
    tailored = (
        "Expérience\n"
        "Benefits Operations, Montréal\n"
        "• Résolu les demandes de prestations par téléphone et courriel.\n"
        "• Résolu les problèmes techniques en première ligne par téléphone, chat et courriel, "
        "en diagnostiquant et en résolvant les problèmes de signal, récepteur, connectivité et "
        "site web.\n\n"
        "Beacon Satellite Support, Montréal\n"
        "• Fourni un support technique niveau 1 par téléphone, chat et courriel, en "
        "diagnostiquant et en résolvant les problèmes de signal, récepteur, connectivité et "
        "site web."
    )

    cleaned = tailor._strip_misplaced_support_domain_bullets(tailored, original)

    benefits, beacon = cleaned.split("Beacon Satellite Support", maxsplit=1)
    assert "demandes de prestations" in benefits
    assert "signal" not in benefits
    assert "signal" in beacon


def test_strip_misplaced_support_domain_bullets_keeps_source_owned_context() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = (
        "Northstar Field Support\n"
        "Delivered Tier 1 technical support by phone, chat, and email for signal, receiver, "
        "connectivity, and website issues."
    )
    tailored = (
        "Expérience\n"
        "Northstar Field Support, Montréal\n"
        "• Fourni un support technique niveau 1 par téléphone, chat et courriel pour des "
        "problèmes de signal, récepteur, connectivité et site web."
    )

    cleaned = tailor._strip_misplaced_support_domain_bullets(tailored, original)

    assert "signal" in cleaned


def test_validate_skills_uses_french_skills_header() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    tailored = "Profil\nAnalyste support.\n\nCompétences\nPython, Kubernetes\n\nExpérience\nSupport"

    cleaned = tailor._validate_skills(tailored, ["Python"])

    assert "Python" in cleaned
    assert "Kubernetes" not in cleaned


def test_strip_hallucinated_education_uses_french_formation_header() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = "ALEX MORGAN\n\nExperience\nSupport analyst."
    tailored = (
        "Profil\nAnalyste support.\n\nFormation\nCertificat cybersécurité\n\nExpérience\nSupport"
    )

    cleaned = tailor._strip_hallucinated_education(tailored, original)

    assert "Formation" not in cleaned
    assert "Certificat cybersécurité" not in cleaned
    assert "Expérience" in cleaned


def test_strip_unbacked_optional_sections_uses_french_language_header() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = "ALEX MORGAN\n\nExpérience\nSupport analyst."
    tailored = "Profil\nAnalyste support.\n\nLangues\nFrançais et anglais.\n\nExpérience\nSupport"

    cleaned = tailor._strip_unbacked_optional_sections(tailored, original)

    assert "Langues" not in cleaned
    assert "Français et anglais" not in cleaned
    assert "Expérience" in cleaned


def test_strip_unearned_credentials_handles_french_profile_terms() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = "ALEX MORGAN\n\nProfil\nAnalyste support.\n\nCompétences\nPython"
    tailored = (
        "ALEX MORGAN\n\n"
        "Profil\n"
        "Analyste certifié en cybersécurité avec expérience support.\n\n"
        "Compétences\nPython"
    )

    cleaned = tailor._strip_unearned_credentials(tailored, original)

    assert "certifié" not in cleaned
    assert "Analyste en cybersécurité" in cleaned


def test_strip_unbacked_responsibility_bullets_drops_french_client_interaction_noise() -> None:
    tailor = ResumeTailor(LLMConfig(model="m"))
    original = "Delivered Tier 1 technical support by phone, chat, and email."
    tailored = (
        "• Géré les demandes techniques et les interactions clients en tant que technicien "
        "de niveau 1 et commercial.\n"
        "• Fourni un support technique de niveau 1 par téléphone, chat et courriel."
    )

    cleaned = tailor._strip_unbacked_responsibility_bullets(tailored, original)

    assert "interactions clients" not in cleaned
    assert "support technique" in cleaned


def test_strip_non_source_bold_headings_keeps_localized_home_lab_section() -> None:
    original = (
        "PROJECTS & HOME LAB\nHome cybersecurity lab & pen-test sandbox — a multi-VM environment."
    )
    tailored = (
        "**PROJETS & LABORATOIRE À DOMICILE**\n"
        "• Laboratoire de cybersécurité à domicile et sandbox de test de pénétration."
    )

    cleaned = ResumeTailor._strip_non_source_bold_entry_headings(tailored, original)

    assert "**PROJETS & LABORATOIRE À DOMICILE**" in cleaned


@pytest.fixture
def sample_resume():
    return ResumeData(
        raw_text=("ALEX MORGAN\nalex@example.com\nSkills\nWindows, Office 365, Troubleshooting"),
        name="ALEX MORGAN",
        email="alex@example.com",
        skills=["Windows", "Office 365", "Troubleshooting"],
    )


@pytest.fixture
def sample_job():
    return JobListing(
        title="Technical Support Specialist",
        company="CGI",
        url="https://example.com/job",
        description="Provide technical support.",
        requirements=["Windows", "Office 365", "ServiceNow"],
        location="Montreal, QC",
        board=JobBoard.INDEED,
    )


@pytest.fixture
def llm_config():
    from job_applicator.config import LLMConfig

    return LLMConfig(
        api_base="http://localhost:8000/v1",
        model="test-model",
    )


class TestResumeTailor:
    def test_init(self, llm_config):
        tailor = ResumeTailor(llm_config)
        assert tailor._config == llm_config

    @pytest.mark.asyncio
    async def test_call_llm_honors_configured_max_tokens(self):
        """_call_llm must pass the configured max_tokens, not a hardcoded value."""
        from job_applicator.config import LLMConfig

        config = LLMConfig(api_base="http://localhost:8000/v1", model="m", max_tokens=1234)
        tailor = ResumeTailor(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acompletion:
            await tailor._call_llm("prompt")

        assert mock_acompletion.call_args.kwargs["max_tokens"] == 1234

    def test_prompt_template_formatting(self):
        prompt = TAILOR_PROMPT_TEMPLATE.format(
            job_title="Test Job",
            job_company="Test Co",
            job_location="Remote",
            job_description="Test desc",
            requirements="Skill1, Skill2",
            resume_text="Resume text",
            skills="Skill1, Skill2",
            education_entries="1. Test University, 2020-2024",
            tone_section="TONE: Corporate",
            user_instructions="No instructions.",
        )
        assert "Test Job" in prompt
        assert "Test Co" in prompt
        assert "Resume text" in prompt

    def test_changes_prompt_template(self):
        prompt = CHANGES_PROMPT_TEMPLATE.format(
            original_preview="Original text",
            tailored_preview="Tailored text",
        )
        assert "Original text" in prompt
        assert "Tailored text" in prompt

    def test_system_prompt_has_few_shot_examples(self):
        """System prompt should contain before/after examples."""
        assert "BEFORE summary" in TAILOR_SYSTEM_PROMPT
        assert "AFTER summary" in TAILOR_SYSTEM_PROMPT
        assert "BEFORE bullet" in TAILOR_SYSTEM_PROMPT
        assert "AFTER bullet" in TAILOR_SYSTEM_PROMPT

    def test_system_prompt_has_third_person_rule(self):
        """System prompt should enforce third person in summaries."""
        assert "THIRD PERSON" in TAILOR_SYSTEM_PROMPT
        assert "'I'" in TAILOR_SYSTEM_PROMPT or "never use" in TAILOR_SYSTEM_PROMPT.lower()

    def test_system_prompt_has_power_word_limits(self):
        """System prompt should limit power word usage."""
        assert "sparingly" in TAILOR_SYSTEM_PROMPT.lower()

    @pytest.mark.asyncio
    async def test_tailor_returns_result(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "ALEX MORGAN\nalex@example.com\n"
            "Skills: Windows, Office 365, Troubleshooting, ServiceNow\n"
            "Experience: Technical Support..."
        )

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tailor.tailor(
                sample_resume, sample_job, matcher=_mock_matcher(sample_job)
            )

        assert isinstance(result, TailoredResume)
        assert result.job_title == "Technical Support Specialist"
        assert result.job_company == "CGI"
        assert result.attempt == 1
        assert len(result.tailored_text) > 0

    @pytest.mark.asyncio
    async def test_refine_increments_attempt(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        initial = TailoredResume(
            original_path="",
            tailored_text="Initial tailored text",
            job_title="Technical Support Specialist",
            job_company="CGI",
            match_score=0.7,
            semantic_score=0.76,
            skill_score=0.6,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            changes_summary="Initial changes",
            attempt=1,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Refined resume text"

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tailor.refine(
                sample_resume,
                initial,
                "Add more detail",
                sample_job,
                matcher=_mock_matcher(sample_job),
            )

        assert result.attempt == 2
        assert result.user_modifications == "Add more detail"

    @pytest.mark.asyncio
    async def test_tailor_populates_scores(self, llm_config, sample_resume, sample_job):
        """TailoredResume should have non-zero semantic_score and skill_score."""
        from job_applicator.embeddings.matching import MatchResult

        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Tailored text"

        mock_match = MatchResult(
            job=sample_job,
            score=0.72,
            semantic_score=0.5,
            skill_score=0.3,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            summary="Good match",
        )
        mock_matcher = MagicMock()
        mock_matcher.match_resume_to_job = AsyncMock(return_value=mock_match)

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tailor.tailor(sample_resume, sample_job, matcher=mock_matcher)

        assert result.semantic_score > 0.0
        assert result.skill_score > 0.0
        assert result.match_score == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_tailor_rejects_empty_completion(self, llm_config, sample_resume, sample_job):
        """An empty LLM completion must raise LLMError (typed), not yield an empty TailoredResume
        that silently flows into cover-letter generation + PDF rendering."""
        from job_applicator.embeddings.matching import MatchResult
        from job_applicator.exceptions import LLMError

        tailor = ResumeTailor(llm_config)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "   "  # empty after stripping

        mock_matcher = MagicMock()
        mock_matcher.match_resume_to_job = AsyncMock(
            return_value=MatchResult(
                job=sample_job,
                score=0.5,
                semantic_score=0.5,
                skill_score=0.5,
                matched_skills=[],
                missing_skills=[],
                summary="",
            )
        )
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(LLMError):
                await tailor.tailor(sample_resume, sample_job, matcher=mock_matcher)

    @pytest.mark.asyncio
    async def test_tailor_accepts_matcher_param(self, llm_config, sample_resume, sample_job):
        """Passing a matcher should reuse it instead of creating a new one."""
        from job_applicator.embeddings.matching import MatchResult

        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Tailored text"

        mock_match = MatchResult(
            job=sample_job,
            score=0.8,
            semantic_score=0.6,
            skill_score=0.4,
            matched_skills=["Windows"],
            missing_skills=[],
            summary="Strong match",
        )
        mock_matcher = MagicMock()
        mock_matcher.match_resume_to_job = AsyncMock(return_value=mock_match)

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await tailor.tailor(sample_resume, sample_job, matcher=mock_matcher)

        mock_matcher.match_resume_to_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_refine_accepts_matcher_param(self, llm_config, sample_resume, sample_job):
        """Refine should accept and use a matcher parameter."""
        from job_applicator.embeddings.matching import MatchResult

        tailor = ResumeTailor(llm_config)
        initial = TailoredResume(
            original_path="",
            tailored_text="Initial text",
            job_title="Technical Support Specialist",
            job_company="CGI",
            match_score=0.7,
            semantic_score=0.5,
            skill_score=0.3,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            changes_summary="changes",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Refined text"

        mock_match = MatchResult(
            job=sample_job,
            score=0.85,
            semantic_score=0.65,
            skill_score=0.5,
            matched_skills=["Windows", "Office 365"],
            missing_skills=[],
            summary="Strong match",
        )
        mock_matcher = MagicMock()
        mock_matcher.match_resume_to_job = AsyncMock(return_value=mock_match)

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tailor.refine(
                sample_resume,
                initial,
                "Add detail",
                sample_job,
                matcher=mock_matcher,
            )

        assert result.semantic_score > 0.0
        assert result.skill_score > 0.0
        mock_matcher.match_resume_to_job.assert_called_once()

    def test_call_llm_temperature_default(self, llm_config):
        """_call_llm should default to temperature=0.4."""
        tailor = ResumeTailor(llm_config)
        import inspect

        sig = inspect.signature(tailor._call_llm)
        assert sig.parameters["temperature"].default == 0.4


class TestTailoredResumeModel:
    def test_model_creation(self):
        resume = TailoredResume(
            original_path="/path/to/resume.pdf",
            tailored_text="Tailored content",
            job_title="Test Job",
            job_company="Test Co",
            match_score=0.75,
            semantic_score=0.8,
            skill_score=0.65,
            matched_skills=["Python"],
            missing_skills=["AWS"],
            changes_summary="Emphasized Python skills",
        )
        assert resume.attempt == 1
        assert resume.user_modifications == ""
        assert resume.output_path == ""

    def test_model_serialization(self):
        resume = TailoredResume(
            original_path="",
            tailored_text="text",
            job_title="Job",
            job_company="Co",
            match_score=0.5,
            semantic_score=0.5,
            skill_score=0.5,
            changes_summary="changes",
        )
        data = resume.model_dump()
        assert "tailored_text" in data
        assert "match_score" in data
        assert "created_at" in data


class TestTailorWithTone:
    @pytest.mark.asyncio
    async def test_tailor_includes_tone_in_prompt(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Tailored with tone"

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            await tailor.tailor(sample_resume, sample_job, matcher=_mock_matcher(sample_job))

        first_call = mock_call.call_args_list[0]
        assert "TONE:" in str(first_call)


class TestParseSections:
    def test_parse_standard_sections(self):
        text = (
            "JOHN DOE\njohn@example.com\n\n"
            "SUMMARY\nExperienced developer.\n\n"
            "EXPERIENCE\nSoftware Engineer at Corp\n2020-2024\n\n"
            "SKILLS\nPython, JavaScript, Docker\n\n"
            "EDUCATION\nBS Computer Science, MIT, 2016-2020\n"
        )
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "SUMMARY" in names
        assert "EXPERIENCE" in names
        assert "SKILLS" in names
        assert "EDUCATION" in names

    def test_parse_mixed_case_headers(self):
        text = "Summary\nSome text.\n\nExperience\nJob stuff.\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "Summary" in names
        assert "Experience" in names

    def test_parse_no_sections_returns_single(self):
        text = "Just a plain resume with no section headers at all."
        sections = parse_sections(text)
        assert len(sections) == 1
        assert sections[0].name == "Full Document"
        assert sections[0].text == text

    def test_section_text_preserved(self):
        text = "SKILLS\nPython, JavaScript\nDocker, Kubernetes\n\nEXPERIENCE\nJob one.\n"
        sections = parse_sections(text)
        skills = next(s for s in sections if s.name == "SKILLS")
        assert "Python" in skills.text
        assert "Docker" in skills.text

    def test_header_with_colon(self):
        text = "Technical Skills:\nPython, Java\n\nWork Experience:\nJob stuff.\n"
        sections = parse_sections(text)
        names = [s.name for s in sections]
        assert "Technical Skills:" in names
        assert "Work Experience:" in names


class TestTailorWorkflow:
    def test_tailor_session_workflow(self):
        """Test the full accept/retry/input workflow with mock data."""
        from job_applicator.models import TailorSession

        session = TailorSession(
            original_text="Original resume",
            job_title="Dev",
            job_company="Co",
        )

        for i in range(3):
            result = TailoredResume(
                original_path="",
                tailored_text=f"Tailored version {i + 1}",
                job_title="Dev",
                job_company="Co",
                match_score=0.5 + i * 0.1,
                semantic_score=0.5,
                skill_score=0.5,
                changes_summary=f"Changes for attempt {i + 1}",
                attempt=i + 1,
                user_modifications="" if i == 0 else "more detail",
            )
            session.add_attempt(result)

        assert len(session.attempts) == 3
        assert session.current.tailored_text == "Tailored version 3"

        session.select(0)
        assert session.current.tailored_text == "Tailored version 1"

        with pytest.raises(IndexError):
            session.select(99)

    def test_parse_sections_and_select(self):
        """Test section parsing for editing workflow."""
        from job_applicator.documents.resume_tailor import parse_sections

        text = (
            "John Doe - Developer\n\n"
            "SUMMARY\nExperienced developer.\n\n"
            "SKILLS\nPython, JavaScript\n\n"
            "EXPERIENCE\nSoftware engineer at Corp (2020-2024)\n"
        )
        sections = parse_sections(text)
        assert len(sections) == 3
        assert sections[0].name == "SUMMARY"
        assert "Experienced developer" in sections[0].text


class TestCoverLetterWorkflow:
    def test_cover_letter_session_workflow(self):
        from job_applicator.models import CoverLetterResult, CoverLetterSession

        session = CoverLetterSession(job_title="Dev", job_company="Co")

        for i in range(3):
            session.add_attempt(
                CoverLetterResult(
                    job_title="Dev",
                    job_company="Co",
                    cover_letter_text=f"Letter version {i + 1}",
                    attempt=i + 1,
                )
            )

        assert len(session.attempts) == 3
        assert session.current.cover_letter_text == "Letter version 3"
        assert session.attempts[0].cover_letter_text == "Letter version 1"

        session.select(0)
        assert session.current.cover_letter_text == "Letter version 1"


class TestAuditFixes:
    """Tests for the 5 audit fixes: date, power words, job titles, education order, first person."""

    def test_tailor_prompt_forbids_first_person(self):
        """Fix 5: System prompt should forbid 'I', 'my', 'me' in summary."""
        from job_applicator.documents.resume_tailor import TAILOR_SYSTEM_PROMPT

        has_third = (
            "THIRD PERSON" in TAILOR_SYSTEM_PROMPT or "third person" in TAILOR_SYSTEM_PROMPT.lower()
        )
        assert has_third
        assert "'I'" in TAILOR_SYSTEM_PROMPT or "'my'" in TAILOR_SYSTEM_PROMPT

    def test_tailor_prompt_limits_power_words(self):
        """Fix 2: System prompt should limit ornate power verbs."""
        from job_applicator.documents.resume_tailor import TAILOR_SYSTEM_PROMPT

        assert "sparingly" in TAILOR_SYSTEM_PROMPT.lower() or "2-3 per job" in TAILOR_SYSTEM_PROMPT

    def test_tailor_prompt_preserves_job_titles(self):
        """Fix 3: System prompt should preserve complete job titles."""
        from job_applicator.documents.resume_tailor import TAILOR_SYSTEM_PROMPT

        assert "NEVER remove or shorten job titles" in TAILOR_SYSTEM_PROMPT
        assert "Dental & Medical" in TAILOR_SYSTEM_PROMPT

    def test_tailor_prompt_enforces_reverse_chronological_education(self):
        """Fix 4: System prompt should enforce reverse-chronological education."""
        from job_applicator.documents.resume_tailor import TAILOR_SYSTEM_PROMPT

        has_order = (
            "REVERSE-CHRONOLOGICAL" in TAILOR_SYSTEM_PROMPT
            or "most recent first" in TAILOR_SYSTEM_PROMPT.lower()
        )
        assert has_order

    def test_cover_letter_prompt_includes_date(self):
        """Fix 1: Cover letter prompt should include today's date."""
        from datetime import datetime as dt

        from job_applicator.documents.cover_letter import CoverLetterGenerator

        generator = CoverLetterGenerator.__new__(CoverLetterGenerator)
        generator._config = MagicMock()

        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = MagicMock()
        user.first_name = "John"
        user.last_name = "Doe"
        user.email = "j@e.com"
        resume = ResumeData(raw_text="Resume", skills=["Python"])

        prompt = generator._build_prompt(
            job,
            user,
            resume,
            tailored_resume_text="Tailored resume text",
        )

        today = dt.now().strftime("%B %d, %Y")
        assert today in prompt
        assert "Today's date:" in prompt
        assert "Do NOT write" in prompt  # instruction to not use [Date] placeholder

    def test_cover_letter_prompt_no_date_without_tailored_text(self):
        """Cover letter prompt without tailored_resume_text should not inject date."""
        from job_applicator.documents.cover_letter import CoverLetterGenerator

        generator = CoverLetterGenerator.__new__(CoverLetterGenerator)
        generator._config = MagicMock()

        job = JobListing(
            title="Dev",
            company="Co",
            url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = MagicMock()
        user.first_name = "John"
        user.last_name = "Doe"
        user.email = "j@e.com"
        resume = ResumeData(raw_text="Resume", skills=["Python"])

        prompt = generator._build_prompt(job, user, resume)

        assert "Today's date:" not in prompt


class TestResumeDateValidator:
    def test_audit_with_no_dates(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="No dates here at all.")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) == 0
        assert not result.is_stale
        assert result.is_ordered
        assert result.latest_date == ""
        assert result.earliest_date == ""

    def test_audit_with_present_date(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EXPERIENCE\nSoftware Engineer\nCorp, City\n2020 - Present")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) > 0
        assert any(e.is_current for e in result.entries)
        present_entry = next(e for e in result.entries if e.is_current)
        assert present_entry.end == "Present"
        assert present_entry.start == "2020"

    def test_audit_detects_staleness(self):
        from datetime import datetime

        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EXPERIENCE\nOld Job\nCorp, City\n2000 - 2005")
        validator = ResumeDateValidator(reference_date=datetime(2030, 1, 1))
        result = validator.audit(resume)
        assert result.is_stale
        assert len(result.staleness_issues) > 0
        assert "2005" in result.staleness_issues[0]

    def test_audit_ordering_issues(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=("EXPERIENCE\nOld Job\nCorp\n2010 - 2015\nNew Job\nCorp\n2018 - 2024")
        )
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.ordering_issues) > 0
        assert not result.is_ordered
        assert any("should come after" in issue for issue in result.ordering_issues)

    def test_audit_year_only_dates(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EXPERIENCE\nJob\nCorp\n2018 - 2020")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) > 0
        entry = result.entries[0]
        assert entry.start == "2018"
        assert entry.end == "2020"
        assert entry.is_current is False

    def test_audit_month_year_format(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EXPERIENCE\nJob\nCorp\nJan 2020 - Jun 2022")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) > 0
        entry = result.entries[0]
        assert entry.start == "January 2020"
        assert entry.end == "June 2022"

    def test_audit_empty_text(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) == 0
        assert not result.is_stale
        assert result.is_ordered

    def test_audit_multiple_entries_chronological(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=("EXPERIENCE\nNew Job\nCorp\n2020 - Present\nOld Job\nCorp\n2015 - 2019")
        )
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) == 2
        assert result.is_ordered
        assert result.latest_date != ""
        assert result.earliest_date != ""

    def test_audit_latest_and_earliest_dates(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=("EXPERIENCE\nNewest Job\nCorp\n2022 - Present\nOldest Job\nCorp\n2010 - 2014")
        )
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        # "Present" resolves to current date (June 2026), earliest is 2010
        assert result.latest_date != ""
        assert result.earliest_date != ""
        assert "2010" in result.earliest_date

    def test_audit_stale_newest_entry_when_not_current(self):
        """A CV whose newest dated entry is old AND not 'Present' fires the NEWEST-ENTRY staleness.
        (The education-AGE heuristic was removed as noise — this is the general check, and it must
        NOT be an education-specific message.)"""
        from datetime import datetime

        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="EDUCATION\nBS Computer Science\nMIT\n1998 - 2002")
        result = ResumeDateValidator(reference_date=datetime(2030, 1, 1)).audit(resume)
        assert result.is_stale
        assert any("Most recent entry" in s for s in result.staleness_issues)
        assert not any("Education" in s for s in result.staleness_issues)

    def test_audit_education_old_but_current_work_not_stale(self):
        from datetime import datetime

        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=(
                "EXPERIENCE\nCurrent Job\nCorp\n2020 - Present\n\n"
                "EDUCATION\nBS CS\nMIT\n2000 - 2004"
            )
        )
        validator = ResumeDateValidator(reference_date=datetime(2030, 1, 1))
        result = validator.audit(resume)
        # A current (Present) role suppresses the newest-entry staleness check, and education-age is
        # no longer flagged (removed as noise) → NOT stale at all.
        assert result.staleness_issues == []
        assert not result.is_stale

    def test_audit_entries_from_different_sections(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(
            raw_text=(
                "EXPERIENCE\nEngineer\nCorp\n2018 - 2022\n\nEDUCATION\nBS CS\nMIT\n2014 - 2018"
            )
        )
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        sections = {e.section for e in result.entries}
        assert "Experience" in sections
        assert "Education" in sections

    def test_audit_section_detection_case_insensitive(self):
        from job_applicator.documents.resume_tailor import ResumeDateValidator

        resume = ResumeData(raw_text="experience\nEngineer\nCorp\n2020 - 2023")
        validator = ResumeDateValidator()
        result = validator.audit(resume)
        assert len(result.entries) == 1
        assert result.entries[0].section == "Experience"


async def test_resume_tailor_call_llm_uses_breaker() -> None:
    """Cycle 2b: ResumeTailor routes its LLM calls through the injected breaker, so
    repeated failures open it and a subsequent call fails fast (CircuitOpenError)
    without hitting the endpoint — the app's largest LLM calls are now protected."""
    from job_applicator.config import LLMConfig
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitBreaker, CircuitOpenError, LLMRuntime

    runtime = LLMRuntime(breaker=CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=60.0))
    tailor = ResumeTailor(LLMConfig(), runtime=runtime)
    assert tailor._breaker is runtime.breaker

    # acompletion fails → _call_llm wraps as LLMError → breaker records the failure.
    with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("down")):
        with pytest.raises(LLMError):
            await tailor._call_llm("prompt")

    # threshold=1 → breaker now OPEN → next call fails fast, endpoint untouched.
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        with pytest.raises(CircuitOpenError):
            await tailor._call_llm("prompt")
        mock_acomp.assert_not_called()


def test_cover_letter_and_tailor_share_one_runtime_breaker() -> None:
    """Cycle 2b: one LLMRuntime passed to both consumers yields ONE shared breaker —
    the mechanism the batch/tailor commands rely on (they build a single runtime via
    _make_runtime and pass it to both CoverLetterGenerator and ResumeTailor, so a
    down endpoint trips one breaker that guards the whole run)."""
    from job_applicator.config import LLMConfig
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.utils.llm import CircuitBreaker, LLMRuntime

    runtime = LLMRuntime(breaker=CircuitBreaker(name="shared"))
    gen = CoverLetterGenerator(LLMConfig(), runtime=runtime)
    tailor = ResumeTailor(LLMConfig(), runtime=runtime)
    assert gen._breaker is tailor._breaker
    assert gen._breaker is runtime.breaker


@pytest.mark.asyncio
async def test_refine_passes_style_guide_to_prompt(sample_resume, sample_job):
    """Cycle 2b polish: refinements must preserve the writing style guide."""
    from job_applicator.documents.style_analyzer import StyleAnalyzer
    from job_applicator.models import StyleGuide

    config = LLMConfig(api_base="http://localhost:8000/v1", model="test")
    tailor = ResumeTailor(config)

    current = TailoredResume(
        original_path="",
        tailored_text="Original tailored text",
        job_title=sample_job.title,
        job_company=sample_job.company,
        match_score=0.75,
        semantic_score=0.0,
        skill_score=0.0,
        matched_skills=["Windows"],
        missing_skills=[],
        changes_summary="Initial tailoring",
        attempt=1,
    )

    style = StyleGuide(
        tone="casual",
        sentence_structure="short",
        vocabulary_level="simple",
        paragraph_style="brief",
        formatting_notes="",
        sample_paragraph="",
    )

    captured_prompt = ""

    async def _capture_call(prompt: str, temperature: float = 0.7):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "Refined text"

    with patch.object(tailor, "_call_llm", side_effect=_capture_call):
        with patch.object(tailor, "_summarize_changes", new_callable=AsyncMock) as mock_changes:
            mock_changes.return_value = "Refined based on feedback"
            await tailor.refine(
                sample_resume,
                current,
                "Make it more concise",
                sample_job,
                matcher=_mock_matcher(sample_job),
                style_guide=style,
            )

    style_section = StyleAnalyzer.format_style_for_prompt(style)
    assert style_section in captured_prompt
    assert "Maintain this writing style" in captured_prompt


class TestEmptySectionStripping:
    """Tests for removing empty Certifications/Languages sections."""

    def test_strip_empty_certifications_languages_removes_both_when_absent(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original = "Name\nSkills\nPython\nExperience\nJob"
        tailored = (
            "Name\n"
            "Skills\nPython\n"
            "Experience\nJob\n"
            "**Certifications**\n"
            "**Languages**\n"
            "References\nAvailable"
        )
        result = tailor._strip_empty_certifications_languages(tailored, original)
        assert "**Certifications**" not in result
        assert "**Languages**" not in result
        assert "Skills" in result
        assert "Experience" in result
        assert "References" in result

    def test_strip_empty_certifications_keeps_section_when_original_has_it(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original = "Name\nCertifications\nAWS CPA\n"
        tailored = "Name\n**Certifications**\nAWS CPA\n"
        result = tailor._strip_empty_certifications_languages(tailored, original)
        assert "**Certifications**" in result

    def test_strip_empty_languages_keeps_section_when_original_has_it(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original = "Name\nLanguages\nEnglish, French\n"
        tailored = "Name\n**Languages**\nEnglish, French\n"
        result = tailor._strip_empty_certifications_languages(tailored, original)
        assert "**Languages**" in result

    def test_strip_unbacked_optional_sections_removes_absent_volunteer(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original = "Name\nSkills\nPython\nExperience\nJob"
        tailored = "Name\nExperience\nJob\nVolunteer\n*None*\nSkills\nPython"

        result = tailor._strip_unbacked_optional_sections(tailored, original)

        assert "Volunteer" not in result
        assert "*None*" not in result
        assert "Skills" in result

    def test_validate_skills_keeps_comma_separated_original_skills(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original_skills = ["Python", "FastAPI", "PostgreSQL", "Docker"]
        tailored = "**Skills**\n• Python, FastAPI, PostgreSQL, Docker\n\n**Experience**\n"
        result = tailor._validate_skills(tailored, original_skills)
        assert "Python, FastAPI, PostgreSQL, Docker" in result

    def test_validate_skills_drops_hallucinated_skills_in_comma_list(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original_skills = ["Python", "FastAPI"]
        tailored = "**Skills**\n• Python, Kubernetes, FastAPI\n\n**Experience**\n"
        result = tailor._validate_skills(tailored, original_skills)
        assert "Python" in result
        assert "FastAPI" in result
        assert "Kubernetes" not in result

    def test_validate_skills_keeps_extra_words_when_core_skill_matches(self, llm_config):
        tailor = ResumeTailor(llm_config)
        original_skills = ["Python", "Docker"]
        tailored = "**Skills**\n• Python (advanced)\n• Docker & Kubernetes\n"
        result = tailor._validate_skills(tailored, original_skills)
        assert "Python (advanced)" in result
        assert "Docker & Kubernetes" in result


@pytest.mark.asyncio
async def test_summarize_changes_raises_not_fabricated(llm_config, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A summary LLM failure RAISES the typed error — never returns a fabricated
    'summary generation failed' string that masks the failure as a real summary."""
    from job_applicator.exceptions import LLMError

    tailor = ResumeTailor(llm_config)

    async def _boom(*_a: object, **_k: object) -> str:
        raise LLMError("summary call failed")

    monkeypatch.setattr(tailor, "_call_llm", _boom)
    with pytest.raises(LLMError):
        await tailor._summarize_changes("original resume text", "tailored resume text")


class TestStripUnearnedCredentials:
    """Deterministic backstop: remove credential/status claims the candidate doesn't hold from the
    tailored SUMMARY — the freeform section the structured-section guards don't cover. Surfaced by
    a live tailor that emitted 'Accredited security operations professional' for a candidate with
    only in-progress coursework."""

    def test_strips_unearned_credential_and_keeps_sentence_grammatical(self, llm_config) -> None:
        tailor = ResumeTailor(llm_config)
        original = "SUMMARY\nOperations professional seeking a security role.\n\nSKILLS\n• Linux\n"
        tailored = (
            "SUMMARY\nAccredited security operations professional with 10+ years.\n\n"
            "SKILLS\n• Linux\n"
        )
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "accredited" not in out.lower()  # the unearned credential is gone
        # the sentence is repaired (capitalized, no leading gap, no double space)
        assert "Security operations professional with 10+ years." in out
        assert "• Linux" in out  # other sections untouched

    def test_preserves_credential_the_candidate_actually_holds(self, llm_config) -> None:
        tailor = ResumeTailor(llm_config)
        original = "SUMMARY\nAnalyst.\n\nCERTIFICATIONS\n• Certified Ethical Hacker\n"
        tailored = "SUMMARY\nCertified security analyst seeking a SOC role.\n\nSKILLS\n• Nmap\n"
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "Certified security analyst" in out  # a real, held credential is kept

    def test_benign_credential_word_elsewhere_does_not_license_overclaim(self, llm_config) -> None:
        # 'certified' as a verb in an EXPERIENCE bullet must NOT license a summary credential claim
        # (the whole-document leak the summary-scoping closes).
        tailor = ResumeTailor(llm_config)
        original = (
            "SUMMARY\nSupport specialist.\n\n"
            "EXPERIENCE\nAgent\n• Certified that tickets were resolved before closing.\n"
        )
        tailored = (
            "SUMMARY\nCertified support specialist seeking a security role.\n\nSKILLS\n• Linux\n"
        )
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "Certified support specialist" not in out
        assert "Support specialist seeking a security role." in out

    def test_handles_x_and_y_credential_phrase(self, llm_config) -> None:
        tailor = ResumeTailor(llm_config)
        original = "SUMMARY\nAnalyst.\n\nSKILLS\n• Linux\n"
        tailored = "SUMMARY\nCertified and licensed analyst with experience.\n\nSKILLS\n• Linux\n"
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "certified" not in out.lower() and "licensed" not in out.lower()
        assert "Analyst with experience." in out  # 'X and Y' collapses cleanly

    def test_only_scrubs_summary_not_other_sections(self, llm_config) -> None:
        tailor = ResumeTailor(llm_config)
        original = "SUMMARY\nAnalyst.\n\nSKILLS\n• Linux\n"
        tailored = (
            "SUMMARY\nAccredited analyst seeking a role.\n\n"
            "EXPERIENCE\nAuditor\n• Certified compliance reports each quarter.\n"
        )
        out = tailor._strip_unearned_credentials(tailored, original)
        assert "Accredited analyst" not in out  # summary scrubbed
        assert "Certified compliance reports each quarter." in out  # experience bullet untouched

    def test_strips_overclaim_under_unlabelled_and_variant_headers(self, llm_config) -> None:
        """Covers a header-less leading paragraph and odd/missing summary labels — not just a bare
        'SUMMARY' — since a 4B that overclaims often omits or renames the header (the silent-bypass
        the adversarial review caught: the scrub region is the leading block, not one header)."""
        tailor = ResumeTailor(llm_config)
        original = "Jane Roe\njane@x.com\n\nOperations lead.\n\nSKILLS\n• Linux\n"
        for header in ("", "PROFILE SUMMARY", "About", "Summary:"):
            lead = f"{header}\n" if header else ""
            tailored = (
                "Jane Roe\njane@x.com\n\n"
                f"{lead}"
                "Accredited security analyst seeking a SOC role.\n\nSKILLS\n• Linux\n"
            )
            out = tailor._strip_unearned_credentials(tailored, original)
            assert "accredited" not in out.lower(), f"header={header!r}"
            assert "Security analyst seeking a SOC role." in out, f"header={header!r}"
            assert "• Linux" in out  # the skills section is never scrubbed
