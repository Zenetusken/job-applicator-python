"""Tests for the immutable source-fact catalog used by document prompts."""

from __future__ import annotations

from job_applicator.documents.source_facts import (
    build_source_fact_catalog,
    format_job_target_context,
    format_source_fact_catalog,
    is_substantive_source_fact,
    select_relevant_source_facts,
)
from job_applicator.models import EducationEntry, ExperienceEntry, JobBoard, JobListing, ResumeData


def _resume() -> ResumeData:
    return ResumeData(
        raw_text=(
            "ALEX MORGAN\n"
            "alex@example.com\n\n"
            "EXPERIENCE\n"
            "Acme Support\n"
            "Support Analyst 2020 - 2022\n"
            "• Resolved account and connectivity tickets by phone and email.\n"
            "• Escalated unresolved incidents with clear notes.\n\n"
            "Legacy Systems\n"
            "Technical Support Advisor 2013 - 2015\n"
            "• Conducted remote desktop sessions for PC and Mac issues.\n\n"
            "EDUCATION\n"
            "Metro College\n"
            "Diploma, Information Technology 2018 - 2020\n"
        ),
        name="ALEX MORGAN",
        email="alex@example.com",
        experience=[
            ExperienceEntry(title="Support Analyst", company="Acme Support"),
            ExperienceEntry(title="Technical Support Advisor", company="Legacy Systems"),
        ],
        education=[
            EducationEntry(
                degree="Diploma, Information Technology",
                institution="Metro College",
            )
        ],
    )


def test_catalog_preserves_tail_facts_and_entry_context() -> None:
    catalog = build_source_fact_catalog(_resume())

    remote = next(fact for fact in catalog.facts if "remote desktop" in fact.text)

    assert "Legacy Systems" in remote.context
    assert "Technical Support Advisor" in remote.context
    assert catalog.facts[-1].text == "Diploma, Information Technology 2018 - 2020"


def test_catalog_rendering_does_not_rewrite_source_text() -> None:
    catalog = build_source_fact_catalog(_resume())
    rendered = format_source_fact_catalog(catalog)

    assert "fact=Escalated unresolved incidents with clear notes." in rendered
    assert "[SRC-001]" in rendered


def test_substantive_fact_filter_excludes_entry_headers() -> None:
    catalog = build_source_fact_catalog(_resume())

    assert any(is_substantive_source_fact(fact) for fact in catalog.facts)
    assert not is_substantive_source_fact(catalog.facts[-1])


def test_substantive_fact_filter_excludes_prior_summary_prose() -> None:
    resume = ResumeData(
        raw_text=(
            "ALEX MORGAN\n\nSUMMARY\nSupport specialist.\n\nPROJECTS\n"
            "• Built a troubleshooting lab."
        ),
        summary="Support specialist.",
    )
    catalog = build_source_fact_catalog(resume)

    summary = next(fact for fact in catalog.facts if fact.kind == "summary")
    project = next(fact for fact in catalog.facts if fact.kind == "projects")

    assert not is_substantive_source_fact(summary)
    assert is_substantive_source_fact(project)


def test_catalog_recognizes_french_projects_as_claim_evidence() -> None:
    resume = ResumeData(
        raw_text=(
            "ALEX MORGAN\n\nCOMPÉTENCES\nSIEM · Wireshark\n\nPROJETS\n"
            "• Construit un laboratoire de cybersécurité à domicile."
        ),
        skills=["SIEM", "Wireshark"],
    )

    project = next(
        fact for fact in build_source_fact_catalog(resume).facts if "laboratoire" in fact.text
    )

    assert project.kind == "projects"
    assert is_substantive_source_fact(project)


def test_catalog_uses_parsed_summary_and_skills_instead_of_wrapped_fragments() -> None:
    resume = ResumeData(
        raw_text=(
            "ALEX MORGAN\n\nSUMMARY\nSecurity operations and incident\n"
            "response experience.\n\nSKILLS\nMicrosoft 365 · network\nmonitoring"
        ),
        summary="Security operations and incident response experience.",
        skills=["Microsoft 365", "network monitoring"],
    )

    catalog = build_source_fact_catalog(resume)
    texts = [fact.text for fact in catalog.facts]

    assert "Security operations and incident response experience." in texts
    assert "Security operations and incident" not in texts
    assert "Microsoft 365" in texts
    assert "network monitoring" in texts


def test_catalog_keeps_indented_education_detail_and_status_atomic() -> None:
    resume = ResumeData(
        raw_text=(
            "ALEX MORGAN\n\nEDUCATION\n"
            "Cybersecurity Certificate 2024 - Present\n"
            "Metro College\n"
            "    Completed: Security Fundamentals, Server Security, and Networking — incl. a\n"
            "    hands-on SIEM lab and incident-response exercise.\n\n"
            "Cisco Coursework 2023 - 2024\n"
            "Dawson College\n"
            "    Full networking curriculum covering routing and switching\n"
            "    (certification exam pending) — plus Linux administration."
        )
    )

    facts = build_source_fact_catalog(resume).facts
    detail = next(fact for fact in facts if fact.text.startswith("Completed:"))
    networking = next(fact for fact in facts if fact.text.startswith("Full networking"))

    assert detail.text.endswith("hands-on SIEM lab and incident-response exercise.")
    assert "certification exam pending" in networking.text
    assert not any(fact.text.startswith("hands-on SIEM") for fact in facts)


def test_relevance_selection_keeps_identity_and_relevant_tail_fact() -> None:
    job = JobListing(
        title="Desktop Support Technician",
        company="Target Co",
        url="https://example.test/job",
        board=JobBoard.INDEED,
        description="Troubleshoot remote desktop, PC, and Mac issues.",
    )

    selected = select_relevant_source_facts(
        build_source_fact_catalog(_resume()), job, max_chars=500
    )
    text = format_source_fact_catalog(selected)

    assert "ALEX MORGAN" in text
    assert "alex@example.com" in text
    assert "remote desktop sessions" in text


def test_relevance_selection_can_supply_only_ranked_factual_context() -> None:
    job = JobListing(
        title="Desktop Support Technician",
        company="Target Co",
        url="https://example.test/job",
        board=JobBoard.INDEED,
        description="Troubleshoot remote desktop, PC, and Mac issues.",
    )

    selected = select_relevant_source_facts(
        build_source_fact_catalog(_resume()),
        job,
        max_chars=10_000,
        max_facts=2,
        relevance_order=True,
        include_identity=False,
    )

    assert len(selected.facts) == 2
    assert selected.facts[0].text == "Conducted remote desktop sessions for PC and Mac issues."
    assert all(fact.kind != "identity" for fact in selected.facts)


def test_job_target_context_preserves_description_head_and_requirements_tail() -> None:
    description = "ROLE OVERVIEW " + ("middle " * 1_000) + "FINAL REQUIREMENTS networking DNS"
    job = JobListing(
        title="Network Analyst",
        company="Target Co",
        url="https://example.test/job",
        board=JobBoard.INDEED,
        description=description,
        requirements=["routing", "incident documentation"],
    )

    context = format_job_target_context(job, max_description_chars=500)

    assert "ROLE OVERVIEW" in context
    assert "FINAL REQUIREMENTS networking DNS" in context
    assert "routing, incident documentation" in context
    assert "middle omitted for prompt budget" in context


def test_relevance_ranking_weights_fact_text_over_entry_context() -> None:
    catalog = build_source_fact_catalog(
        ResumeData(
            raw_text=(
                "ALEX MORGAN\n\nEXPERIENCE\n"
                "Technical Support and Sales | Acme | 2020 - 2022\n"
                "• Drove new-client acquisition and retention.\n\n"
                "Operations | Other Co | 2018 - 2020\n"
                "• Administered Windows endpoints and resolved desktop incidents."
            )
        )
    )
    evidence = type(catalog)(
        facts=[fact for fact in catalog.facts if is_substantive_source_fact(fact)]
    )
    job = JobListing(
        title="Desktop Support",
        company="Target Co",
        url="https://example.test/job",
        board=JobBoard.INDEED,
        description="Windows endpoint support and desktop incident resolution",
    )

    selected = select_relevant_source_facts(
        evidence,
        job,
        max_facts=2,
        relevance_order=True,
        include_identity=False,
    )

    assert selected.facts[0].text.startswith("Administered Windows endpoints")
