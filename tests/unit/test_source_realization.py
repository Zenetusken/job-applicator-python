"""Tests for deterministic realization of selected source facts."""

from __future__ import annotations

from job_applicator.documents.source_realization import (
    realize_cover_statement,
    realize_cover_statements,
    realize_resume_statement,
)
from job_applicator.models import SourceFact


def test_resume_realization_preserves_fact_text() -> None:
    fact = SourceFact(
        fact_id="SRC-007",
        kind="education",
        text="Certification exam pending",
        context="Coursework | College",
    )

    statement = realize_resume_statement(fact)

    assert statement.text == "Certification exam pending."
    assert statement.fact_ids == ["SRC-007"]
    assert fact.context not in statement.text


def test_cover_experience_realization_only_adds_first_person_frame() -> None:
    fact = SourceFact(
        fact_id="SRC-010",
        kind="experience",
        text="Triaged and escalated complex issues per documented procedures",
        context="Support Specialist | Acme",
    )

    statement = realize_cover_statement(fact, language="English")

    assert statement.text == ("I triaged and escalated complex issues per documented procedures.")
    assert "ensuring" not in statement.text
    assert "Acme" not in statement.text


def test_cover_project_realization_keeps_source_phrase_verbatim() -> None:
    fact = SourceFact(
        fact_id="SRC-011",
        kind="projects",
        text="Home cybersecurity lab - a multi-VM environment",
        claim_eligible=True,
    )

    statement = realize_cover_statement(fact, language="English")

    assert statement.text == (
        "My project work includes this example — Home cybersecurity lab - a multi-VM environment."
    )


def test_cover_plan_varies_only_the_fixed_frame_for_repeated_kinds() -> None:
    facts = [
        SourceFact(fact_id="SRC-011", kind="projects", text="Built a DNS lab"),
        SourceFact(fact_id="SRC-012", kind="projects", text="Built a routing lab"),
    ]

    statements = realize_cover_statements(facts, language="English")

    assert statements[0].text == "My project work includes this example — Built a DNS lab."
    assert statements[1].text == (
        "My project work also includes this example — Built a routing lab."
    )
    assert [statement.fact_ids for statement in statements] == [["SRC-011"], ["SRC-012"]]
