"""Deterministic prose realization for selected source-resume facts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from job_applicator.models import SourceBackedStatement, SourceFact


def _sentence(value: str) -> str:
    text = value.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def realize_resume_statement(fact: SourceFact) -> SourceBackedStatement:
    """Render one source fact as a resume-summary sentence without paraphrasing it."""

    return SourceBackedStatement(text=_sentence(fact.text), fact_ids=[fact.fact_id])


def realize_cover_statement(
    fact: SourceFact,
    *,
    language: str,
    occurrence: int = 0,
) -> SourceBackedStatement:
    """Render one selected fact with a fixed grammatical frame and no free-form rewriting."""

    if language == "French":
        first_prefixes = {
            "experience": "Mon expérience comprend notamment ce travail — ",
            "projects": "Mes projets comprennent notamment cet exemple — ",
            "education": "Ma formation comprend notamment cet élément — ",
        }
        later_prefixes = {
            "experience": "Mon expérience comprend également ce travail — ",
            "projects": "Mes projets comprennent également cet exemple — ",
            "education": "Ma formation comprend également cet élément — ",
        }
        prefixes = later_prefixes if occurrence else first_prefixes
        prefix = prefixes.get(fact.kind, "Mon parcours comprend notamment cet élément — ")
        text = f"{prefix}{fact.text}"
    elif fact.kind == "experience":
        source = fact.text
        text = f"I {source[:1].lower()}{source[1:]}"
    else:
        first_prefixes = {
            "projects": "My project work includes this example — ",
            "education": "My education includes this record — ",
        }
        later_prefixes = {
            "projects": "My project work also includes this example — ",
            "education": "My education also includes this record — ",
        }
        prefixes = later_prefixes if occurrence else first_prefixes
        prefix = prefixes.get(fact.kind, "My background includes this record — ")
        text = f"{prefix}{fact.text}"
    return SourceBackedStatement(text=_sentence(text), fact_ids=[fact.fact_id])


def realize_cover_statements(
    facts: Sequence[SourceFact], *, language: str
) -> list[SourceBackedStatement]:
    """Render an ordered evidence plan while varying only fixed discourse frames."""

    seen: Counter[str] = Counter()
    statements: list[SourceBackedStatement] = []
    for fact in facts:
        statements.append(
            realize_cover_statement(
                fact,
                language=language,
                occurrence=seen[fact.kind],
            )
        )
        seen[fact.kind] += 1
    return statements
