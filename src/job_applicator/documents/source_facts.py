"""Build a generic, context-preserving fact catalog from a source resume."""

from __future__ import annotations

import re

from job_applicator.models import (
    JobListing,
    ResumeData,
    SourceFact,
    SourceFactCatalog,
    SourceFactKind,
)

_BULLET_RE = re.compile(r"^\s*[\u2022*+-]\s*")
_DATE_RANGE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b\s*(?:[-\u2013\u2014]|to|a)\s*"
    r"(?:present|current|actuel|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9+#./-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SECTION_ALIASES: dict[str, SourceFactKind] = {
    "summary": "summary",
    "professional summary": "summary",
    "profile": "summary",
    "profil": "summary",
    "technical skills": "skills",
    "skills": "skills",
    "competences": "skills",
    "core competencies": "skills",
    "projects": "projects",
    "projects & home lab": "projects",
    "project experience": "projects",
    "professional experience": "experience",
    "work experience": "experience",
    "experience": "experience",
    "education": "education",
    "education & certifications": "education",
    "formation": "education",
    "formation et certifications": "education",
    "languages": "languages",
    "langues": "languages",
}
_STOPWORDS = {
    "and",
    "avec",
    "dans",
    "des",
    "for",
    "from",
    "les",
    "pour",
    "the",
    "this",
    "with",
}


def _fold(value: str) -> str:
    import unicodedata

    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _normalize_line(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\f", " ")).strip()


def _section_kind(value: str) -> SourceFactKind | None:
    normalized = _fold(value).strip(" *_#:-")
    return _SECTION_ALIASES.get(normalized)


def _looks_like_entry_header(value: str) -> bool:
    return bool(_DATE_RANGE_RE.search(_fold(value)))


def _context_text(lines: list[str]) -> str:
    return " | ".join(lines[-3:])


def build_source_fact_catalog(resume: ResumeData) -> SourceFactCatalog:
    """Extract every non-empty source line while preserving bullet and entry context."""

    facts: list[SourceFact] = []
    kind: SourceFactKind = "identity"
    entry_context: list[str] = []
    entry_has_bullets = False
    pending_bullet = ""
    pending_context = ""
    pending_claim_eligible = False
    structured_summary_added = False
    structured_skills_added = False

    def add_fact(
        text: str,
        *,
        fact_kind: SourceFactKind,
        context: str = "",
        claim_eligible: bool = False,
    ) -> None:
        normalized = _normalize_line(text)
        if not normalized:
            return
        facts.append(
            SourceFact(
                fact_id=f"SRC-{len(facts) + 1:03d}",
                kind=fact_kind,
                text=normalized,
                context=context,
                claim_eligible=claim_eligible,
            )
        )

    def flush_bullet() -> None:
        nonlocal pending_bullet, pending_context, pending_claim_eligible
        if pending_bullet:
            add_fact(
                pending_bullet,
                fact_kind=kind,
                context=pending_context,
                claim_eligible=pending_claim_eligible,
            )
        pending_bullet = ""
        pending_context = ""
        pending_claim_eligible = False

    for raw_line in resume.raw_text.replace("\f", "\n").splitlines():
        line = _normalize_line(raw_line)
        if not line:
            flush_bullet()
            continue

        section = _section_kind(line)
        if section is not None:
            flush_bullet()
            kind = section
            entry_context = []
            entry_has_bullets = False
            if kind == "summary" and resume.summary and not structured_summary_added:
                for sentence in _SENTENCE_RE.split(_normalize_line(resume.summary)):
                    add_fact(sentence, fact_kind="summary")
                structured_summary_added = True
            elif kind == "skills" and resume.skills and not structured_skills_added:
                for skill in resume.skills:
                    add_fact(skill, fact_kind="skills")
                structured_skills_added = True
            continue

        if (kind == "summary" and structured_summary_added) or (
            kind == "skills" and structured_skills_added
        ):
            continue

        bullet_match = _BULLET_RE.match(raw_line)
        if bullet_match:
            flush_bullet()
            pending_bullet = _normalize_line(raw_line[bullet_match.end() :])
            pending_context = _context_text(entry_context)
            pending_claim_eligible = True
            entry_has_bullets = True
            continue

        # Education details are often indented, soft-wrapped prose rather than marked bullets.
        # Keep one indented block atomic so qualifiers and credential status cannot be separated
        # from the coursework they govern.
        if kind == "education" and raw_line[:1].isspace():
            if pending_bullet:
                pending_bullet = f"{pending_bullet} {line}"
            else:
                pending_bullet = line
                pending_context = _context_text(entry_context)
                pending_claim_eligible = True
            continue

        if pending_bullet and not _looks_like_entry_header(line):
            pending_bullet = f"{pending_bullet} {line}"
            continue
        flush_bullet()

        if kind in {"experience", "education"}:
            starts_new_entry = entry_has_bullets or (
                _looks_like_entry_header(line)
                and any(_looks_like_entry_header(item) for item in entry_context)
            )
            if starts_new_entry:
                entry_context = []
                entry_has_bullets = False
            entry_context.append(line)
            context = _context_text(entry_context[:-1])
        else:
            context = ""
        add_fact(line, fact_kind=kind, context=context)

    flush_bullet()
    return SourceFactCatalog(facts=facts)


def format_source_fact_catalog(catalog: SourceFactCatalog) -> str:
    """Render facts with targeting context for the selection stage."""

    lines: list[str] = []
    for fact in catalog.facts:
        context = f" | context={fact.context}" if fact.context else ""
        lines.append(f"[{fact.fact_id}] kind={fact.kind}{context} | fact={fact.text}")
    return "\n".join(lines)


def format_job_target_context(job: JobListing, *, max_description_chars: int = 6_000) -> str:
    """Preserve both the role overview and the requirements-heavy tail for selection prompts."""

    description = job.description.strip()
    if len(description) > max_description_chars:
        head_chars = max_description_chars // 2
        tail_chars = max_description_chars - head_chars
        description = (
            description[:head_chars].rstrip()
            + "\n[... middle omitted for prompt budget ...]\n"
            + description[-tail_chars:].lstrip()
        )
    requirements = ", ".join(job.requirements) or "Not separately provided"
    return (
        f"Job title: {job.title}\n"
        f"Company: {job.company}\n"
        f"Job description:\n{description}\n"
        f"Requirements: {requirements}"
    )


def is_substantive_source_fact(fact: SourceFact) -> bool:
    """Return whether a primary body fact can stand as generated prose evidence."""

    return fact.claim_eligible


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in (_fold(match.group(0)) for match in _TOKEN_RE.finditer(value))
        if len(token) >= 3 and token not in _STOPWORDS
    }


def select_relevant_source_facts(
    catalog: SourceFactCatalog,
    job: JobListing,
    *,
    max_chars: int = 8_000,
    max_facts: int | None = None,
    relevance_order: bool = False,
    include_identity: bool = True,
) -> SourceFactCatalog:
    """Fit a catalog to a prompt budget without tail truncation or source rewriting."""

    rendered = format_source_fact_catalog(catalog)
    if (
        len(rendered) <= max_chars
        and max_facts is None
        and not relevance_order
        and include_identity
    ):
        return catalog

    target_tokens = _tokens(" ".join([job.title, job.description, *job.requirements]))
    ranked: list[tuple[int, int, SourceFact]] = []
    for index, fact in enumerate(catalog.facts):
        fact_tokens = _tokens(fact.text)
        context_tokens = _tokens(fact.context)
        score = len(target_tokens & fact_tokens) * 10
        score += len(target_tokens & context_tokens) * 2
        if fact.kind in {"experience", "projects", "education"}:
            score += 2
        if fact.kind in {"identity", "summary"}:
            score += 1
        ranked.append((score, -index, fact))

    selected = [fact for fact in catalog.facts if include_identity and fact.kind == "identity"]
    used = len(format_source_fact_catalog(SourceFactCatalog(facts=selected)))
    selected_ids = {fact.fact_id for fact in selected}
    for _score, _negative_index, fact in sorted(ranked, reverse=True):
        if fact.fact_id in selected_ids:
            continue
        line_length = len(format_source_fact_catalog(SourceFactCatalog(facts=[fact]))) + 1
        if selected and used + line_length > max_chars:
            continue
        selected.append(fact)
        selected_ids.add(fact.fact_id)
        used += line_length
        if used >= max_chars or (max_facts is not None and len(selected) >= max_facts):
            break

    if relevance_order:
        return SourceFactCatalog(facts=selected)
    ordered = [fact for fact in catalog.facts if fact.fact_id in selected_ids]
    return SourceFactCatalog(facts=ordered)
