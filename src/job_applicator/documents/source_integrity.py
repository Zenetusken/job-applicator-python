"""Generic deterministic integrity checks against an authoritative source resume."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from job_applicator.models import ResumeData

_HIGH_RISK_NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$€£]\s*)?\d(?:[\d,.]*\d)?"
    r"(?:\s*%|\+\s*(?:years?|ans|clients?|tickets?|cases?|cas|users?|utilisateurs?)?"
    r"|\s+(?:years?|ans|clients?|tickets?|cases?|cas|users?|utilisateurs?|daily|par\s+jour))",
    re.IGNORECASE,
)
_NUMBER_CORE_RE = re.compile(r"\d(?:[\d,.]*\d)?")


@dataclass(frozen=True)
class SourceIntegrityReport:
    """Deterministic source-integrity evidence surfaced in packet certification."""

    source_checked: bool
    failures: list[str]
    warnings: list[str]
    missing_contact_fields: list[str]
    missing_experience_companies: list[str]
    missing_education_institutions: list[str]
    unsupported_numeric_claims: list[str]


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()


def _entity_present(entity: str, generated: str) -> bool:
    folded_generated = _fold(generated)
    candidates = [entity, entity.split("(", 1)[0].strip()]
    return any(candidate and _fold(candidate) in folded_generated for candidate in candidates)


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _missing_source_entities(
    source: ResumeData,
    generated_resume: str,
) -> tuple[list[str], list[str]]:
    companies = _ordered_unique([entry.company.strip() for entry in source.experience])
    institutions = _ordered_unique([entry.institution.strip() for entry in source.education])
    missing_companies = [
        company for company in companies if not _entity_present(company, generated_resume)
    ]
    missing_institutions = [
        institution
        for institution in institutions
        if not _entity_present(institution, generated_resume)
    ]
    return missing_companies, missing_institutions


def _missing_contact_fields(source: ResumeData, generated_resume: str) -> list[str]:
    missing: list[str] = []
    generated_folded = _fold(generated_resume)
    if source.name and _fold(source.name) not in generated_folded:
        missing.append("name")
    if source.email and source.email.casefold() not in generated_resume.casefold():
        missing.append("email")
    if source.phone:
        source_digits = re.sub(r"\D", "", source.phone)
        generated_digits = re.sub(r"\D", "", generated_resume)
        if source_digits and source_digits not in generated_digits:
            missing.append("phone")
    return missing


def _numeric_cores(value: str) -> set[str]:
    return {
        match.group(0).replace(",", ".")
        for claim in _HIGH_RISK_NUMBER_RE.findall(value)
        if (match := _NUMBER_CORE_RE.search(claim)) is not None
    }


def _unsupported_numeric_claims(
    generated_resume: str,
    generated_cover: str,
    source_text: str,
) -> list[str]:
    source_numbers = _numeric_cores(source_text)
    unsupported: list[str] = []
    for document, text in (("resume", generated_resume), ("cover_letter", generated_cover)):
        for claim in _HIGH_RISK_NUMBER_RE.findall(text):
            match = _NUMBER_CORE_RE.search(claim)
            if match is None or match.group(0).replace(",", ".") in source_numbers:
                continue
            normalized_claim = re.sub(r"\s+", " ", claim).strip()
            unsupported.append(f"{document}: {normalized_claim}")
    return _ordered_unique(unsupported)


def assess_source_integrity(
    *,
    source: ResumeData | None,
    generated_resume: str,
    generated_cover: str,
    require_resume_structure: bool = True,
) -> SourceIntegrityReport:
    """Check source-preservation invariants without attempting to grade prose."""

    if source is None:
        return SourceIntegrityReport(
            source_checked=False,
            failures=[],
            warnings=[],
            missing_contact_fields=[],
            missing_experience_companies=[],
            missing_education_institutions=[],
            unsupported_numeric_claims=[],
        )

    missing_contacts = (
        _missing_contact_fields(source, generated_resume) if require_resume_structure else []
    )
    missing_companies, missing_institutions = (
        _missing_source_entities(source, generated_resume) if require_resume_structure else ([], [])
    )
    unsupported_numbers = _unsupported_numeric_claims(
        generated_resume,
        generated_cover,
        source.raw_text,
    )
    failures = [
        *(f"missing source contact field: {field}" for field in missing_contacts),
        *(f"missing source employer: {company}" for company in missing_companies),
        *(f"missing source institution: {institution}" for institution in missing_institutions),
        *(f"unsupported numeric claim: {claim}" for claim in unsupported_numbers),
    ]
    return SourceIntegrityReport(
        source_checked=True,
        failures=failures,
        warnings=[],
        missing_contact_fields=missing_contacts,
        missing_experience_companies=missing_companies,
        missing_education_institutions=missing_institutions,
        unsupported_numeric_claims=unsupported_numbers,
    )
