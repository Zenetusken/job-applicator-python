"""Heuristic quality gate for generated CV and cover-letter artifacts.

Single-artifact mode is a smoke check for a generated CV and/or cover letter.
Private packet-set mode scores generated CV+cover-letter packets against a
local manifest that is not committed to the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from job_applicator.utils.language import detect_language

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"[\+]?[\d\s\-().]{10,}")
_PLACEHOLDER_RE = re.compile(r"\b(lorem ipsum|todo|tbd|placeholder|your name|company name)\b", re.I)
_SECTION_ALIASES = {
    "experience": ("experience", "expérience"),
    "education": ("education", "éducation", "formation"),
    "skills": ("skills", "compétences", "competences"),
}
_CLOSINGS = ("sincerely", "best regards", "regards", "cordialement", "merci")
_DEFAULT_PACKET_SET = "~/.job-applicator/document-quality-eval/packet-set.jsonl"
_DOCUMENT_QUALITY_SET_ENV = "DOCUMENT_QUALITY_SET"
_DIMENSIONS = ("usefulness", "specificity", "coherence", "writing_quality", "formatting_polish")
_DEFAULT_DIMENSION_FLOOR = 3.0
_DEFAULT_OVERALL_FLOOR = 3.0
_DEFAULT_MAX_ARTIFACT_AGE_DAYS = 14
_OPTIONAL_MIN_CASES = 1
_REQUIRED_MIN_CASES = 3
_PROVENANCE_FIELDS = (
    "run_id",
    "source_job_url",
    "template",
    "format",
    "model",
    "generator_version",
)
_GENERIC_COVER_PHRASES = (
    "i am excited to apply",
    "i am writing to express my interest",
    "perfect fit",
    "dynamic team",
    "fast-paced environment",
)
_STOPWORDS = {
    "about",
    "across",
    "also",
    "and",
    "are",
    "based",
    "can",
    "company",
    "experience",
    "for",
    "from",
    "have",
    "including",
    "into",
    "job",
    "looking",
    "more",
    "our",
    "role",
    "skills",
    "that",
    "the",
    "their",
    "this",
    "through",
    "will",
    "with",
    "work",
    "you",
    "your",
}
_COMPANY_STOPWORDS = {
    "and",
    "canada",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "group",
    "inc",
    "in",
    "llc",
    "ltd",
    "plc",
    "the",
}
_ENGLISH_MARKERS = {
    "and",
    "background",
    "client",
    "education",
    "experience",
    "for",
    "role",
    "skills",
    "support",
    "the",
    "with",
}
_FRENCH_MARKERS = {
    "avec",
    "compétences",
    "competences",
    "dans",
    "de",
    "des",
    "du",
    "expérience",
    "experience",
    "formation",
    "le",
    "les",
    "pour",
}


@dataclass(frozen=True)
class QualityReport:
    kind: str
    passed: bool
    score: int
    failures: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class LanguageQualityReport:
    expected: str
    checked_blocks: int
    mismatched_blocks: int
    mismatched_sections: list[str]


@dataclass(frozen=True)
class PacketQualityReport:
    packet_id: str
    passed: bool
    overall: float
    dimensions: dict[str, float]
    failures: list[str]
    warnings: list[str]
    resume: QualityReport
    cover_letter: QualityReport
    category: str | None = None
    language: str | None = None
    generated_at: str | None = None
    generated_at_source: str = "artifact_mtime"
    provenance: dict[str, str] | None = None
    language_quality: LanguageQualityReport | None = None


@dataclass(frozen=True)
class PacketSetThresholds:
    min_dimension_score: float
    min_overall_score: float
    min_cases: int
    max_artifact_age_days: int


@dataclass(frozen=True)
class PacketSetCoverage:
    required_categories: list[str]
    present_categories: list[str]
    missing_categories: list[str]
    required_languages: list[str]
    present_languages: list[str]
    missing_languages: list[str]


@dataclass(frozen=True)
class PacketSetFreshness:
    oldest_artifact_age_days: float | None
    newest_artifact_age_days: float | None
    stale_packet_ids: list[str]


@dataclass(frozen=True)
class PacketSetCertification:
    certified: bool
    mode: str
    required: bool
    thresholds: PacketSetThresholds
    coverage: PacketSetCoverage
    freshness: PacketSetFreshness
    certification_failures: list[str]
    certification_warnings: list[str]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9+#.-]*", text.lower())


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = text.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in low)
    return hits / len(keywords)


def _phrase_present(text: str, phrase: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", text.casefold())
    normalized_phrase = re.sub(r"\s+", " ", phrase.casefold()).strip()
    return bool(normalized_phrase and normalized_phrase in normalized_text)


def _clamp_dimension(value: float) -> float:
    return max(0.0, min(4.0, value))


def _sentence_lengths(text: str) -> list[int]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return [len(_words(sentence)) for sentence in sentences]


def _as_text_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    raise ValueError(f"{field} must be a list or comma-separated string")


def _case_value(case: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in case and case[name] not in (None, ""):
            return case[name]
    return None


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.casefold() if text else None


def _normalize_language_label(value: Any) -> str | None:
    label = _normalize_label(value)
    if label is None:
        return None
    return _language_code_from_label(label) or label


def _case_provenance(case: dict[str, Any]) -> dict[str, str] | None:
    provenance: dict[str, str] = {}
    for field in _PROVENANCE_FIELDS:
        value = _case_value(case, field)
        if value is not None:
            provenance[field] = str(value)
    return provenance or None


def _resolve_case_path(raw: Any, *, base_dir: Path, field: str) -> Path:
    if raw is None:
        raise ValueError(f"missing required {field}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _read_existing_text(path: Path, *, field: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{field} is not a file: {path}")
    return path.read_text(encoding="utf-8")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat_utc(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_generated_at(raw: Any, *, field: str = "generated_at") -> datetime:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp: {text}") from exc
    return _as_utc(parsed)


def _case_generated_at(
    case: dict[str, Any],
    *,
    resume_path: Path,
    cover_path: Path,
) -> tuple[str, str]:
    explicit = _case_value(case, "generated_at")
    if explicit is not None:
        return _isoformat_utc(_parse_generated_at(explicit)), "generated_at"
    oldest_mtime = min(resume_path.stat().st_mtime, cover_path.stat().st_mtime)
    generated_at = datetime.fromtimestamp(oldest_mtime, tz=UTC)
    return _isoformat_utc(generated_at), "artifact_mtime"


def _load_packet_records(packet_set: Path) -> list[dict[str, Any]]:
    text = packet_set.read_text(encoding="utf-8")
    records: list[Any]
    if packet_set.suffix == ".jsonl":
        records = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{packet_set}:{lineno}: invalid JSON: {exc}") from exc
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{packet_set}: invalid JSON: {exc}") from exc
        records = loaded.get("cases", []) if isinstance(loaded, dict) else loaded

    if not isinstance(records, list):
        raise ValueError("packet set must be a JSON list or JSONL objects")
    typed: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"packet case {index} must be an object")
        typed.append(record)
    return typed


def _keywords_from_job_description(job_description: str, *, limit: int = 12) -> list[str]:
    counts: dict[str, int] = {}
    for word in _words(job_description):
        if len(word) < 3 or word in _STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts, key=lambda item: (-counts[item], item))
    return ranked[:limit]


def _case_keywords(case: dict[str, Any], *, base_dir: Path) -> tuple[list[str], str]:
    keywords = _as_text_list(_case_value(case, "keywords", "required_keywords"), field="keywords")
    job_description = str(_case_value(case, "job_description", "description") or "")
    job_description_path = _case_value(case, "job_description_path", "description_path")
    if job_description_path is not None:
        path = _resolve_case_path(
            job_description_path,
            base_dir=base_dir,
            field="job_description_path",
        )
        job_description = (job_description + "\n" + _read_existing_text(path, field="job")).strip()
    if not keywords and job_description:
        keywords = _keywords_from_job_description(job_description)
    return keywords, job_description


def _case_coherence_terms(case: dict[str, Any], *, keywords: list[str]) -> list[str]:
    explicit = _as_text_list(
        _case_value(case, "coherence_terms", "shared_terms"),
        field="coherence_terms",
    )
    return explicit or keywords


def _company_terms(company: str) -> list[str]:
    words = [
        word
        for word in _words(company)
        if word not in _COMPANY_STOPWORDS and (len(word) >= 3 or word.isupper())
    ]
    terms = [company] if company else []
    terms.extend(word for word in words if word not in terms)
    return terms


def _target_mention_score(text: str, *, job_title: str, company: str) -> float:
    scores: list[float] = []
    if job_title:
        scores.append(1.0 if _phrase_present(text, job_title) else 0.0)
    company_terms = _company_terms(company)
    if company_terms:
        scores.append(1.0 if any(_phrase_present(text, term) for term in company_terms) else 0.0)
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def _name_mention_score(*, resume_text: str, cover_text: str, applicant_name: str) -> float:
    if not applicant_name:
        return 1.0
    return (0.5 if _phrase_present(resume_text, applicant_name) else 0.0) + (
        0.5 if _phrase_present(cover_text, applicant_name) else 0.0
    )


def _language_hint(text: str) -> str:
    words = _words(text)
    if not words:
        return "unknown"
    english = sum(1 for word in words if word in _ENGLISH_MARKERS)
    french = sum(1 for word in words if word in _FRENCH_MARKERS)
    if french >= 4 and french > english * 1.5:
        return "fr"
    if english >= 4 and english > french * 1.5:
        return "en"
    return "unknown"


def _language_coherence_score(resume_text: str, cover_text: str) -> float:
    resume_language = _language_hint(resume_text)
    cover_language = _language_hint(cover_text)
    if "unknown" in (resume_language, cover_language):
        return 1.0
    return 1.0 if resume_language == cover_language else 0.0


def _language_code_from_label(label: str | None) -> str | None:
    if label in {"fr", "french", "français", "francais"}:
        return "fr"
    if label in {"en", "english", "anglais"}:
        return "en"
    return None


def _ordered_unique(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


_RESUME_LANGUAGE_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "profile": ("profile", "profil", "summary", "professional summary", "resume", "résumé"),
    "skills": ("skills", "technical skills", "compétences", "competences"),
    "experience": (
        "experience",
        "professional experience",
        "work experience",
        "expérience",
        "expérience professionnelle",
        "experience professionnelle",
    ),
    "education": (
        "education",
        "éducation",
        "formation",
        "education & certifications",
        "education and certifications",
        "éducation & certifications",
        "formation et certifications",
    ),
    "languages": ("languages", "language", "langues"),
    "projects": (
        "projects",
        "project",
        "projets",
        "projects & home lab",
        "projets & laboratoire à domicile",
        "projets & laboratoire a domicile",
    ),
}


def _resume_language_section(line: str) -> str | None:
    label = re.sub(r"[*_`]", "", line).strip().strip(":")
    normalized = re.sub(r"\s+", " ", label.casefold())
    for section, aliases in _RESUME_LANGUAGE_SECTION_ALIASES.items():
        if normalized in aliases:
            return section
    return None


def _clean_language_block(line: str) -> str:
    cleaned = re.sub(r"^\s*[•*-]\s*", "", line)
    cleaned = re.sub(r"[*_`]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_contact_or_artifact_line(line: str) -> bool:
    cleaned = _clean_language_block(line)
    if _EMAIL_RE.search(cleaned) or _PHONE_RE.search(cleaned):
        return True
    if "|" in cleaned and len(_words(cleaned)) < 12:
        return True
    return False


def _looks_like_education_prose(line: str) -> bool:
    low = _clean_language_block(line).casefold()
    prose_markers = (
        "completed",
        "cours complétés",
        "cours completes",
        "including",
        "incl.",
        "incluant",
        "curriculum",
        "programme",
        "network components",
        "composants réseau",
        "composants reseau",
        "certification exam",
        "examen de certification",
    )
    return any(marker in low for marker in prose_markers) or low.endswith(".")


def _substantial_resume_blocks(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_section = "header"
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        section = _resume_language_section(raw_line)
        if section is not None:
            sections.append((current_section, current_lines))
            current_section = section
            current_lines = []
            continue
        current_lines.append(raw_line)
    sections.append((current_section, current_lines))

    blocks: list[tuple[str, str]] = []
    for section, lines in sections:
        if section == "profile":
            block = _clean_language_block(" ".join(line for line in lines if line.strip()))
            if len(_words(block)) >= 8 and not _is_contact_or_artifact_line(block):
                blocks.append(("resume:profile", block))
            continue
        if section in {"experience", "projects"}:
            for line in lines:
                block = _clean_language_block(line)
                if len(_words(block)) >= 8 and not _is_contact_or_artifact_line(block):
                    blocks.append((f"resume:{section}", block))
            continue
        if section == "education":
            for line in lines:
                block = _clean_language_block(line)
                if (
                    len(_words(block)) >= 8
                    and _looks_like_education_prose(block)
                    and not _is_contact_or_artifact_line(block)
                ):
                    blocks.append(("resume:education", block))
    return blocks


def _substantial_cover_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for paragraph in _paragraphs(text):
        cleaned = _clean_language_block(paragraph)
        low = cleaned.casefold()
        if low.startswith(("dear ", "bonjour ", *(_CLOSINGS))):
            continue
        if len(_words(cleaned)) >= 8:
            blocks.append(("cover:body", cleaned))
    return blocks


def _assess_declared_language_quality(
    *,
    expected_language: str | None,
    resume_text: str,
    cover_text: str,
) -> LanguageQualityReport | None:
    expected = _language_code_from_label(expected_language)
    if expected is None:
        return None

    blocks = [
        *_substantial_resume_blocks(resume_text),
        *_substantial_cover_blocks(cover_text),
    ]
    mismatched = [
        label
        for label, block in blocks
        if detect_language(block) != expected or _has_foreign_connector_leakage(block, expected)
    ]
    return LanguageQualityReport(
        expected=expected,
        checked_blocks=len(blocks),
        mismatched_blocks=len(mismatched),
        mismatched_sections=_ordered_unique(mismatched),
    )


def _has_foreign_connector_leakage(block: str, expected: str) -> bool:
    if expected != "fr":
        return False
    return bool(
        re.search(
            r"\b(and|including|completed|full|network components|routing and switching|in "
            r"fedora)\b",
            block,
            flags=re.IGNORECASE,
        )
    )


def _duplicate_resume_bullets(text: str) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("•", "-", "*")):
            continue
        normalized = _clean_language_block(stripped).casefold().strip(" .;:")
        if len(_words(normalized)) < 6:
            continue
        if normalized in seen and normalized not in duplicates:
            duplicates.append(normalized)
        seen.add(normalized)
    return duplicates


def _score_usefulness(
    *,
    resume_report: QualityReport,
    cover_report: QualityReport,
    packet_text: str,
    keywords: list[str],
) -> float:
    structure = (resume_report.score + cover_report.score) / 200
    coverage = _keyword_coverage(packet_text, keywords) if keywords else 0.0
    return _clamp_dimension(4.0 * ((0.55 * structure) + (0.45 * coverage)))


def _score_specificity(
    *,
    resume_text: str,
    cover_text: str,
    keywords: list[str],
    job_title: str,
    company: str,
) -> float:
    packet_text = f"{resume_text}\n{cover_text}"
    packet_coverage = _keyword_coverage(packet_text, keywords) if keywords else 0.0
    cover_coverage = _keyword_coverage(cover_text, keywords) if keywords else 0.0
    target_score = _target_mention_score(cover_text, job_title=job_title, company=company)
    generic_hits = sum(1 for phrase in _GENERIC_COVER_PHRASES if phrase in cover_text.lower())
    generic_penalty = min(1.0, generic_hits * 0.25)
    raw = 4.0 * ((0.6 * packet_coverage) + (0.25 * cover_coverage) + (0.15 * target_score))
    return _clamp_dimension(raw - generic_penalty)


def _score_coherence(
    *,
    resume_text: str,
    cover_text: str,
    applicant_name: str,
    job_title: str,
    company: str,
    coherence_terms: list[str],
) -> tuple[float, bool]:
    identity_score = _name_mention_score(
        resume_text=resume_text,
        cover_text=cover_text,
        applicant_name=applicant_name,
    )
    target_score = _target_mention_score(cover_text, job_title=job_title, company=company)
    source_backed_terms = [term for term in coherence_terms if _phrase_present(resume_text, term)]
    if source_backed_terms:
        shared_terms = [term for term in source_backed_terms if _phrase_present(cover_text, term)]
        shared_term_score = len(shared_terms) / len(source_backed_terms)
    else:
        shared_term_score = 0.0 if coherence_terms else 1.0
    language_score = _language_coherence_score(resume_text, cover_text)

    raw = 4.0 * (
        (0.20 * identity_score)
        + (0.25 * target_score)
        + (0.45 * shared_term_score)
        + (0.10 * language_score)
    )
    return _clamp_dimension(raw), bool(coherence_terms and not source_backed_terms)


def _score_writing_quality(*, cover_text: str, cover_report: QualityReport) -> float:
    score = 4.0 - (0.75 * len(cover_report.failures)) - (0.25 * len(cover_report.warnings))
    lengths = _sentence_lengths(cover_text)
    if lengths:
        average = sum(lengths) / len(lengths)
        if average < 8 or average > 34:
            score -= 0.4
    paragraphs = _paragraphs(cover_text)
    starts = [_words(paragraph)[0] for paragraph in paragraphs if _words(paragraph)]
    if len(starts) >= 4 and len(set(starts)) / len(starts) < 0.65:
        score -= 0.4
    return _clamp_dimension(score)


def _score_formatting_polish(
    *, resume_text: str, cover_text: str, resume_report: QualityReport, cover_report: QualityReport
) -> float:
    score = 4.0
    score -= 0.6 * len(resume_report.failures)
    score -= 0.8 * len(cover_report.failures)
    score -= 0.25 * (len(resume_report.warnings) + len(cover_report.warnings))
    long_lines = sum(1 for line in f"{resume_text}\n{cover_text}".splitlines() if len(line) > 140)
    if long_lines > 3:
        score -= 0.4
    return _clamp_dimension(score)


def assess_packet_case(
    case: dict[str, Any],
    *,
    base_dir: Path,
    min_dimension_score: float = _DEFAULT_DIMENSION_FLOOR,
    min_overall_score: float = _DEFAULT_OVERALL_FLOOR,
) -> PacketQualityReport:
    packet_id = str(_case_value(case, "id", "packet_id", "name") or "packet")
    resume_path = _resolve_case_path(
        _case_value(case, "resume_path", "resume"),
        base_dir=base_dir,
        field="resume_path",
    )
    cover_path = _resolve_case_path(
        _case_value(case, "cover_letter_path", "cover_letter"),
        base_dir=base_dir,
        field="cover_letter_path",
    )
    resume_text = _read_existing_text(resume_path, field="resume_path")
    cover_text = _read_existing_text(cover_path, field="cover_letter_path")
    generated_at, generated_at_source = _case_generated_at(
        case,
        resume_path=resume_path,
        cover_path=cover_path,
    )
    applicant_name = str(_case_value(case, "applicant_name", "profile_name") or "")
    job_title = str(_case_value(case, "job_title", "title") or "")
    company = str(_case_value(case, "company", "employer") or "")
    keywords, _job_description = _case_keywords(case, base_dir=base_dir)
    coherence_terms = _case_coherence_terms(case, keywords=keywords)
    declared_language = _normalize_language_label(_case_value(case, "language"))
    case_min_dimension_score = float(
        _case_value(case, "min_dimension_score", "dimension_floor") or min_dimension_score
    )
    case_min_dimension_score = max(min_dimension_score, case_min_dimension_score)
    case_min_overall_score = float(
        _case_value(case, "min_overall_score", "overall_floor") or min_overall_score
    )
    case_min_overall_score = max(min_overall_score, case_min_overall_score)

    resume_report = assess_resume(resume_text, keywords=keywords)
    cover_report = assess_cover_letter(cover_text, applicant_name=applicant_name)
    packet_text = f"{resume_text}\n{cover_text}"
    coherence_score, coherence_without_source = _score_coherence(
        resume_text=resume_text,
        cover_text=cover_text,
        applicant_name=applicant_name,
        job_title=job_title,
        company=company,
        coherence_terms=coherence_terms,
    )
    dimensions = {
        "usefulness": _score_usefulness(
            resume_report=resume_report,
            cover_report=cover_report,
            packet_text=packet_text,
            keywords=keywords,
        ),
        "specificity": _score_specificity(
            resume_text=resume_text,
            cover_text=cover_text,
            keywords=keywords,
            job_title=job_title,
            company=company,
        ),
        "coherence": coherence_score,
        "writing_quality": _score_writing_quality(
            cover_text=cover_text,
            cover_report=cover_report,
        ),
        "formatting_polish": _score_formatting_polish(
            resume_text=resume_text,
            cover_text=cover_text,
            resume_report=resume_report,
            cover_report=cover_report,
        ),
    }
    rounded_dimensions = {name: round(score, 2) for name, score in dimensions.items()}
    overall = round(sum(rounded_dimensions.values()) / len(rounded_dimensions), 2)

    failures = [f"resume: {item}" for item in resume_report.failures]
    failures.extend(f"cover_letter: {item}" for item in cover_report.failures)
    warnings = [f"resume: {item}" for item in resume_report.warnings]
    warnings.extend(f"cover_letter: {item}" for item in cover_report.warnings)
    if coherence_without_source:
        warnings.append("coherence terms are not source-backed; shared-term evidence reduced")
    language_quality = _assess_declared_language_quality(
        expected_language=declared_language,
        resume_text=resume_text,
        cover_text=cover_text,
    )
    if language_quality is not None and language_quality.mismatched_sections:
        sections = ", ".join(language_quality.mismatched_sections)
        failures.append(
            f"declared {language_quality.expected} packet contains "
            f"{language_quality.mismatched_blocks} substantial prose block(s) outside the "
            f"declared language: {sections}"
        )
    duplicate_bullets = _duplicate_resume_bullets(resume_text)
    if duplicate_bullets:
        failures.append(f"duplicate resume bullet(s): {len(duplicate_bullets)}")
    if not keywords:
        failures.append("packet has no keywords or job_description for usefulness/specificity")
    for name in _DIMENSIONS:
        score = rounded_dimensions[name]
        if score < case_min_dimension_score:
            failures.append(
                f"{name} score {score:.2f} is below required {case_min_dimension_score:.2f}"
            )
    if overall < case_min_overall_score:
        failures.append(
            f"overall score {overall:.2f} is below required {case_min_overall_score:.2f}"
        )

    return PacketQualityReport(
        packet_id=packet_id,
        passed=not failures,
        overall=overall,
        dimensions=rounded_dimensions,
        failures=failures,
        warnings=warnings,
        resume=resume_report,
        cover_letter=cover_report,
        category=_normalize_label(_case_value(case, "category", "job_category")),
        language=declared_language,
        generated_at=generated_at,
        generated_at_source=generated_at_source,
        provenance=_case_provenance(case),
        language_quality=language_quality,
    )


def _default_packet_set_path() -> Path:
    return Path(os.environ.get(_DOCUMENT_QUALITY_SET_ENV, _DEFAULT_PACKET_SET)).expanduser()


def assess_packet_set(
    packet_set: Path,
    *,
    min_dimension_score: float = _DEFAULT_DIMENSION_FLOOR,
    min_overall_score: float = _DEFAULT_OVERALL_FLOOR,
) -> list[PacketQualityReport]:
    records = _load_packet_records(packet_set)
    return [
        assess_packet_case(
            case,
            base_dir=packet_set.parent,
            min_dimension_score=min_dimension_score,
            min_overall_score=min_overall_score,
        )
        for case in records
    ]


def _effective_min_cases(*, required: bool, min_cases: int | None) -> int:
    if min_cases is not None:
        return min_cases
    return _REQUIRED_MIN_CASES if required else _OPTIONAL_MIN_CASES


def _packet_set_thresholds(
    *,
    min_dimension_score: float,
    min_overall_score: float,
    min_cases: int,
    max_artifact_age_days: int,
) -> PacketSetThresholds:
    return PacketSetThresholds(
        min_dimension_score=min_dimension_score,
        min_overall_score=min_overall_score,
        min_cases=min_cases,
        max_artifact_age_days=max_artifact_age_days,
    )


def _empty_coverage(
    *,
    required_categories: list[str],
    required_languages: list[str],
) -> PacketSetCoverage:
    return PacketSetCoverage(
        required_categories=required_categories,
        present_categories=[],
        missing_categories=required_categories,
        required_languages=required_languages,
        present_languages=[],
        missing_languages=required_languages,
    )


def _missing_certification(
    *,
    required: bool,
    thresholds: PacketSetThresholds,
    required_categories: list[str],
    required_languages: list[str],
    message: str,
) -> PacketSetCertification:
    return PacketSetCertification(
        certified=False,
        mode="packet_set",
        required=required,
        thresholds=thresholds,
        coverage=_empty_coverage(
            required_categories=required_categories,
            required_languages=required_languages,
        ),
        freshness=PacketSetFreshness(
            oldest_artifact_age_days=None,
            newest_artifact_age_days=None,
            stale_packet_ids=[],
        ),
        certification_failures=[message],
        certification_warnings=[],
    )


def _packet_age_days(report: PacketQualityReport, *, now: datetime) -> float:
    if report.generated_at is None:
        return 0.0
    generated_at = _parse_generated_at(
        report.generated_at,
        field=f"{report.packet_id}.generated_at",
    )
    return (_as_utc(now) - generated_at).total_seconds() / 86400


def _sorted_unique(items: list[str | None]) -> list[str]:
    return sorted({item for item in items if item})


def certify_packet_set(
    reports: list[PacketQualityReport],
    *,
    required: bool = False,
    min_dimension_score: float = _DEFAULT_DIMENSION_FLOOR,
    min_overall_score: float = _DEFAULT_OVERALL_FLOOR,
    min_cases: int | None = None,
    max_artifact_age_days: int = _DEFAULT_MAX_ARTIFACT_AGE_DAYS,
    required_categories: list[str] | None = None,
    required_languages: list[str] | None = None,
    now: datetime | None = None,
) -> PacketSetCertification:
    """Summarize whether a packet set is broad and fresh enough to certify."""
    effective_min_cases = _effective_min_cases(required=required, min_cases=min_cases)
    thresholds = _packet_set_thresholds(
        min_dimension_score=min_dimension_score,
        min_overall_score=min_overall_score,
        min_cases=effective_min_cases,
        max_artifact_age_days=max_artifact_age_days,
    )
    required_category_labels = [
        _normalize_label(value) or "" for value in required_categories or []
    ]
    required_language_labels = [
        _normalize_language_label(value) or "" for value in required_languages or []
    ]
    required_category_labels = [value for value in required_category_labels if value]
    required_language_labels = [value for value in required_language_labels if value]

    passing = [report for report in reports if report.passed]
    present_categories = _sorted_unique([report.category for report in passing])
    present_languages = _sorted_unique([report.language for report in passing])
    missing_categories = [
        label for label in required_category_labels if label not in present_categories
    ]
    missing_languages = [
        label for label in required_language_labels if label not in present_languages
    ]
    coverage = PacketSetCoverage(
        required_categories=required_category_labels,
        present_categories=present_categories,
        missing_categories=missing_categories,
        required_languages=required_language_labels,
        present_languages=present_languages,
        missing_languages=missing_languages,
    )

    checked_at = now or datetime.now(UTC)
    ages = [(report.packet_id, _packet_age_days(report, now=checked_at)) for report in reports]
    future_packet_ids = [packet_id for packet_id, age_days in ages if age_days < -0.25]
    stale_packet_ids = [
        packet_id for packet_id, age_days in ages if age_days > max_artifact_age_days
    ]
    age_values = [age_days for _packet_id, age_days in ages]
    freshness = PacketSetFreshness(
        oldest_artifact_age_days=round(max(age_values), 2) if age_values else None,
        newest_artifact_age_days=round(min(age_values), 2) if age_values else None,
        stale_packet_ids=stale_packet_ids,
    )

    failures: list[str] = []
    warnings: list[str] = []
    failed_packets = [report.packet_id for report in reports if not report.passed]
    if failed_packets:
        failures.append(f"packet case failures: {', '.join(failed_packets)}")
    if len(passing) < effective_min_cases:
        failures.append(
            f"passing packet count {len(passing)} is below required {effective_min_cases}"
        )
    if stale_packet_ids:
        failures.append(
            "stale packets exceed max artifact age "
            f"{max_artifact_age_days} days: {', '.join(stale_packet_ids)}"
        )
    if future_packet_ids:
        failures.append(f"packet generated_at is in the future: {', '.join(future_packet_ids)}")
    if missing_categories:
        failures.append(f"missing required categories: {', '.join(missing_categories)}")
    if missing_languages:
        failures.append(f"missing required languages: {', '.join(missing_languages)}")
    if reports and not present_categories:
        warnings.append("no passing packets declare category metadata")
    if reports and not present_languages:
        warnings.append("no passing packets declare language metadata")

    return PacketSetCertification(
        certified=not failures,
        mode="packet_set",
        required=required,
        thresholds=thresholds,
        coverage=coverage,
        freshness=freshness,
        certification_failures=failures,
        certification_warnings=warnings,
    )


def _packet_payload(
    packet_set: Path,
    reports: list[PacketQualityReport],
    certification: PacketSetCertification,
) -> dict[str, Any]:
    failed = [report for report in reports if not report.passed]
    overall = round(sum(report.overall for report in reports) / len(reports), 2) if reports else 0.0
    return {
        "packet_set": str(packet_set),
        "passed": not failed,
        "certified": certification.certified,
        "mode": certification.mode,
        "required": certification.required,
        "thresholds": asdict(certification.thresholds),
        "coverage": asdict(certification.coverage),
        "freshness": asdict(certification.freshness),
        "certification_failures": certification.certification_failures,
        "certification_warnings": certification.certification_warnings,
        "count": len(reports),
        "overall": overall,
        "dimension_means": {
            name: round(sum(report.dimensions[name] for report in reports) / len(reports), 2)
            if reports
            else 0.0
            for name in _DIMENSIONS
        },
        "packets": [asdict(report) for report in reports],
    }


def _unavailable_payload(
    *,
    packet_set: Path,
    reason: str,
    message: str,
    required: bool,
    min_dimension_score: float,
    min_overall_score: float,
    min_cases: int,
    max_artifact_age_days: int,
    required_categories: list[str] | None,
    required_languages: list[str] | None,
) -> dict[str, Any]:
    normalized_categories = [
        value
        for value in (_normalize_label(item) for item in required_categories or [])
        if value is not None
    ]
    normalized_languages = [
        value
        for value in (_normalize_label(item) for item in required_languages or [])
        if value is not None
    ]
    thresholds = _packet_set_thresholds(
        min_dimension_score=min_dimension_score,
        min_overall_score=min_overall_score,
        min_cases=min_cases,
        max_artifact_age_days=max_artifact_age_days,
    )
    certification = _missing_certification(
        required=required,
        thresholds=thresholds,
        required_categories=normalized_categories,
        required_languages=normalized_languages,
        message=message,
    )
    return {
        "packet_set": str(packet_set),
        "passed": False,
        "certified": False,
        "mode": "packet_set",
        "required": required,
        "reason": reason,
        "message": message,
        "thresholds": asdict(certification.thresholds),
        "coverage": asdict(certification.coverage),
        "freshness": asdict(certification.freshness),
        "certification_failures": certification.certification_failures,
        "certification_warnings": certification.certification_warnings,
        "count": 0,
        "overall": 0.0,
        "dimension_means": {name: 0.0 for name in _DIMENSIONS},
        "packets": [],
    }


def _emit_unavailable(
    *,
    packet_set: Path,
    reason: str,
    message: str,
    required: bool,
    json_output: bool,
    min_dimension_score: float,
    min_overall_score: float,
    min_cases: int,
    max_artifact_age_days: int,
    required_categories: list[str] | None,
    required_languages: list[str] | None,
    exit_code: int,
) -> int:
    if json_output:
        payload = _unavailable_payload(
            packet_set=packet_set,
            reason=reason,
            message=message,
            required=required,
            min_dimension_score=min_dimension_score,
            min_overall_score=min_overall_score,
            min_cases=min_cases,
            max_artifact_age_days=max_artifact_age_days,
            required_categories=required_categories,
            required_languages=required_languages,
        )
        print(json.dumps(payload, indent=2))
    else:
        print(message)
    return exit_code


def _format_values(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def _render_packet_set_payload(payload: dict[str, Any], reports: list[PacketQualityReport]) -> None:
    certification_verdict = "CERTIFIED" if payload["certified"] else "NOT CERTIFIED"
    passed_count = sum(1 for report in reports if report.passed)
    thresholds = payload["thresholds"]
    freshness = payload["freshness"]
    coverage = payload["coverage"]
    print(f"Document packet certification: {certification_verdict}")
    print(
        f"  packets: {passed_count}/{payload['count']} passing; required={thresholds['min_cases']}"
    )
    print(
        "  freshness: "
        f"newest={freshness['newest_artifact_age_days']}d "
        f"oldest={freshness['oldest_artifact_age_days']}d "
        f"max={thresholds['max_artifact_age_days']}d "
        f"stale={_format_values(freshness['stale_packet_ids'])}"
    )
    print(
        "  coverage: "
        f"categories present={_format_values(coverage['present_categories'])} "
        f"missing={_format_values(coverage['missing_categories'])}; "
        f"languages present={_format_values(coverage['present_languages'])} "
        f"missing={_format_values(coverage['missing_languages'])}"
    )
    for item in payload["certification_failures"]:
        print(f"  certification failure: {item}")
    for item in payload["certification_warnings"]:
        print(f"  certification warning: {item}")

    verdict = "PASS" if payload["passed"] else "FAIL"
    print(
        f"{verdict} document packet quality: packets={payload['count']} "
        f"overall={payload['overall']:.2f}/4"
    )
    print("dimension means:")
    for name in _DIMENSIONS:
        print(f"  {name}: {payload['dimension_means'][name]:.2f}/4")
    for report in reports:
        row_verdict = "PASS" if report.passed else "FAIL"
        dimensions = ", ".join(f"{name}={report.dimensions[name]:.2f}" for name in _DIMENSIONS)
        print(f"{row_verdict} {report.packet_id}: overall={report.overall:.2f} ({dimensions})")
        for item in report.failures:
            print(f"  failure: {item}")
        for item in report.warnings:
            print(f"  warning: {item}")


def _run_packet_set(
    *,
    packet_set: Path | None = None,
    required: bool = False,
    json_output: bool = False,
    min_dimension_score: float = _DEFAULT_DIMENSION_FLOOR,
    min_overall_score: float = _DEFAULT_OVERALL_FLOOR,
    min_cases: int | None = None,
    max_artifact_age_days: int = _DEFAULT_MAX_ARTIFACT_AGE_DAYS,
    required_categories: list[str] | None = None,
    required_languages: list[str] | None = None,
) -> int:
    resolved = (packet_set or _default_packet_set_path()).expanduser()
    effective_min_cases = _effective_min_cases(required=required, min_cases=min_cases)
    if not resolved.exists():
        return _emit_unavailable(
            packet_set=resolved,
            reason="missing_packet_set",
            message=(
                f"no document quality packet set at {resolved} - generated packets are not "
                "certified"
            ),
            required=required,
            json_output=json_output,
            min_dimension_score=min_dimension_score,
            min_overall_score=min_overall_score,
            min_cases=effective_min_cases,
            max_artifact_age_days=max_artifact_age_days,
            required_categories=required_categories,
            required_languages=required_languages,
            exit_code=2 if required else 0,
        )
    if not resolved.is_file():
        return _emit_unavailable(
            packet_set=resolved,
            reason="invalid_packet_set",
            message=f"document quality packet set is not a file: {resolved}",
            required=required,
            json_output=json_output,
            min_dimension_score=min_dimension_score,
            min_overall_score=min_overall_score,
            min_cases=effective_min_cases,
            max_artifact_age_days=max_artifact_age_days,
            required_categories=required_categories,
            required_languages=required_languages,
            exit_code=2,
        )
    try:
        reports = assess_packet_set(
            resolved,
            min_dimension_score=min_dimension_score,
            min_overall_score=min_overall_score,
        )
    except (OSError, ValueError) as exc:
        return _emit_unavailable(
            packet_set=resolved,
            reason="invalid_packet_set",
            message=f"document quality packet set is invalid: {exc}",
            required=required,
            json_output=json_output,
            min_dimension_score=min_dimension_score,
            min_overall_score=min_overall_score,
            min_cases=effective_min_cases,
            max_artifact_age_days=max_artifact_age_days,
            required_categories=required_categories,
            required_languages=required_languages,
            exit_code=2,
        )
    if not reports:
        return _emit_unavailable(
            packet_set=resolved,
            reason="empty_packet_set",
            message=(
                f"document quality packet set has no cases at {resolved} - generated packets are "
                "not certified"
            ),
            required=required,
            json_output=json_output,
            min_dimension_score=min_dimension_score,
            min_overall_score=min_overall_score,
            min_cases=effective_min_cases,
            max_artifact_age_days=max_artifact_age_days,
            required_categories=required_categories,
            required_languages=required_languages,
            exit_code=2 if required else 0,
        )

    certification = certify_packet_set(
        reports,
        required=required,
        min_dimension_score=min_dimension_score,
        min_overall_score=min_overall_score,
        min_cases=effective_min_cases,
        max_artifact_age_days=max_artifact_age_days,
        required_categories=required_categories,
        required_languages=required_languages,
    )
    payload = _packet_payload(resolved, reports, certification)
    if json_output:
        print(json.dumps(payload, indent=2))
    else:
        _render_packet_set_payload(payload, reports)

    if not payload["passed"]:
        return 1
    if required and not payload["certified"]:
        return 1
    return 0


def run_packet_set_quality(
    *,
    packet_set: Path | None = None,
    required: bool = False,
    json_output: bool = False,
    min_dimension_score: float = _DEFAULT_DIMENSION_FLOOR,
    min_overall_score: float = _DEFAULT_OVERALL_FLOOR,
    min_cases: int | None = None,
    max_artifact_age_days: int = _DEFAULT_MAX_ARTIFACT_AGE_DAYS,
    required_categories: list[str] | None = None,
    required_languages: list[str] | None = None,
) -> int:
    """Run the private packet-set quality gate and return its process-style exit code."""
    return _run_packet_set(
        packet_set=packet_set,
        required=required,
        json_output=json_output,
        min_dimension_score=min_dimension_score,
        min_overall_score=min_overall_score,
        min_cases=min_cases,
        max_artifact_age_days=max_artifact_age_days,
        required_categories=required_categories,
        required_languages=required_languages,
    )


def assess_cover_letter(text: str, *, applicant_name: str = "") -> QualityReport:
    failures: list[str] = []
    warnings: list[str] = []
    paragraphs = _paragraphs(text)
    words = _words(text)

    if len(words) < 100:
        failures.append("cover letter is too short to be useful")
    if len(words) > 650:
        warnings.append("cover letter is unusually long")
    body_paragraphs = [
        p for p in paragraphs if not p.lower().startswith(("dear ", "bonjour ", *(_CLOSINGS)))
    ]
    if len(body_paragraphs) not in (3, 4):
        warnings.append("cover letter should usually have three focused body paragraphs")
    if re.search(r"^\s*[-*#]", text, re.M):
        failures.append("cover letter contains markdown/list formatting")
    if _PLACEHOLDER_RE.search(text):
        failures.append("cover letter contains placeholder text")

    low = text.lower()
    if not any(closing in low for closing in _CLOSINGS):
        failures.append("cover letter is missing a recognized sign-off")
    if applicant_name and applicant_name.lower() not in low:
        failures.append("cover letter sign-off does not include the applicant name")

    starts = [_words(p)[0] for p in body_paragraphs if _words(p)]
    if len(starts) >= 3 and len(set(starts)) == 1:
        warnings.append("cover letter paragraphs start repetitively")

    score = max(0, 100 - 30 * len(failures) - 10 * len(warnings))
    return QualityReport("cover_letter", not failures, score, failures, warnings)


def assess_resume(text: str, *, keywords: list[str]) -> QualityReport:
    failures: list[str] = []
    warnings: list[str] = []
    words = _words(text)
    low = text.lower()

    if len(words) < 100:
        failures.append("resume is too short to be a complete CV")
    if not _EMAIL_RE.search(text):
        failures.append("resume is missing an email address")
    if not _PHONE_RE.search(text):
        failures.append("resume is missing a phone number")
    if _PLACEHOLDER_RE.search(text):
        failures.append("resume contains placeholder text")

    for section, aliases in _SECTION_ALIASES.items():
        if not any(alias in low for alias in aliases):
            failures.append(f"resume is missing a {section} section")

    coverage = _keyword_coverage(text, keywords)
    if keywords and coverage < 0.5:
        failures.append(f"resume covers only {coverage:.0%} of required job keywords")
    elif keywords and coverage < 0.75:
        warnings.append(f"resume covers {coverage:.0%} of required job keywords")

    score = max(0, 100 - 25 * len(failures) - 10 * len(warnings))
    return QualityReport("resume", not failures, score, failures, warnings)


def assess_artifacts(
    *,
    resume_path: Path | None = None,
    cover_letter_path: Path | None = None,
    applicant_name: str = "",
    keywords: list[str] | None = None,
) -> list[QualityReport]:
    """Assess one generated CV, one generated cover letter, or both."""
    reports: list[QualityReport] = []
    if cover_letter_path is not None:
        reports.append(
            assess_cover_letter(
                cover_letter_path.read_text(encoding="utf-8"),
                applicant_name=applicant_name,
            )
        )
    if resume_path is not None:
        reports.append(
            assess_resume(
                resume_path.read_text(encoding="utf-8"),
                keywords=list(keywords or []),
            )
        )
    if not reports:
        raise ValueError("provide --cover-letter and/or --resume")
    return reports


def artifact_payload(reports: list[QualityReport]) -> list[dict[str, Any]]:
    return [asdict(report) for report in reports]


def render_artifact_reports(reports: list[QualityReport]) -> None:
    for report in reports:
        verdict = "PASS" if report.passed else "FAIL"
        print(f"{verdict} {report.kind}: score={report.score}")
        for item in report.failures:
            print(f"  failure: {item}")
        for item in report.warnings:
            print(f"  warning: {item}")


def run_artifact_quality(
    *,
    resume_path: Path | None = None,
    cover_letter_path: Path | None = None,
    applicant_name: str = "",
    keywords: list[str] | None = None,
    json_output: bool = False,
) -> int:
    """Run the single-artifact smoke gate and return its process-style exit code."""
    reports = assess_artifacts(
        resume_path=resume_path,
        cover_letter_path=cover_letter_path,
        applicant_name=applicant_name,
        keywords=keywords,
    )
    if json_output:
        print(json.dumps(artifact_payload(reports), indent=2))
    else:
        render_artifact_reports(reports)
    return 0 if all(report.passed for report in reports) else 1


def validate_score_floor(option_name: str, value: float) -> None:
    if value < 0 or value > 4:
        raise ValueError(f"--{option_name} must be between 0 and 4")


def validate_positive_int(option_name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"--{option_name} must be at least 1")


def validate_nonnegative_int(option_name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"--{option_name} must be at least 0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cover-letter", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--applicant-name", default="")
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Required job keyword to check in the résumé. Repeatable.",
    )
    parser.add_argument(
        "--packet-set",
        nargs="?",
        const=_default_packet_set_path(),
        type=Path,
        help=(
            "Score a private generated packet set. If no path is supplied, uses "
            f"${_DOCUMENT_QUALITY_SET_ENV} or {_DEFAULT_PACKET_SET}."
        ),
    )
    parser.add_argument(
        "--required",
        action="store_true",
        help="Exit non-zero when private packet-set evidence is unavailable or empty.",
    )
    parser.add_argument(
        "--min-dimension-score",
        type=float,
        default=_DEFAULT_DIMENSION_FLOOR,
        help="Minimum required 0-4 score for each packet quality dimension.",
    )
    parser.add_argument(
        "--min-overall-score",
        type=float,
        default=_DEFAULT_OVERALL_FLOOR,
        help="Minimum required 0-4 overall score for each packet.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=None,
        help="Minimum passing packet cases required for set certification.",
    )
    parser.add_argument(
        "--max-artifact-age-days",
        type=int,
        default=None,
        help="Maximum generated packet age in days for set certification. Defaults to 14.",
    )
    parser.add_argument(
        "--required-category",
        action="append",
        default=[],
        help="Required packet category for set certification. Repeatable.",
    )
    parser.add_argument(
        "--required-language",
        action="append",
        default=[],
        help="Required packet language for set certification. Repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    for floor_name in ("min_dimension_score", "min_overall_score"):
        floor = getattr(args, floor_name)
        try:
            validate_score_floor(floor_name.replace("_", "-"), floor)
        except ValueError as exc:
            parser.error(str(exc))
    if args.min_cases is not None:
        try:
            validate_positive_int("min-cases", args.min_cases)
        except ValueError as exc:
            parser.error(str(exc))
    if args.max_artifact_age_days is not None:
        try:
            validate_nonnegative_int("max-artifact-age-days", args.max_artifact_age_days)
        except ValueError as exc:
            parser.error(str(exc))

    if args.packet_set is not None:
        if args.resume or args.cover_letter:
            parser.error("--packet-set cannot be combined with --resume or --cover-letter")
        return _run_packet_set(
            packet_set=args.packet_set,
            required=args.required,
            json_output=args.json,
            min_dimension_score=args.min_dimension_score,
            min_overall_score=args.min_overall_score,
            min_cases=args.min_cases,
            max_artifact_age_days=(
                args.max_artifact_age_days
                if args.max_artifact_age_days is not None
                else _DEFAULT_MAX_ARTIFACT_AGE_DAYS
            ),
            required_categories=list(args.required_category),
            required_languages=list(args.required_language),
        )
    if args.required:
        parser.error("--required is only valid with --packet-set")
    if (
        args.min_cases is not None
        or args.max_artifact_age_days is not None
        or args.required_category
        or args.required_language
    ):
        parser.error(
            "--min-cases, --max-artifact-age-days, --required-category, and "
            "--required-language are only valid with --packet-set"
        )

    try:
        return run_artifact_quality(
            resume_path=args.resume,
            cover_letter_path=args.cover_letter,
            applicant_name=args.applicant_name,
            keywords=list(args.keyword),
            json_output=args.json,
        )
    except ValueError:
        parser.error("provide --cover-letter and/or --resume")
    return 2


if __name__ == "__main__":
    sys.exit(main())
