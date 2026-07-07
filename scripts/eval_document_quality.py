#!/usr/bin/env python
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
from pathlib import Path
from typing import Any

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
_DIMENSIONS = ("usefulness", "specificity", "writing_quality", "formatting_polish")
_DEFAULT_DIMENSION_FLOOR = 3.0
_DEFAULT_OVERALL_FLOOR = 3.0
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


@dataclass(frozen=True)
class QualityReport:
    kind: str
    passed: bool
    score: int
    failures: list[str]
    warnings: list[str]


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


def _target_mention_score(text: str, *, job_title: str, company: str) -> float:
    terms = [term for term in (job_title, company) if term]
    if not terms:
        return 1.0
    low = text.lower()
    hits = sum(1 for term in terms if term.lower() in low)
    return hits / len(terms)


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
    applicant_name = str(_case_value(case, "applicant_name", "profile_name") or "")
    job_title = str(_case_value(case, "job_title", "title") or "")
    company = str(_case_value(case, "company", "employer") or "")
    keywords, _job_description = _case_keywords(case, base_dir=base_dir)
    case_min_dimension_score = float(
        _case_value(case, "min_dimension_score", "dimension_floor") or min_dimension_score
    )
    case_min_overall_score = float(
        _case_value(case, "min_overall_score", "overall_floor") or min_overall_score
    )

    resume_report = assess_resume(resume_text, keywords=keywords)
    cover_report = assess_cover_letter(cover_text, applicant_name=applicant_name)
    packet_text = f"{resume_text}\n{cover_text}"
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
    )


def _default_packet_set_path() -> Path:
    return Path(os.environ.get(_DOCUMENT_QUALITY_SET_ENV, _DEFAULT_PACKET_SET)).expanduser()


def _not_certified(message: str, *, required: bool) -> int:
    print(message)
    return 2 if required else 0


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


def _packet_payload(packet_set: Path, reports: list[PacketQualityReport]) -> dict[str, Any]:
    failed = [report for report in reports if not report.passed]
    overall = round(sum(report.overall for report in reports) / len(reports), 2) if reports else 0.0
    return {
        "packet_set": str(packet_set),
        "passed": not failed,
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


def _run_packet_set(
    *,
    packet_set: Path | None = None,
    required: bool = False,
    json_output: bool = False,
    min_dimension_score: float = _DEFAULT_DIMENSION_FLOOR,
    min_overall_score: float = _DEFAULT_OVERALL_FLOOR,
) -> int:
    resolved = (packet_set or _default_packet_set_path()).expanduser()
    if not resolved.exists():
        return _not_certified(
            f"no document quality packet set at {resolved} — generated packets are not certified",
            required=required,
        )
    if not resolved.is_file():
        print(f"document quality packet set is not a file: {resolved}", file=sys.stderr)
        return 2
    try:
        reports = assess_packet_set(
            resolved,
            min_dimension_score=min_dimension_score,
            min_overall_score=min_overall_score,
        )
    except (OSError, ValueError) as exc:
        print(f"document quality packet set is invalid: {exc}", file=sys.stderr)
        return 2
    if not reports:
        return _not_certified(
            f"document quality packet set has no cases at {resolved} — generated packets are not "
            "certified",
            required=required,
        )

    payload = _packet_payload(resolved, reports)
    if json_output:
        print(json.dumps(payload, indent=2))
    else:
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

    return 0 if payload["passed"] else 1


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
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    for floor_name in ("min_dimension_score", "min_overall_score"):
        floor = getattr(args, floor_name)
        if floor < 0 or floor > 4:
            parser.error(f"--{floor_name.replace('_', '-')} must be between 0 and 4")

    if args.packet_set is not None:
        if args.resume or args.cover_letter:
            parser.error("--packet-set cannot be combined with --resume or --cover-letter")
        return _run_packet_set(
            packet_set=args.packet_set,
            required=args.required,
            json_output=args.json,
            min_dimension_score=args.min_dimension_score,
            min_overall_score=args.min_overall_score,
        )
    if args.required:
        parser.error("--required is only valid with --packet-set")

    reports: list[QualityReport] = []
    if args.cover_letter:
        reports.append(
            assess_cover_letter(
                args.cover_letter.read_text(encoding="utf-8"),
                applicant_name=args.applicant_name,
            )
        )
    if args.resume:
        reports.append(
            assess_resume(args.resume.read_text(encoding="utf-8"), keywords=list(args.keyword))
        )
    if not reports:
        parser.error("provide --cover-letter and/or --resume")

    if args.json:
        print(json.dumps([asdict(report) for report in reports], indent=2))
    else:
        for report in reports:
            verdict = "PASS" if report.passed else "FAIL"
            print(f"{verdict} {report.kind}: score={report.score}")
            for item in report.failures:
                print(f"  failure: {item}")
            for item in report.warnings:
                print(f"  warning: {item}")

    return 0 if all(report.passed for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
