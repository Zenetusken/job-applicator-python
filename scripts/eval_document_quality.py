#!/usr/bin/env python
"""Heuristic quality gate for generated CV and cover-letter artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"[\+]?[\d\s\-().]{10,}")
_PLACEHOLDER_RE = re.compile(r"\b(lorem ipsum|todo|tbd|placeholder|your name|company name)\b", re.I)
_SECTION_ALIASES = {
    "experience": ("experience", "expérience"),
    "education": ("education", "éducation", "formation"),
    "skills": ("skills", "compétences", "competences"),
}
_CLOSINGS = ("sincerely", "best regards", "regards", "cordialement", "merci")


@dataclass(frozen=True)
class QualityReport:
    kind: str
    passed: bool
    score: int
    failures: list[str]
    warnings: list[str]


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
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

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
