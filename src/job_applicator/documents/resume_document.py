"""Canonical, source-preserving résumé section parsing and retention evidence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from job_applicator.documents.resume import section_header
from job_applicator.exceptions import TailorIntegrityError


def canonical_resume_text(text: str) -> str:
    """Normalize transport whitespace without changing source wording."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


@dataclass(frozen=True)
class ResumeDocumentSection:
    """One source section with its original heading and canonical body lines."""

    kind: str
    heading: str
    body_lines: tuple[str, ...]

    def render(self) -> str:
        return "\n".join((self.heading, *self.body_lines)).rstrip()


@dataclass(frozen=True)
class ResumeDocument:
    """A canonical résumé that round-trips every source section."""

    preamble_lines: tuple[str, ...]
    sections: tuple[ResumeDocumentSection, ...]

    @classmethod
    def parse(cls, text: str) -> ResumeDocument:
        canonical = canonical_resume_text(text)
        preamble: list[str] = []
        sections: list[ResumeDocumentSection] = []
        current_kind: str | None = None
        current_heading = ""
        current_body: list[str] = []

        def flush() -> None:
            nonlocal current_kind, current_heading, current_body
            if current_kind is not None:
                sections.append(
                    ResumeDocumentSection(
                        kind=current_kind,
                        heading=current_heading,
                        body_lines=tuple(current_body),
                    )
                )
            current_kind = None
            current_heading = ""
            current_body = []

        for line in canonical.split("\n"):
            kind = section_header(line)
            if kind is not None:
                flush()
                current_kind = kind
                current_heading = line
            elif current_kind is None:
                preamble.append(line)
            else:
                current_body.append(line)
        flush()
        if not sections:
            raise TailorIntegrityError(
                "Source-preserving tailoring requires recognizable résumé sections."
            )
        return cls(preamble_lines=tuple(preamble), sections=tuple(sections))

    def render(self) -> str:
        blocks = ["\n".join(self.preamble_lines).rstrip()]
        blocks.extend(section.render() for section in self.sections)
        return "\n\n".join(block for block in blocks if block).strip()

    def summary_sections(self) -> tuple[ResumeDocumentSection, ...]:
        return tuple(section for section in self.sections if section.kind == "Summary")

    def non_summary_text(self) -> str:
        blocks = ["\n".join(self.preamble_lines).rstrip()]
        blocks.extend(section.render() for section in self.sections if section.kind != "Summary")
        return "\n\n".join(block for block in blocks if block).strip()

    def non_summary_sha256(self) -> str:
        return hashlib.sha256(self.non_summary_text().encode("utf-8")).hexdigest()

    def with_summary(self, summary: str, *, language: str) -> ResumeDocument:
        """Replace the one summary body or insert a localized summary section."""

        summaries = self.summary_sections()
        if len(summaries) > 1:
            raise TailorIntegrityError(
                "Source-preserving tailoring found multiple summary sections."
            )
        summary_text = re.sub(r"\s+", " ", summary).strip()
        if not summary_text:
            raise TailorIntegrityError("Source-preserving tailoring produced an empty summary.")
        heading = "PROFIL" if language == "French" else "SUMMARY"
        replacement = ResumeDocumentSection(
            kind="Summary",
            heading=summaries[0].heading if summaries else heading,
            body_lines=(summary_text,),
        )
        if summaries:
            sections = tuple(
                replacement if section.kind == "Summary" else section for section in self.sections
            )
        else:
            sections = (replacement, *self.sections)
        candidate = ResumeDocument(self.preamble_lines, sections)
        if candidate.non_summary_sha256() != self.non_summary_sha256():
            raise TailorIntegrityError(
                "Source-preserving tailoring changed content outside the summary."
            )
        return candidate


def protected_span_recall(text: str, protected_spans: list[str]) -> tuple[int, list[str]]:
    """Return exact normalized protected-span retention evidence."""

    normalized = re.sub(r"\s+", " ", canonical_resume_text(text)).casefold()
    missing = [
        span
        for span in protected_spans
        if re.sub(r"\s+", " ", span).casefold().strip() not in normalized
    ]
    return len(protected_spans) - len(missing), missing
