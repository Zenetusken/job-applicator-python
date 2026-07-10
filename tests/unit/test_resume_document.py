"""Tests for canonical source-preserving résumé documents."""

from __future__ import annotations

import pytest

from job_applicator.documents.resume_document import (
    ResumeDocument,
    canonical_resume_text,
    protected_span_recall,
)
from job_applicator.exceptions import TailorIntegrityError

SOURCE = """ALEX MORGAN\r
alex@example.com\r
\r
SUMMARY\r
Original summary.\r
\r
EXPERIENCE\r
Analyst  2020 - 2024\r
Acme\r
• Preserved a critical qualifier.\r
\r
EDUCATION\r
Coursework  2019 - 2020\r
College\r
(exam pending)\r
"""


def test_document_round_trips_canonical_source() -> None:
    document = ResumeDocument.parse(SOURCE)

    assert document.render() == canonical_resume_text(SOURCE)


def test_summary_replacement_preserves_non_summary_digest() -> None:
    source = ResumeDocument.parse(SOURCE)
    tailored = source.with_summary("Targeted source-backed summary.", language="English")

    assert tailored.non_summary_sha256() == source.non_summary_sha256()
    assert "Original summary" not in tailored.render()
    assert "Targeted source-backed summary." in tailored.render()
    assert "(exam pending)" in tailored.render()


def test_summary_is_inserted_when_source_has_none() -> None:
    source = ResumeDocument.parse("ALEX\n\nEXPERIENCE\nAcme role")

    tailored = source.with_summary("Grounded summary.", language="English")

    assert tailored.sections[0].kind == "Summary"
    assert tailored.non_summary_sha256() == source.non_summary_sha256()


def test_multiple_summary_sections_fail_closed() -> None:
    source = ResumeDocument.parse("ALEX\n\nSUMMARY\nOne\n\nPROFILE\nTwo\n\nSKILLS\nPython")

    with pytest.raises(TailorIntegrityError, match="multiple summary"):
        source.with_summary("Replacement.", language="English")


def test_unstructured_source_fails_closed() -> None:
    with pytest.raises(TailorIntegrityError, match="recognizable résumé sections"):
        ResumeDocument.parse("ALEX MORGAN\nNo headings here")


def test_protected_span_recall_reports_missing_source_evidence() -> None:
    retained, missing = protected_span_recall(
        SOURCE,
        ["critical qualifier", "exam pending", "UpClick"],
    )

    assert retained == 2
    assert missing == ["UpClick"]
