"""Section-header robustness — the shared case-insensitive/qualifier-tolerant `section_header`
matcher and its two consumers (the summary boundary in `parse_text` + `ResumeDateValidator`
section attribution). Regression guard for the all-caps-qualified-header bug that made `summary`
swallow ~the whole document AND aborted `tailor` on a valid CV via a false ordering issue.
"""

from __future__ import annotations

import pytest

from job_applicator.documents.resume import ResumeLoader, section_header
from job_applicator.documents.resume_tailor import ResumeDateValidator


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("SUMMARY", "Summary"),
        ("TECHNICAL SKILLS", "Skills"),
        ("PROFESSIONAL EXPERIENCE", "Experience"),
        ("EDUCATION & CERTIFICATIONS", "Education"),
        ("**Professional Summary**", "Summary"),
        ("Work Experience", "Experience"),
        ("Employment History", "Experience"),
        ("Skills: Python, SQL", "Skills"),
        ("# Education", "Education"),
        # non-headers → None
        ("2018 - Present", None),
        ("CanadaLife, Montréal, QC", None),
        ("Answer inquiries, complaints and escalations via telephone", None),
        ("Communication & interpersonal skills", None),
    ],
)
def test_section_header_matches_qualified_all_caps(line: str, expected: str | None) -> None:
    assert section_header(line) == expected


# A real-shaped all-caps, qualified-header CV (the form the case-sensitive parser choked on).
_ALLCAPS_CV = (
    "ANDREI PETROV\njane@example.com · 514-555-0199\n"
    "SUMMARY\nSecurity-focused analyst with hands-on SOC lab experience and a strong ops record.\n"
    "TECHNICAL SKILLS\nSIEM, EDR, IDS/IPS\n"
    "PROFESSIONAL EXPERIENCE\nSOC Analyst\nAcme Corp\n2022 - Present\n"
    "EDUCATION & CERTIFICATIONS\nBSc Security\nUniversity\n2018 - 2021\n"
)


def test_summary_bounded_by_next_all_caps_header() -> None:
    """`summary` stops at the next section (all-caps, qualified) header, not at EOF — the bug made
    summary ~97% of the whole document on a real all-caps CV."""
    data = ResumeLoader().parse_text(_ALLCAPS_CV)
    assert "Security-focused analyst" in data.summary
    assert "SIEM" not in data.summary  # stopped BEFORE Technical Skills
    assert len(data.summary) < len(_ALLCAPS_CV) * 0.5  # bounded, not the whole doc


def test_date_validator_attributes_qualified_sections_no_false_ordering() -> None:
    """`ResumeDateValidator` buckets entries under all-caps qualified headers (not all 'Unknown'),
    so the within-section ordering check doesn't false-flag the education↔experience boundary —
    the flag that once aborted `tailor` on a valid CV."""
    audit = ResumeDateValidator().audit(ResumeLoader().parse_text(_ALLCAPS_CV))
    sections = {e.section for e in audit.entries}
    assert "Experience" in sections and "Education" in sections  # attributed, not 'Unknown'
    assert audit.ordering_issues == []  # no false cross-boundary inversion


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # #1 (review): all-caps 'AND' compound must split like '&' (was falling through → None)
        ("EDUCATION AND CERTIFICATIONS", "Education"),
        ("LICENSES AND CERTIFICATIONS", "Certifications"),
        ("Awards And Honors", "Awards"),
        # #2 (review): volunteer / internship are DISTINCT sections, not merged into Experience
        ("VOLUNTEER", "Volunteer"),
        ("Volunteer Experience", "Volunteer"),
        ("INTERNSHIPS", "Internship"),
    ],
)
def test_section_header_compound_and_distinct_sections(line: str, expected: str | None) -> None:
    assert section_header(line) == expected


def test_summary_captures_inline_prose() -> None:
    """#3 (review): an inline 'Summary: <prose>' header keeps its prose (was dropped — accumulation
    started at the next line). The short-summary stray-guard is fallback-only, so it stays."""
    data = ResumeLoader().parse_text(
        "Jane Doe\njane@x.com\nSummary: Security analyst with SOC lab experience.\nSkills\nSIEM"
    )
    assert "Security analyst with SOC lab experience." in data.summary
