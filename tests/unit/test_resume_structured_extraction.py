"""Fixtures-first contract for structured experience/education extraction (audit #14).

Authored BEFORE the extractor to prove it is NOT overfit to one résumé. The multi-format effort
concentrates on the DATE axis (where real résumé variability lives); the entry structure stays a
conservative title-first model that DEGRADES SAFELY (emits nothing → the raw_text fallback) on
anything it can't confidently parse — never a fabricated/garbage entry (no-failure-masking).

Two fixture classes:
  * must-parse         — common formats the extractor SHOULD handle (varied dates, en/fr, tabs)
  * must-degrade-safely — exotic/ambiguous input the extractor must reduce to [] (no garbage)
"""

from __future__ import annotations

import pytest

from job_applicator.documents.resume import (
    extract_education,
    extract_experience,
    parse_date_range,
)

# --------------------------------------------------------------------- date primitive

_DATE_CASES = [
    # (text, start_year, start_month, end_year, end_month, is_current)
    ("2020 – 2023", 2020, None, 2023, None, False),
    ("2020 - 2023", 2020, None, 2023, None, False),  # ascii hyphen
    ("2020 — 2023", 2020, None, 2023, None, False),  # em dash
    ("2020 to 2023", 2020, None, 2023, None, False),
    ("2020 – Present", 2020, None, None, None, True),
    ("2020 – Current", 2020, None, None, None, True),
    ("January 2020 – March 2023", 2020, 1, 2023, 3, False),
    ("Jan 2020 – Mar 2023", 2020, 1, 2023, 3, False),
    ("01/2020 – 03/2023", 2020, 1, 2023, 3, False),
    ("01/2020 – Present", 2020, 1, None, None, True),
    ("janvier 2020 – présent", 2020, 1, None, None, True),  # French month + présent
    ("mars 2019 – juin 2021", 2019, 3, 2021, 6, False),  # French months
    ("2020 – actuel", 2020, None, None, None, True),  # French "current"
    ("2019 à 2022", 2019, None, 2022, None, False),  # French "à" separator
]


@pytest.mark.parametrize("text,sy,sm,ey,em,cur", _DATE_CASES)
def test_parse_date_range_formats(
    text: str, sy: int, sm: int | None, ey: int | None, em: int | None, cur: bool
) -> None:
    dr = parse_date_range(text)
    assert dr is not None, f"failed to parse: {text!r}"
    assert (dr.start_year, dr.start_month) == (sy, sm)
    assert (dr.end_year, dr.end_month, dr.is_current) == (ey, em, cur)


@pytest.mark.parametrize(
    "text",
    [
        "no dates here at all",
        "a single 2020 year with no range",
        "phone 438-398-2741",  # a hyphenated number is not a date range
        "",
    ],
)
def test_parse_date_range_rejects_non_ranges(text: str) -> None:
    assert parse_date_range(text) is None


# --------------------------------------------------------------------- experience (must-parse)

# F1 — YYYY–YYYY, space-aligned, Title / Company,Location / bullets (the common North-American form)
_F1 = """PROFESSIONAL EXPERIENCE
Customer Service Manager                         2022 – 2025
Olympic Linen Supplies, Montréal
Managed daily delivery operations and driver coordination
Negotiated contract disputes directly with clients
Operations Coordinator                           2021 – 2022
Entreprises SMG Inc., Montréal
Booked contracts and assigned operators
LANGUAGES
French and English
"""

# F2 — Month YYYY dates, current role
_F2 = """Experience
Security Analyst                                 January 2021 – Present
Acme Corp, Toronto
Monitored SIEM alerts and triaged incidents
Help Desk Technician                             March 2018 – December 2020
Beta LLC
Resolved tier-1 tickets
Education
"""

# F3 — MM/YYYY dates + a tab between title and date
_F3 = """WORK EXPERIENCE
Software Engineer\t06/2019 – 09/2022
Globex, Remote
Built backend services
Skills
"""


def test_f1_yyyy_range_titlefirst() -> None:
    exp = extract_experience(_F1)
    assert [(e.title, e.company, e.start_date, e.end_date) for e in exp] == [
        ("Customer Service Manager", "Olympic Linen Supplies", "2022", "2025"),
        ("Operations Coordinator", "Entreprises SMG Inc.", "2021", "2022"),
    ]
    assert exp[0].location == "Montréal"
    assert len(exp[0].bullets) == 2 and len(exp[1].bullets) == 1


def test_f2_month_year_and_present() -> None:
    exp = extract_experience(_F2)
    assert [(e.title, e.company) for e in exp] == [
        ("Security Analyst", "Acme Corp"),
        ("Help Desk Technician", "Beta LLC"),
    ]
    assert exp[0].start_date == "January 2021" and exp[0].end_date == "Present"
    assert exp[1].end_date == "December 2020"


def test_f3_mm_yyyy_and_tab() -> None:
    exp = extract_experience(_F3)
    assert len(exp) == 1
    assert exp[0].title == "Software Engineer" and exp[0].company == "Globex"
    assert exp[0].start_date == "06/2019" and exp[0].end_date == "09/2022"


# ------------------------------------------------------ experience (must-degrade-safely → [])

_EXOTIC_NO_DATES = """EXPERIENCE
Did a lot of great things at various companies over the years
Also volunteered on weekends
"""

_DATES_MIDSENTENCE = """SUMMARY
From 2015 to 2020 I worked across several teams delivering value.
"""


@pytest.mark.parametrize(
    "text", [_EXOTIC_NO_DATES, _DATES_MIDSENTENCE, "", "No section headers here"]
)
def test_experience_degrades_to_empty(text: str) -> None:
    # No confident, date-range-anchored entry header → nothing extracted (raw_text fallback stays).
    assert extract_experience(text) == []


# --------------------------------------------------------------------- education

_EDU = """EDUCATION & CERTIFICATIONS
Undergraduate Certificate — Operational Cybersecurity              2024 – Present
Polytechnique Montréal
B.A., Accounting — Université du Québec à Montréal                 2012 – 2015
EXPERIENCE
"""


def test_education_multiline_and_singleline() -> None:
    edu = extract_education(_EDU)
    # Multi-line: institution on its own line; single-line: "degree — institution" in the label.
    assert len(edu) == 2
    assert edu[0].degree == "Undergraduate Certificate — Operational Cybersecurity"
    assert edu[0].institution == "Polytechnique Montréal"
    assert (edu[0].start_date, edu[0].end_date) == ("2024", "Present")
    assert edu[1].degree == "B.A., Accounting"
    assert edu[1].institution == "Université du Québec à Montréal"


def test_education_degrades_to_empty() -> None:
    assert extract_education("EDUCATION\nStudied hard, learned things\n") == []


# --------------------------------------------------------------- review regressions (no-fabricate)


def test_long_space_run_is_linear_not_redos() -> None:
    """A `pdftotext -layout` line with a huge space run and no trailing date must NOT hang
    (catastrophic backtracking) and must extract nothing wrong. Was ~3.2s at 400 spaces; the
    split-on-gap _entry_header is linear. Executing this at all guards the hang."""
    text = "EXPERIENCE\nSecurity Operations Analyst" + " " * 4000 + "Toronto, Ontario\nEducation\n"
    assert extract_experience(text) == []  # right-aligned tail is a place, not a date → no entry


def test_bullet_after_header_is_not_a_fabricated_company() -> None:
    """A title-only header followed by bullets must yield company='' (honest empty) and keep the
    bullet — never company='• Monitored …' (a fabricated field + a dropped first bullet)."""
    exp = extract_experience(
        "EXPERIENCE\n"
        "Security Analyst                         2020 – 2023\n"
        "• Monitored SIEM alerts and triaged incidents\n"
        "• Ran incident response\n"
        "Education\n"
    )
    assert len(exp) == 1
    assert exp[0].company == ""
    assert exp[0].bullets == [
        "Monitored SIEM alerts and triaged incidents",
        "Ran incident response",
    ]


def test_education_description_is_not_a_fabricated_institution() -> None:
    """A degree header followed by a 'Relevant coursework: …' description (not an institution line)
    must yield institution='' — not the coursework text."""
    edu = extract_education(
        "EDUCATION\n"
        "B.Sc. Computer Science                    2015 – 2019\n"
        "Relevant coursework: algorithms, operating systems, networks\n"
        "EXPERIENCE\n"
    )
    assert len(edu) == 1
    assert edu[0].institution == ""


def test_impossible_month_is_rejected_not_coerced() -> None:
    """An impossible MM in MM/YYYY rejects the token (not a fabricated bare year)."""
    assert parse_date_range("13/2020 – 09/2021") is None
    assert parse_date_range("01/2020 – 09/2021") is not None


def test_company_location_folds_city_province() -> None:
    """'Company, City, Province' → company / 'City, Province' (both short tails fold in)."""
    exp = extract_experience(
        "EXPERIENCE\nAnalyst                              2020 – 2023\nAcme Corp, Montreal, QC\n"
    )
    assert exp[0].company == "Acme Corp"
    assert exp[0].location == "Montreal, QC"
