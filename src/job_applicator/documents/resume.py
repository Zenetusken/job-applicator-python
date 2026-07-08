"""Resume parser and loader."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from job_applicator.documents.ocr import OCRService
from job_applicator.exceptions import DocumentError, JobApplicatorError, ResumeNotFoundError
from job_applicator.models import EducationEntry, ExperienceEntry, ResumeData
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.resume")

OCR_THRESHOLD = 100

# Section headers used for confidence scoring. Vocabulary aligned with
# _extract_skills_section so a "Core Competencies"/"Proficiencies" resume earns
# section credit too. Matched as line-anchored HEADERS (see _SECTION_HEADER_RE),
# never as bare substrings.
_KNOWN_SECTIONS = frozenset(
    (
        "experience",
        "expérience",
        "education",
        "formation",
        "éducation",
        "skills",
        "compétences",
        "competences",
        "competencies",
        "proficiencies",
        "certifications",
        "languages",
        "langues",
        "projects",
        "projets",
    )
)
_SECTION_HEADER_RE = re.compile(
    r"(?im)^\s*\*{0,2}\s*"
    r"(?:(?:technical|core|key|professional|relevant|soft)\s+)?"  # match "Core Competencies"
    r"(" + "|".join(sorted(_KNOWN_SECTIONS)) + r")\b"
)


# Résumé section-header recognition — case-insensitive, tolerant of a leading qualifier
# ("PROFESSIONAL Experience", "Technical Skills"), a trailing compound ("Education &
# Certifications"), markdown/ATX/colon wrapping, and inline content ("Skills: Python, …").
# SHARED so the parser (summary boundary) and ResumeDateValidator (date-section attribution)
# can't drift — the case-sensitive/exact-match versions did, and an all-caps qualified header
# like "PROFESSIONAL EXPERIENCE" slipped BOTH: it made `summary` swallow ~the whole document
# and forced a false "ordering issue" that aborts `tailor` on a valid CV.
_SECTION_QUALIFIER_RE = re.compile(
    r"^(?:professional|technical|core|key|relevant|soft|additional|other|primary|work)\s+",
    re.IGNORECASE,
)
_SECTION_HEAD: dict[str, str] = {
    "summary": "Summary",
    "profile": "Summary",
    "resume": "Summary",
    "résumé": "Summary",
    "objective": "Summary",
    "about": "Summary",
    "about me": "Summary",
    "career summary": "Summary",
    "profil": "Summary",
    "skills": "Skills",
    "compétences": "Skills",
    "competences": "Skills",
    "competencies": "Skills",
    "proficiencies": "Skills",
    "expertise": "Skills",
    "experience": "Experience",
    "expérience": "Experience",
    "expérience professionnelle": "Experience",
    "history": "Experience",
    "employment": "Experience",
    "employment history": "Experience",
    "work history": "Experience",
    # Distinct sections — the per-section ordering check must NOT compare a volunteer/internship
    # entry against a paid role across their real boundary (that invents a false inversion).
    "internship": "Internship",
    "internships": "Internship",
    "volunteer": "Volunteer",
    "volunteering": "Volunteer",
    "volunteer experience": "Volunteer",
    "education": "Education",
    "éducation": "Education",
    "academic": "Education",
    "formation": "Education",
    "formation et certifications": "Education",
    "certifications": "Certifications",
    "certification": "Certifications",
    "certificates": "Certifications",
    "licenses": "Certifications",
    "licences": "Certifications",
    "languages": "Languages",
    "language": "Languages",
    "langues": "Languages",
    "projects": "Projects",
    "project": "Projects",
    "projets": "Projects",
    "references": "References",
    "interests": "Interests",
    "awards": "Awards",
}


def section_header(line: str) -> str | None:
    """The canonical résumé section a header LINE denotes, or ``None`` if the line isn't a header.

    Case-insensitive; strips a leading qualifier + markdown/ATX/colon wrapping, takes the head word
    before any ``&``/``and``/``/``/``,`` compound, and matches a short header word — so
    ``PROFESSIONAL EXPERIENCE``, ``EDUCATION & CERTIFICATIONS``, ``**Summary**``, and
    ``Skills: Python, …`` all resolve, while a paragraph/bullet never does.
    """
    s = line.strip().strip("#").strip().strip("*").strip()
    if not s:
        return None
    # header word before any inline content ("Skills: Python" → "Skills"), qualifier stripped
    s = _SECTION_QUALIFIER_RE.sub("", s.split(":", 1)[0].strip()).strip()
    if len(s) > 35:  # a header word/phrase is short; a paragraph is not
        return None
    # Head word before any '&'/'and'/'/'/',' compound. Split on the LOWERED string so a capitalized
    # 'AND' ("EDUCATION AND CERTIFICATIONS") is split too — else the all-caps compound falls thru,
    # exactly the mis-bucketing this matcher exists to prevent.
    head = re.split(r"\s*(?:&|/|,|\band\b)\s+", s.lower(), maxsplit=1)[0].strip()
    return _SECTION_HEAD.get(head)


# --------------------------------------------------------------------------- date-range parsing
#
# ONE hardened date-range parser, shared by the structured extractors below AND
# ``ResumeDateValidator`` (documents/resume_tailor.py) — a single source so the two can't drift.
# Handles YYYY, Month YYYY (English + French), MM/YYYY; end tokens Present/Current/présent/actuel;
# separators –/—/-/to/à/au. Almost every entry-segmentation strategy anchors on detecting a date
# range, so this is the load-bearing primitive for multi-format support.

MONTH_MAP: dict[str, int] = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # French
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10,
    "novembre": 11, "décembre": 12, "decembre": 12,
    "janv": 1, "févr": 2, "fevr": 2, "avr": 4, "juil": 7, "déc": 12,
}  # fmt: skip

_MIN_YEAR, _MAX_YEAR = 1950, 2100  # plausible résumé-date bound — rejects phone/ID numeric noise
_PRESENT_RE = r"present|current|now|ongoing|pr[eé]sent|actuel(?:le)?"
# dash separators may be spaceless; word separators (to/à/au) require surrounding spaces
_SEP_RE = r"(?:\s*[-–—]\s*|\s+(?:to|à|au)\s+)"
# Month token built from MONTH_MAP keys (longest-first) so only REAL month names match — a title
# word before a year ("Manager 2022") can't be misread as a month-year, so parse_date_range stays
# robust when scanning raw lines (the ResumeDateValidator use), not just gap-split headers.
_MONTH_ALT = "|".join(sorted((re.escape(k) for k in MONTH_MAP), key=len, reverse=True))
_MON_SIDE = rf"(?:{_MONTH_ALT})\.?\s+\d{{4}}"  # "January 2020" / "janv. 2020"
_MMY_SIDE = r"\d{1,2}/\d{4}"
_YR_SIDE = r"\d{4}"
_SIDE = rf"(?:{_MON_SIDE}|{_MMY_SIDE}|{_YR_SIDE})"
_RANGE_RE = re.compile(rf"({_SIDE}){_SEP_RE}({_SIDE}|{_PRESENT_RE})", re.IGNORECASE)
_DATE_SUB = rf"(?:{_SIDE}){_SEP_RE}(?:{_SIDE}|{_PRESENT_RE})"
# A line-tail is a date range iff it is EXACTLY one (anchored). Used to test the text AFTER a
# right-alignment gap — cheap/linear on the short tail. (Do NOT put a lazy `.+?` prefix in front of
# this: `.+?` + `\s{2,}` + `\s*` all competing over one whitespace run backtracks catastrophically
# on the long space runs a `pdftotext -layout` PDF emits — a real parser hang. _entry_header splits
# in Python instead so the label and the gap can never overlap.)
_DATE_ONLY_RE = re.compile(rf"^{_DATE_SUB}$", re.IGNORECASE)
_GAP_RE = re.compile(r"\s{2,}|\t")  # right-alignment gap between a label and a right-aligned date


@dataclass(frozen=True)
class DateRange:
    """A parsed start→end date range. ``end_*`` are ``None`` when ``is_current``."""

    start_year: int | None
    start_month: int | None
    end_year: int | None
    end_month: int | None
    is_current: bool


def _parse_side(s: str) -> tuple[int, int | None] | None:
    """Parse one side of a range ('2020' | 'January 2020' | '01/2020') → (year, month | None)."""
    s = s.strip()
    m = re.fullmatch(r"([A-Za-zÀ-ÿ]{3,})\.?\s+(\d{4})", s)
    if m:
        month = MONTH_MAP.get(m.group(1).lower().rstrip("."))
        if month is None:
            return None  # an unknown word before a year is NOT a confident month → reject
        return int(m.group(2)), month
    m = re.fullmatch(r"(\d{1,2})/(\d{4})", s)
    if m:
        mm = int(m.group(1))
        if not 1 <= mm <= 12:
            return None  # an impossible month → not a valid MM/YYYY token; reject, don't fabricate
        return int(m.group(2)), mm
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return int(m.group(1)), None
    return None


def parse_date_range(text: str) -> DateRange | None:
    """Parse the first ``start – end`` date range in ``text``, or ``None`` if none is confident."""
    m = _RANGE_RE.search(text)
    if not m:
        return None
    start = _parse_side(m.group(1))
    if start is None or not (_MIN_YEAR <= start[0] <= _MAX_YEAR):
        return None
    end_raw = m.group(2).strip()
    if re.fullmatch(_PRESENT_RE, end_raw, re.IGNORECASE):
        return DateRange(start[0], start[1], None, None, True)
    end = _parse_side(end_raw)
    if end is None or not (_MIN_YEAR <= end[0] <= _MAX_YEAR):
        return None
    return DateRange(start[0], start[1], end[0], end[1], False)


# --------------------------------------------------------------- structured experience/education
#
# Conservative, section-scoped, date-range-anchored extraction. Emits an entry ONLY on a confident
# match; on any format it can't confidently parse it emits NOTHING (→ the raw_text fallback in the
# consumers), never a fabricated/garbage entry (no-failure-masking). The entry structure is the
# common "title-first" form (a header line ending in a right-aligned date range, then a
# company/institution line, then bullets); breadth comes from the multi-format date parser above,
# not from lowering the confidence bar.


def _entry_header(line: str) -> tuple[str, str, str, DateRange] | None:
    """An entry header = ``<label>`` + a right-alignment gap (2+ spaces / tab) + a date range at
    EOL. Returns (label, start_str, end_str, DateRange) or ``None`` — a date mid-sentence is NOT a
    header. Splits on the gap in Python (rightmost gap whose whole tail is a date range) rather than
    one lazy+greedy regex, so it stays LINEAR on long space runs (see `_DATE_ONLY_RE`)."""
    line = line.rstrip()
    for gap in reversed(list(_GAP_RE.finditer(line))):
        tail = line[gap.end() :].strip()
        if not _DATE_ONLY_RE.match(tail):
            continue
        label = line[: gap.start()].strip().strip("-–—,").strip()
        if len(label) < 2:
            return None
        dr = parse_date_range(tail)
        if dr is None:
            return None
        raw = _RANGE_RE.search(tail)  # raw start/end substrings for the string fields
        start_str = raw.group(1).strip() if raw else ""
        end_str = "Present" if dr.is_current else (raw.group(2).strip() if raw else "")
        return label, start_str, end_str, dr
    return None


def _section_body(text: str, section: str) -> list[str]:
    """Lines strictly inside ``section`` (until the NEXT section header of any kind)."""
    out: list[str] = []
    in_section = False
    for line in text.split("\n"):
        head = section_header(line)
        if head == section:
            in_section = True
            continue
        if in_section and head is not None:
            break
        if in_section:
            out.append(line)
    return out


def _debullet(s: str) -> str:
    return re.sub(r"^[\s•·▪‣◦*–—\-]+", "", s.strip()).strip()


def _is_place_tail(tail: str) -> bool:
    """A comma tail is a place iff it's short (≤3 words) and digit-free (a city/province, not part
    of a company name)."""
    return bool(tail) and len(tail.split()) <= 3 and not any(c.isdigit() for c in tail)


def _company_location(line: str) -> tuple[str, str]:
    """Split 'Company, City[, Province]' → (company, location); fold up to two short, digit-free
    trailing comma tails into the location (so 'Acme, Montreal, QC' → 'Acme' / 'Montreal, QC')."""
    company, sep, loc = line.strip().rpartition(",")
    loc = loc.strip()
    if not (sep and company.strip() and _is_place_tail(loc)):
        return line.strip(), ""
    company2, sep2, loc2 = company.rpartition(",")  # a second "…, City, Province" tail
    loc2 = loc2.strip()
    if sep2 and company2.strip() and _is_place_tail(loc2):
        return company2.strip(), f"{loc2}, {loc}"
    return company.strip(), loc


_DESC_LABEL_RE = re.compile(
    r"^[\w][\w &/'()-]{0,30}:"
)  # "Relevant coursework:", "GPA:", "Completed:"
_SKILL_ROW_LABELS = (
    "security ops",
    "threat & detection",
    "networking",
    "cloud & systems",
    "tools",
    "platforms",
)
_SKILL_ROW_LABEL_GAP_RE = re.compile(r"^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 &/+.-]{1,32})\s{2,}(.+)$")


def _strip_skill_row_label(part: str) -> str:
    """Drop fixed-width skills-grid labels without touching ordinary skill names.

    PDF extraction of the v1 CV yields rows such as ``Security ops            SIEM`` and
    ``Threat & detection MITRE ATT&CK``. The label is layout, not a skill. Keep the first real
    skill so matching/tailoring does not inherit label-contaminated names.
    """
    cleaned = part.rsplit("\t", 1)[-1].strip()
    gap = _SKILL_ROW_LABEL_GAP_RE.match(cleaned)
    if gap:
        label, value = gap.groups()
        if len(label.split()) <= 4 and value.strip():
            return value.strip()

    lower = cleaned.lower()
    for label in _SKILL_ROW_LABELS:
        prefix = f"{label} "
        if lower.startswith(prefix):
            value = cleaned[len(prefix) :].strip()
            if value:
                return value
    return cleaned


def _looks_like_entity(line: str) -> bool:
    """True if ``line`` plausibly NAMES an employer/school — a compact proper-noun phrase, NOT a
    bullet, a prose sentence, or a 'Label: …' description. Guards the company/institution slot: on a
    non-title-first résumé (bullets or a description right after the header) it returns False so the
    slot stays '' (honest empty) instead of grabbing an arbitrary line as a fabricated field
    (no-failure-masking). NOTE: it cannot disambiguate a company-first layout ('Company' on the
    header, 'Title' next) — that residual mis-label is a known title-first assumption."""
    s = line.strip()
    if not s or _debullet(s) != s:  # bullet-led (•/·/-/–/… ) → a bullet, not an employer/school
        return False
    if len(s) > 60 or len(s.split()) > 8:  # a description/sentence, not a compact name
        return False
    if _DESC_LABEL_RE.match(s):  # "Relevant coursework: …" — a description label, not a name
        return False
    return True


def _next_content_line(body: list[str], start: int) -> int:
    """Index of the next non-empty line at or after ``start`` (len(body) if none)."""
    j = start
    while j < len(body) and not body[j].strip():
        j += 1
    return j


def extract_experience(text: str) -> list[ExperienceEntry]:
    """Extract structured experience entries from a résumé (conservative; [] on no match)."""
    body = _section_body(text, "Experience")
    entries: list[ExperienceEntry] = []
    i = 0
    while i < len(body):
        header = _entry_header(body[i])
        if header is None:
            i += 1
            continue
        title, start_date, end_date, _dr = header
        company = location = ""
        cursor = i + 1
        j = _next_content_line(body, i + 1)
        # Only take the next line as the employer if it plausibly NAMES one — else leave company ''
        # and let the line fall through to the bullet loop (a bullet-as-company is a fabrication).
        if (
            j < len(body)
            and _entry_header(body[j]) is None
            and section_header(body[j]) is None
            and _looks_like_entity(body[j])
        ):
            company, location = _company_location(body[j])
            cursor = j + 1
        bullets: list[str] = []
        while cursor < len(body):
            if not body[cursor].strip():
                cursor += 1
                continue
            if _entry_header(body[cursor]) is not None or section_header(body[cursor]) is not None:
                break
            bullets.append(_debullet(body[cursor]))
            cursor += 1
        entries.append(
            ExperienceEntry(
                title=title,
                company=company,
                location=location,
                start_date=start_date,
                end_date=end_date,
                bullets=bullets,
            )
        )
        i = cursor
    return entries


def _degree_institution(label: str) -> tuple[str, str]:
    """Split a single-line 'Degree — Institution' label; else (label, '')."""
    parts = re.split(r"\s+[–—-]\s+", label, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return label, ""


def extract_education(text: str) -> list[EducationEntry]:
    """Extract structured education entries from résumé text (conservative; [] on no match)."""
    body = _section_body(text, "Education")
    entries: list[EducationEntry] = []
    i = 0
    while i < len(body):
        header = _entry_header(body[i])
        if header is None:
            i += 1
            continue
        label, start_date, end_date, _dr = header
        cursor = i + 1
        j = _next_content_line(body, i + 1)
        if (
            j < len(body)
            and _entry_header(body[j]) is None
            and section_header(body[j]) is None
            and _looks_like_entity(body[j])  # a school NAME, not a description/coursework line
        ):
            degree, institution = label, body[j].strip()  # institution on its own line
            cursor = j + 1
        else:
            degree, institution = _degree_institution(label)  # single-line "Degree — Institution"
        # Skip any description lines until the next header/section (EducationEntry has no bullets).
        while cursor < len(body):
            if not body[cursor].strip():
                cursor += 1
                continue
            if _entry_header(body[cursor]) is not None or section_header(body[cursor]) is not None:
                break
            cursor += 1
        entries.append(
            EducationEntry(
                degree=degree,
                institution=institution,
                location="",
                start_date=start_date,
                end_date=end_date,
            )
        )
        i = cursor
    return entries


class ResumeLoader:
    """Load and parse resume files."""

    def __init__(self) -> None:
        self._ocr_service: OCRService | None = None

    @property
    def ocr_service(self) -> OCRService:
        if self._ocr_service is None:
            self._ocr_service = OCRService()
        return self._ocr_service

    def load(self, path: str | Path, ocr_mode: str = "auto") -> ResumeData:
        """Load a resume from file. Supports PDF, DOCX, plain text, and images.

        Guarantees a TYPED failure: any underlying parser error (python-docx,
        PyMuPDF, OCR, ...) is wrapped as ``DocumentError`` so callers only ever see a
        ``JobApplicatorError`` — never a raw third-party traceback.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise ResumeNotFoundError(f"Resume not found: {file_path}")

        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                data = self._load_pdf(file_path, ocr_mode=ocr_mode)
            elif suffix == ".docx":
                data = self._load_docx(file_path)
            elif suffix in (".txt", ".md"):
                data = self._load_text(file_path)
            elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
                data = self._load_image(file_path, ocr_mode=ocr_mode)
            else:
                fmt = suffix or "(no file extension)"
                raise DocumentError(f"Unsupported resume format: {fmt} ({file_path.name})")
        except JobApplicatorError:
            raise
        except Exception as exc:
            raise DocumentError(
                f"Could not parse resume {file_path.name}: {type(exc).__name__}: {exc}"
            ) from exc

        if not data.raw_text.strip():
            raise DocumentError(
                f"Resume has no extractable text: {file_path.name} (the file is empty, or its "
                "text could not be extracted — for a scanned PDF try --ocr-mode on)"
            )
        return data

    def _load_docx(self, path: Path) -> ResumeData:
        """Load a DOCX resume.

        Walks the document body in order, extracting BOTH paragraphs and table cells. Real
        résumés routinely put the contact header and the skills section in tables; reading
        only ``doc.paragraphs`` (the prior behaviour) silently dropped that content from the
        text, so the email/phone/skills never reached matching or tailoring — a tailored CV
        could even go out missing the contact block (audit AI-H5). Document order is preserved
        so a leading contact table stays first, where ``parse_text``'s first-line name
        heuristic can find it.
        """
        try:
            from docx import Document
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise DocumentError("python-docx not installed. Run: pip install python-docx") from exc

        doc = Document(str(path))
        parts: list[str] = []
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                line = Paragraph(child, doc).text
                if line.strip():
                    parts.append(line)
            elif isinstance(child, CT_Tbl):
                # Append each unique cell's text. python-docx repeats the same cell across a
                # merged grid span, so dedup by the underlying <w:tc> element to avoid doubling
                # merged contact/header cells. Best-effort PER TABLE: a structurally malformed
                # grid (e.g. an orphan vMerge continuation — a real export artifact) raises in
                # python-docx's grid walk, so skip that one table with a logged warning rather
                # than failing the whole résumé. The table is collected atomically (extend only
                # on success) so a mid-walk failure contributes nothing, never a partial table.
                # Paragraphs and well-formed tables are unaffected — strictly more robust than
                # the prior paragraphs-only reader, and the skip is disclosed (not masked).
                try:
                    cell_texts: list[str] = []
                    seen: set[object] = set()
                    for row in Table(child, doc).rows:
                        for cell in row.cells:
                            if cell._tc in seen:
                                continue
                            seen.add(cell._tc)
                            cell_text = cell.text.strip()
                            if cell_text:
                                cell_texts.append(cell_text)
                    parts.extend(cell_texts)
                except Exception as exc:
                    logger.warning("Skipping a malformed table in %s: %s", path.name, exc)
        text = "\n".join(parts)
        return self.parse_text(text, method="docx")

    def _load_pdf(self, path: Path, ocr_mode: str = "auto") -> ResumeData:
        if ocr_mode not in {"auto", "on", "off"}:
            raise DocumentError(f"Invalid ocr_mode '{ocr_mode}'. Valid modes: auto, on, off")

        if self._is_password_protected(path):
            raise DocumentError(
                f"PDF {path} is password-protected. "
                "Remove the password or save an unprotected copy."
            )

        if ocr_mode == "on":
            text = self.ocr_service.extract_text_from_pdf(path)
            return self.parse_text(text, method="ocr")

        if ocr_mode == "off":
            result = self._pdf_consensus(path, methods=("pdftotext", "pymupdf"))
            if len(result.raw_text.strip()) < OCR_THRESHOLD:
                raise DocumentError(
                    f"PDF {path} contains insufficient extractable text; "
                    "enable OCR with --ocr-mode auto or --force-ocr"
                )
            return result

        # ocr_mode == "auto": run text extractors, fall back to OCR if they are short.
        try:
            auto_result = self._pdf_consensus(path, methods=("pdftotext", "pymupdf"))
            if len(auto_result.raw_text.strip()) >= OCR_THRESHOLD:
                return auto_result
            fallback_confidence = auto_result.parse_confidence
            fallback_result = auto_result
        except DocumentError:
            fallback_confidence = 0.0
            fallback_result = None

        logger.info("OCR fallback triggered for %s", path)
        try:
            ocr_text = self.ocr_service.extract_text_from_pdf(path)
            ocr_result = self.parse_text(ocr_text, method="ocr")
            if fallback_result is None or ocr_result.parse_confidence > fallback_confidence:
                logger.info(
                    "OCR result is better for %s (confidence=%.2f)",
                    path,
                    ocr_result.parse_confidence,
                )
                best = ocr_result
            else:
                logger.info("OCR did not improve confidence for %s; keeping text extractor", path)
                best = fallback_result
        except DocumentError:
            if fallback_result is None:
                raise
            logger.warning("OCR failed for %s; using best text extractor", path)
            best = fallback_result

        # Guard against a silently-empty/short parse: if every text extractor
        # AND OCR yielded less than the OCR_THRESHOLD, fail loudly rather than
        # return a ResumeData with too little text for downstream match/tailor.
        if len(best.raw_text.strip()) < OCR_THRESHOLD:
            raise DocumentError(
                f"PDF {path} contains insufficient extractable text "
                f"({len(best.raw_text.strip())} chars); "
                "enable OCR with --ocr-mode on or --force-ocr"
            )
        return best

    def _run_pdftotext(self, path: Path) -> str:
        """Extract text using pdftotext; return empty string on failure."""
        import subprocess  # nosec B404
        import tempfile

        exe = shutil.which("pdftotext")
        if not exe:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            try:
                result = subprocess.run(  # noqa: S603 # nosec B603
                    [exe, "-layout", str(path), tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return Path(tmp.name).read_text(encoding="utf-8")
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return ""

    def _run_pymupdf(self, path: Path) -> str:
        """Extract text using PyMuPDF; return empty string on failure."""
        try:
            import fitz
        except ImportError:
            return ""
        try:
            doc = fitz.open(str(path))
            try:
                return "".join(page.get_text() for page in doc)
            finally:
                doc.close()
        except Exception:
            return ""

    def _pdf_consensus(self, path: Path, methods: tuple[str, ...]) -> ResumeData:
        """Extract PDF text with the given methods and return the best parse.

        Each method produces a candidate; the candidate with the highest
        heuristic confidence wins. This protects against a single parser
        returning garbled or incomplete text.
        """
        candidates: list[tuple[ResumeData, str]] = []
        for method in methods:
            raw = self._extract_pdf_with_method(path, method)
            if raw.strip():
                parsed = self.parse_text(raw, method=method)
                candidates.append((parsed, method))
                logger.debug(
                    "PDF %s via %s: confidence=%.2f", path, method, parsed.parse_confidence
                )

        if not candidates:
            raise DocumentError(
                f"PDF {path} contains insufficient extractable text; "
                "enable OCR with --ocr-mode auto or --force-ocr"
            )

        candidates.sort(key=lambda item: item[0].parse_confidence, reverse=True)
        best, method = candidates[0]
        logger.info(
            "Selected PDF parser %s for %s (confidence=%.2f)", method, path, best.parse_confidence
        )
        return best

    def _extract_pdf_with_method(self, path: Path, method: str) -> str:
        if method == "pdftotext":
            return self._run_pdftotext(path)
        if method == "pymupdf":
            return self._run_pymupdf(path)
        if method == "ocr":
            return self.ocr_service.extract_text_from_pdf(path)
        raise DocumentError(f"Unknown PDF extraction method: {method}")

    def _is_password_protected(self, path: Path) -> bool:
        """Best-effort detection of password-protected PDFs.

        PyMuPDF OPENS an encrypted PDF without raising and sets ``needs_pass`` —
        so check that flag rather than only whether ``fitz.open`` raised (which it
        usually doesn't). Falls back to sniffing the exception message for the
        rarer open-time failure.
        """
        try:
            import fitz
        except ImportError:
            return False
        try:
            doc = fitz.open(str(path))
            try:
                # needs_pass: encrypted AND not yet authenticated (the real gate).
                return bool(getattr(doc, "needs_pass", False))
            finally:
                doc.close()
        except Exception as exc:
            message = str(exc).lower()
            return "password" in message or "encrypted" in message or "crypt" in message

    @staticmethod
    def _extract_phone(text: str) -> str:
        """First phone-like run (10-15 digits), or "".

        Rejects numeric tables / year lists like "2019 2020 2021 2022 2023"
        (3+ space-separated 4-digit groups), which the old ``{10,}``-char pattern
        false-matched as a phone — polluting the ``phone`` field and inflating
        parse confidence.
        """
        for match in re.finditer(r"[\+]?\d[\d\s\-\(\)]{8,}\d", text):
            candidate = match.group(0).strip()
            digits = sum(c.isdigit() for c in candidate)
            if not 10 <= digits <= 15:
                continue
            groups = candidate.split()
            if len(groups) >= 3 and all(g.isdigit() and len(g) == 4 for g in groups):
                continue
            return candidate
        return ""

    def _compute_confidence(self, text: str) -> float:
        """Heuristic confidence based on length, contact info, and section coverage."""
        stripped = text.strip()
        if not stripped:
            return 0.0

        score = 0.0
        # Length signal: up to 0.3 for a reasonably long resume.
        score += min(len(stripped) / 2000, 0.3)

        # Contact signals.
        if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", stripped):
            score += 0.2
        if self._extract_phone(stripped):
            score += 0.2

        # Section coverage — count DISTINCT section headers anchored at line
        # starts (a bare "skills" inside "no skills listed" must not inflate it).
        found = {w.lower() for w in _SECTION_HEADER_RE.findall(stripped)}
        score += 0.1 * len(found)

        return min(round(score, 2), 1.0)

    def _load_image(self, path: Path, ocr_mode: str = "auto") -> ResumeData:
        if ocr_mode == "off":
            raise DocumentError(f"Image resume {path} requires OCR, but ocr_mode is 'off'")
        text = self.ocr_service.extract_text_from_image(path)
        return self.parse_text(text, method="ocr")

    def _load_text(self, path: Path) -> ResumeData:
        """Load plain text resume."""
        text = path.read_text(encoding="utf-8")
        return self.parse_text(text, method="text")

    def parse_text(self, text: str, method: str = "") -> ResumeData:
        """Parse raw text into structured ResumeData."""
        lines = text.strip().split("\n")
        name = lines[0].strip() if lines else ""
        name = re.sub(r"^[*_]+|[*_]+$", "", name).strip()

        # Extract email
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        email = email_match.group(0) if email_match else ""

        # Extract phone (rejects numeric tables / year lists — see _extract_phone).
        phone = self._extract_phone(text)

        # Extract skills section
        skills: list[str] = []
        summary = ""
        skills_section = self._extract_skills_section(text)

        if skills_section is not None:
            # Parse skills - handle various formats:
            # - Comma separated: "Python, Java, C++"
            # - Bullet list: "•\n\nSkill Name"
            # - Line list: "Skill1\nSkill2\nSkill3"
            # - Two-column bullets: "Skill1 • Skill2"

            # Remove bullet characters and empty lines. PDF fixed-width skills grids often wrap a
            # row by indenting the continuation line under the skill column; join those before
            # splitting so "threat" + "intelligence" stays one skill.
            lines = skills_section.split("\n")
            logical_lines: list[str] = []
            for line in lines:
                raw = line.rstrip()
                stripped = raw.strip()
                # Skip empty lines, bullets, and separators
                if stripped and stripped not in ("•", "·", "-", "|", "/"):
                    indent = len(raw) - len(raw.lstrip(" "))
                    if indent >= 8 and logical_lines and not re.match(r"^[•·\-\|/]", stripped):
                        logical_lines[-1] = f"{logical_lines[-1]} {stripped}"
                    else:
                        logical_lines.append(stripped)

            clean_lines = []
            for line in logical_lines:
                stripped = line.strip()
                if stripped:
                    # Split on middle bullets (two-column format)
                    parts = re.split(r"\s+[•·]\s+", stripped)
                    for part in parts:
                        # Drop a leading "Category<tab/gap>" label: skills grids put the row
                        # label before the first skill, which otherwise contaminates matching
                        # (e.g. "Networking<TAB>TCP/IP" must parse as "TCP/IP").
                        part = _strip_skill_row_label(part)
                        # Remove leading bullets
                        part = re.sub(r"^[•·\-\|/]\s*", "", part).strip()
                        # >= 2 so common short skills survive (Go, C#, AI, ML, UX);
                        # single chars are dropped as too noisy/ambiguous.
                        if part and len(part) >= 2:
                            clean_lines.append(part)

            # Skills may be comma-separated (often wrapping across several lines),
            # one-per-line, or a mix. Split any line containing commas into its
            # tokens and keep comma-free lines as single skills, so a wrapped comma
            # list ("Python, asyncio,\npytest, Git") parses to individual skills
            # instead of one blob per line. (The old "comma-split only when a single
            # line" path mis-parsed every wrapped comma list into per-line blobs,
            # which then matched nothing during skill-coverage scoring.)
            for clean_line in clean_lines:
                if "," in clean_line:
                    # Split on commas OUTSIDE parentheses so "Linux (Fedora, CLI, Bash)" stays ONE
                    # skill (like "cloud security (IaaS/SaaS)") instead of "Linux (Fedora" + "Bash)"
                    # — stray-paren tokens that embed as garbage vectors (audit AI-H4).
                    skills.extend(
                        tok.strip()
                        for tok in re.split(r",\s*(?![^(]*\))", clean_line)
                        if len(tok.strip()) >= 2
                    )
                else:
                    skills.append(clean_line)

        # Extract summary/objective — find the Summary/Objective/Profile section via the shared
        # robust header matcher, bounded by the NEXT section header; else the first paragraph after
        # the contact block. (The old case-sensitive "Summary" test + exact-case header break let an
        # all-caps 'PROFESSIONAL EXPERIENCE' slip → the fallback ran to EOF and summary swallowed
        # ~the whole document; the shared matcher stops at any real header regardless of case/form.)
        summary_lines = text.strip().split("\n")
        start: int | None = None
        inline_summary = ""
        found_header = False
        for i, ln in enumerate(summary_lines):
            if section_header(ln) == "Summary":
                start = i + 1
                found_header = True
                # Capture inline prose on the header line itself ("Summary: <one-line summary>").
                inline_summary = ln.split(":", 1)[1].strip() if ":" in ln else ""
                break
        if start is None:
            # No explicit Summary header: begin after the name + contact (email/phone) lines.
            contact_end = 0
            for i, ln in enumerate(summary_lines[:5]):
                if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", ln) or re.search(
                    r"[\+]?[\d\s\-\(\)]{10,}", ln
                ):
                    contact_end = i + 1
            start = contact_end
        paragraph_lines: list[str] = [inline_summary] if inline_summary else []
        for ln in summary_lines[start:]:
            stripped = ln.strip()
            if not stripped:
                if paragraph_lines:
                    break
                continue
            if section_header(ln):  # reached the next section → stop accumulating
                break
            paragraph_lines.append(stripped)
        if paragraph_lines:
            summary = " ".join(paragraph_lines)
            # In the FALLBACK case (no explicit header) reject a too-short / all-caps stray header;
            # with an explicit Summary header the content IS the summary, so keep it as-is.
            if not found_header and (len(summary) <= 50 or summary.isupper()):
                summary = ""

        # Drop a leading setext/markdown underline that can follow a header.
        summary = re.sub(r"^[\-=~*]+\s*\n?", "", summary).strip()

        # Structured experience/education — conservative, section-scoped, date-range-anchored.
        # Emits entries only on a confident parse (else []), so a format it can't read degrades to
        # the raw_text fallback in the consumers rather than fabricating (no-failure-masking).
        experience = extract_experience(text)
        education = extract_education(text)

        confidence = self._compute_confidence(text)
        logger.info(
            "Parsed resume: name=%s, skills=%d, exp=%d, edu=%d, confidence=%.2f",
            name,
            len(skills),
            len(experience),
            len(education),
            confidence,
        )
        return ResumeData(
            raw_text=text,
            name=name,
            email=email,
            phone=phone,
            summary=summary,
            skills=skills,
            experience=experience,
            education=education,
            parse_confidence=confidence,
            parse_method=method,
        )

    def _extract_skills_section(self, text: str) -> str | None:
        """Extract the text between the Skills header and the next section header.

        Returns None if no standalone Skills section header is found.
        Handles markdown bold headers ("**Skills**") and optional colon.
        """
        # Match a standalone Skills header (optional markdown bold, optional
        # colon, optional leading whitespace). Also allow inline skills after
        # the colon on the same line, e.g. "Skills: Python, Java". Recognizes
        # the common qualified variants the tailor uses too ("Technical
        # Skills", "Core Competencies", "Key Skills", etc.).
        pattern = re.compile(
            r"^\s*\*{0,2}\s*"
            r"(?:(?:Technical|Core|Key|Professional|Relevant|Soft)\s+)?"
            r"(?:Skills|Competencies|Proficiencies)"
            r"\s*\*{0,2}\s*:?\s*(.*)$",
            re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(text)
        if not match:
            return None

        inline_skills = match.group(1).strip()
        start = match.end()
        remaining = text[start:]

        # If there were inline skills after the colon, prepend them to the section
        if inline_skills:
            remaining = inline_skills + "\n" + remaining

        # Strip markdown/setext underline lines (e.g. "------" or "======") that
        # immediately follow the header, so they are not parsed as a skill.
        remaining = re.sub(r"^[\-=~*]+\s*\n", "", remaining, count=1)

        # Find the next known section header (allow inline content after colon too)
        next_header = re.compile(
            r"^\s*\*{0,2}\s*(?:Experience|Education|Certifications|Languages|Interests|Projects|Volunteer|References|Awards)\s*\*{0,2}\s*:?",
            re.IGNORECASE | re.MULTILINE,
        )
        next_match = next_header.search(remaining)
        if next_match:
            return remaining[: next_match.start()]
        return remaining
