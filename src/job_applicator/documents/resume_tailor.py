"""LLM-powered resume tailoring — rewrites resume content for a specific job."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import TYPE_CHECKING

from job_applicator.config import LLMConfig
from job_applicator.documents.grounding_verifier import GroundingVerifier
from job_applicator.documents.resume import MONTH_MAP, parse_date_range, section_header
from job_applicator.documents.style_analyzer import StyleAnalyzer
from job_applicator.exceptions import ConfigError, GroundingUnavailableError, LLMError
from job_applicator.models import (
    DateAuditResult,
    JobListing,
    ResumeData,
    StyleGuide,
    TailoredResume,
)
from job_applicator.utils.language import resolve_output_language
from job_applicator.utils.llm import (
    CircuitOpenError,
    LLMRuntime,
    litellm_completion_kwargs,
    litellm_model,
    quiet_litellm,
)
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

if TYPE_CHECKING:
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.embeddings.matching import JobMatcher, MatchResult

logger = get_logger("documents.resume_tailor")


def _alnum_boundary_pattern(term: str) -> re.Pattern[str]:
    """Case-insensitive pattern matching ``term`` not flanked by alphanumerics.

    Used instead of ``\\b`` so partial matches like "Java" inside "JavaScript"
    are rejected, while still matching terms that end in symbols (e.g. "C++").
    """
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


@dataclass
class ResumeSection:
    """A parsed section of a resume."""

    name: str
    text: str
    start_line: int
    end_line: int


KNOWN_HEADERS: frozenset[str] = frozenset(
    {
        "summary",
        "experience",
        "skills",
        "education",
        "certifications",
        "projects",
        "objective",
        "profile",
        "resume",
        "résumé",
        "work experience",
        "employment",
        "qualifications",
        "achievements",
        "interests",
        "languages",
        "references",
        "volunteer",
        "awards",
        "professional summary",
        "professional experience",
        "work history",
        "technical skills",
        "core competencies",
        "additional information",
        "profil",
        "compétences",
        "competences",
        "expérience",
        "expérience professionnelle",
        "experience professionnelle",
        "formation",
        "formation et certifications",
        "éducation",
        "éducation & certifications",
        "education & certifications",
        "langues",
        "projets",
        "bénévolat",
        "benevolat",
        "références",
    }
)

_COLON_HEADER_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s*$")


def _is_section_header(stripped: str) -> bool:
    """Return True if *stripped* line looks like a resume section header."""
    normalized = re.sub(r"\s+", " ", stripped.strip())
    if section_header(normalized) is not None:
        return True
    if normalized.lower() in KNOWN_HEADERS:
        return True
    if _COLON_HEADER_RE.match(stripped):
        return True
    return False


def _looks_like_section_header(stripped: str) -> bool:
    """Return True if stripped line is a known section header.

    Strips markdown bold so headers like '**Languages**' are recognized.
    """
    cleaned = re.sub(r"^\*+", "", stripped)
    cleaned = re.sub(r"\*+$", "", cleaned).strip()
    return _is_section_header(cleaned)


def _split_csv_outside_parentheses(text: str) -> list[str]:
    """Split comma lists while keeping parenthetical skill names intact."""
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
            current.append(char)
            continue
        if char == ")":
            depth = max(0, depth - 1)
            current.append(char)
            continue
        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _source_section(text: str, canonical: str) -> tuple[str, list[str]] | None:
    """Return ``(source header, body lines)`` for a canonical résumé section."""
    lines = text.replace("\f", "\n").splitlines()
    start: int | None = None
    header = ""
    for index, line in enumerate(lines):
        if section_header(line) == canonical:
            start = index + 1
            header = line.strip().strip("*").strip()
            break
    if start is None:
        return None

    body: list[str] = []
    for line in lines[start:]:
        if section_header(line) is not None:
            break
        body.append(line.rstrip())
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return None
    return header, body


def _section_bounds(lines: list[str], canonical: str) -> tuple[int, int] | None:
    """Return ``[start, end)`` bounds for a canonical section in generated text."""
    start: int | None = None
    for index, line in enumerate(lines):
        if section_header(line) == canonical:
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if section_header(lines[index]) is not None:
            end = index
            break
    return start, end


def _has_bold_section_headers(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*\*\*[^*\n]+\*\*\s*$", text))


def _format_source_section_header(source_header: str, *, bold: bool) -> str:
    header = source_header.title()
    return f"**{header}**" if bold else header


def _normalize_source_owned_fragment(text: str) -> str:
    normalized = text.lower().replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[*_`]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _insertion_index(lines: list[str], before: tuple[str, ...]) -> int:
    for canonical in before:
        bounds = _section_bounds(lines, canonical)
        if bounds is not None:
            return bounds[0]
    return len(lines)


def parse_sections(text: str) -> list[ResumeSection]:
    """Parse resume text into sections by detecting headers."""
    lines = text.split("\n")
    headers: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_section_header(stripped):
            headers.append((i, stripped))

    if not headers:
        return [
            ResumeSection(
                name="Full Document",
                text=text,
                start_line=0,
                end_line=len(lines) - 1,
            )
        ]

    sections: list[ResumeSection] = []

    for idx, (line_num, header_name) in enumerate(headers):
        start = line_num + 1
        if idx + 1 < len(headers):
            end = headers[idx + 1][0] - 1
        else:
            end = len(lines) - 1
        section_text = "\n".join(lines[start : end + 1]).strip()
        sections.append(
            ResumeSection(
                name=header_name,
                text=section_text,
                start_line=start,
                end_line=end,
            )
        )

    return sections


@dataclass
class ParsedDate:
    """A parsed date range from a resume entry."""

    label: str
    section: str
    start_year: int | None
    start_month: int | None
    end_year: int | None
    end_month: int | None
    is_current: bool


class ResumeDateValidator:
    """Parse dates from résumé text and run two LIGHT coherence checks.

    Implemented:
    - Chronological ordering (most recent first, within each section)
    - Staleness of the newest entry (its end date > STALE_THRESHOLD_YEARS ago, unless it is
      currently "Present") — a soft "have you updated this CV?" signal.
    - Experience gap/overlap surfacing when the heuristic parser has enough structured dates.

    The date parser is heuristic (regex over raw_text), so treat these signals as advisory, never
    blocking, unless a future structured-extraction pass supplies higher-confidence role ranges.
    """

    STALE_THRESHOLD_YEARS = 2

    def __init__(self, reference_date: datetime | None = None) -> None:
        self._now = reference_date or datetime.now()

    def audit(self, resume: ResumeData) -> DateAuditResult:
        """Audit all dates in a resume.

        Returns:
            DateAuditResult with parsed entries, warnings, and flags
        """
        entries = self._parse_all_dates(resume.raw_text)
        warnings: list[str] = []
        ordering_issues: list[str] = []
        staleness_issues: list[str] = []
        employment_gaps: list[str] = []
        overlap_issues: list[str] = []

        # Check chronological ordering within each section
        by_section: dict[str, list[ParsedDate]] = {}
        for e in entries:
            by_section.setdefault(e.section, []).append(e)

        for section, section_entries in by_section.items():
            for i in range(len(section_entries) - 1):
                curr = section_entries[i]
                nxt = section_entries[i + 1]
                curr_end = self._date_sort_key(curr, use_end=True)
                nxt_end = self._date_sort_key(nxt, use_end=True)
                if curr_end < nxt_end:
                    ordering_issues.append(
                        f"{section}: '{curr.label}' ({self._fmt_date(curr)}) "
                        f"should come after '{nxt.label}' ({self._fmt_date(nxt)})"
                    )

        employment_gaps, overlap_issues = self._employment_gap_and_overlap_issues(entries)

        # Check staleness
        latest_ts = 0
        latest_label = ""
        for e in entries:
            ts = self._date_sort_key(e, use_end=True)
            if ts > latest_ts:
                latest_ts = ts
                latest_label = e.label

        latest_dt = self._ts_to_datetime(latest_ts)
        if latest_dt:
            months_old = (self._now.year - latest_dt.year) * 12 + self._now.month - latest_dt.month
            if months_old > self.STALE_THRESHOLD_YEARS * 12:
                staleness_issues.append(
                    f"Most recent entry '{latest_label}' is dated "
                    f"{latest_dt.strftime('%B %Y')} — "
                    f"{months_old // 12} years and {months_old % 12} months ago. "
                    f"CV may be outdated."
                )

        # (Removed: education-age staleness. Flagging education that "ended >10 years ago" is noise
        # for an experienced career-changer — old education is normal, not a red flag — and it
        # mis-fired on in-progress coursework. The genuine red flag is employment GAPS, which this
        # validator does NOT detect; tracked as a follow-up rather than kept as a wrong signal.)

        # Check for missing dates
        for e in entries:
            if e.start_year is None and e.end_year is None:
                warnings.append(f"Entry '{e.label}' has no parseable dates.")

        # Generate summary warnings
        if ordering_issues:
            warnings.append(
                f"{len(ordering_issues)} ordering issue(s) found — "
                "entries should be most recent first."
            )

        if staleness_issues:
            warnings.append("CV may be outdated — review staleness warnings below.")
        if employment_gaps:
            warnings.append("Potential employment gap(s) found — review before sending.")
        if overlap_issues:
            warnings.append("Potential overlapping experience dates found — review before sending.")

        # Build result
        from job_applicator.models import DateEntry

        result_entries = []
        for e in entries:
            result_entries.append(
                DateEntry(
                    label=e.label,
                    section=e.section,
                    start=self._fmt_date_part(e.start_year, e.start_month),
                    end="Present" if e.is_current else self._fmt_date_part(e.end_year, e.end_month),
                    is_current=e.is_current,
                )
            )

        # Find earliest/latest
        all_starts = [
            self._date_sort_key(e, use_end=False) for e in entries if e.start_year is not None
        ]
        all_ends = [
            self._date_sort_key(e, use_end=True)
            for e in entries
            if e.end_year is not None or e.is_current
        ]

        latest_str = ""
        earliest_str = ""
        if all_ends:
            latest_dt = self._ts_to_datetime(max(all_ends))
            if latest_dt:
                latest_str = latest_dt.strftime("%B %Y")
        if all_starts:
            earliest_dt = self._ts_to_datetime(min(all_starts))
            if earliest_dt:
                earliest_str = earliest_dt.strftime("%B %Y")

        return DateAuditResult(
            entries=result_entries,
            warnings=warnings,
            ordering_issues=ordering_issues,
            staleness_issues=staleness_issues,
            employment_gaps=employment_gaps,
            overlap_issues=overlap_issues,
            is_stale=bool(staleness_issues),
            is_ordered=not ordering_issues,
            latest_date=latest_str,
            earliest_date=earliest_str,
        )

    def _employment_gap_and_overlap_issues(
        self, entries: list[ParsedDate]
    ) -> tuple[list[str], list[str]]:
        experience_entries = [
            e
            for e in entries
            if ("experience" in e.section.lower() or "expérience" in e.section.lower())
            and e.start_year is not None
        ]
        ranges: list[tuple[ParsedDate, int, int]] = []
        gaps: list[str] = []
        overlaps: list[str] = []
        for entry in experience_entries:
            start = self._date_sort_key(entry, use_end=False)
            end = self._date_sort_key(entry, use_end=True)
            if start and end:
                if end < start:
                    overlaps.append(
                        f"'{entry.label}' has an impossible date range ({self._fmt_date(entry)})"
                    )
                    continue
                ranges.append((entry, start, end))
        ranges.sort(key=lambda item: item[1])

        for (prev, _prev_start, prev_end), (nxt, next_start, _next_end) in pairwise(ranges):
            gap_months = self._months_between(prev_end, next_start)
            if gap_months > 6:
                gaps.append(
                    f"{gap_months} month gap between '{prev.label}' ({self._fmt_date(prev)}) "
                    f"and '{nxt.label}' ({self._fmt_date(nxt)})"
                )
            if next_start <= prev_end and not self._is_year_only_boundary(prev, nxt):
                overlaps.append(
                    f"'{prev.label}' ({self._fmt_date(prev)}) overlaps with "
                    f"'{nxt.label}' ({self._fmt_date(nxt)})"
                )
        return gaps, overlaps

    @staticmethod
    def _is_year_only_boundary(prev: ParsedDate, nxt: ParsedDate) -> bool:
        """Avoid false overlap warnings for adjacent year-only roles.

        A résumé line like ``2017-2021`` followed by ``2021-Present`` does not tell us whether the
        roles overlapped; treating year-only endpoints as Jan/Dec creates noisy false positives.
        Month-specific overlaps are still surfaced.
        """
        return (
            prev.end_year is not None
            and nxt.start_year is not None
            and prev.end_year == nxt.start_year
            and prev.end_month is None
            and nxt.start_month is None
        )

    @staticmethod
    def _months_between(first_end: int, next_start: int) -> int:
        first_year, first_month = divmod(first_end, 100)
        next_year, next_month = divmod(next_start, 100)
        return (next_year - first_year) * 12 + (next_month - first_month) - 1

    def _parse_all_dates(self, text: str) -> list[ParsedDate]:
        """Extract all date entries from resume text.

        Uses the ONE shared hardened parser (``documents.resume.parse_date_range`` — YYYY /
        Month YYYY en+fr / MM/YYYY / Present·Current·présent·actuel / –—-·to·à) so the validator and
        the structured extractors can't drift. Sections via the shared ``section_header`` matcher.
        """
        entries: list[ParsedDate] = []
        lines = text.split("\n")
        current_section = ""

        # Section headers — via the SHARED robust matcher (case-insensitive, qualifier/compound
        # tolerant), so 'PROFESSIONAL EXPERIENCE' / 'EDUCATION & CERTIFICATIONS' bucket correctly.
        # The old anchored ^EXPERIENCE$ regex rejected those → every entry fell to 'Unknown' → the
        # within-section ordering check compared across the real section boundary → a FALSE ordering
        # issue that aborts `tailor` on a valid CV.
        for i, line in enumerate(lines):
            sec = section_header(line)
            if sec:
                current_section = sec
                continue

            date_range = parse_date_range(line)
            if date_range is None:
                continue

            entries.append(
                ParsedDate(
                    label=self._find_label(lines, i),
                    section=current_section or "Unknown",
                    start_year=date_range.start_year,
                    start_month=date_range.start_month,
                    end_year=date_range.end_year,
                    end_month=date_range.end_month,
                    is_current=date_range.is_current,
                )
            )

        return entries

    def _find_label(self, lines: list[str], date_line_idx: int) -> str:
        """Find the label for a date entry by looking at preceding lines."""
        date_line = lines[date_line_idx].strip()
        if date_line:
            cleaned_date_line = re.sub(
                r"\(?\b(?:\d{1,2}/)?\d{4}\s*(?:-|–|—|to|à)\s*"
                r"(?:present|current|présent|actuel|(?:\d{1,2}/)?\d{4})\b\)?",
                "",
                date_line,
                flags=re.IGNORECASE,
            )
            cleaned_date_line = re.sub(r"\s+", " ", cleaned_date_line).strip(" ,-–—")
            if cleaned_date_line and not _looks_like_section_header(cleaned_date_line):
                return re.sub(r"\*{1,2}", "", cleaned_date_line).strip()

        for j in range(date_line_idx - 1, max(date_line_idx - 5, -1), -1):
            stripped = lines[j].strip()
            if not stripped:
                continue
            # Skip dates, bullets, and empty markers
            if re.match(r"^[\*•·\-\s]+$", stripped):
                continue
            if re.search(r"\d{4}\s*[-\u2013\u2014]", stripped):
                continue
            # Strip markdown bold
            cleaned = re.sub(r"\*{1,2}", "", stripped).strip()
            if cleaned:
                return cleaned
        return "Unknown"

    def _date_sort_key(self, entry: ParsedDate, use_end: bool = False) -> int:
        """Generate a sortable integer for a date entry."""
        if use_end:
            if entry.is_current:
                return self._now.year * 100 + self._now.month
            if entry.end_year:
                return entry.end_year * 100 + (entry.end_month or 12)
            return 0
        if entry.start_year:
            return entry.start_year * 100 + (entry.start_month or 1)
        return 0

    def _ts_to_datetime(self, ts: int) -> datetime | None:
        """Convert sort key timestamp to datetime."""
        if ts <= 0:
            return None
        year = ts // 100
        month = ts % 100
        try:
            return datetime(year, month, 1)
        except ValueError:
            return None

    def _fmt_date(self, entry: ParsedDate) -> str:
        """Format a date entry as a string."""
        start = self._fmt_date_part(entry.start_year, entry.start_month)
        if entry.is_current:
            return f"{start} - Present"
        end = self._fmt_date_part(entry.end_year, entry.end_month)
        return f"{start} - {end}"

    def _fmt_date_part(self, year: int | None, month: int | None) -> str:
        """Format a year/month pair."""
        if not year:
            return "?"
        if month:
            for name, num in MONTH_MAP.items():
                if num == month and len(name) > 3:
                    return f"{name.title()} {year}"
        return str(year)


TAILOR_SYSTEM_PROMPT = (
    "You are an expert resume writer. Tailor the candidate's resume to better "
    "match a specific job posting. Return ONLY the tailored resume text — no "
    "explanation, no thinking, no notes. Start directly with the name.\n\n"
    "DECISION FRAMEWORK — how to handle each section:\n\n"
    "1. CONTACT INFO — preserve exactly as-is. Do not reformat or reorder.\n\n"
    "2. PROFESSIONAL SUMMARY — rewrite entirely. This is the most important "
    "section to tailor. Mirror the job posting's language, but state ONLY facts "
    "already in the candidate's résumé — introduce no new claim. NEVER describe "
    "the candidate with a credential/status they do not hold (no 'accredited', "
    "'certified', 'licensed', 'qualified', 'expert'), and NEVER state or imply "
    "they have worked in the job's environment, tech stack, or domain (e.g. do "
    "not write 'within a cloud-native environment' unless the résumé says so) — "
    "frame the target role as one they SEEK or are TRANSITIONING to. Mention the "
    "candidate's years of experience, key relevant skills from their actual "
    "skill list, and the type of role they seek. 2-3 sentences. "
    "Write in THIRD PERSON — never use 'I', 'my', or 'me'. "
    "Example: 'Experienced IT professional seeking to leverage...' not "
    "'I am seeking to leverage...'\n\n"
    "3. SKILLS — preserve the exact skill names from the original. Do NOT add "
    "new skills. You may reorder to put the most job-relevant skills first. "
    "Do NOT rename or paraphrase skill names — use them verbatim.\n\n"
    "4. EXPERIENCE — enhance each bullet point using these techniques:\n"
    "   - Start with a strong action verb (Resolved, Delivered, Managed, "
    "Achieved, Implemented)\n"
    "   - Vary your action verbs — do NOT use the same verb twice, and do NOT "
    "use ornate/power verbs (e.g., 'orchestrated', 'spearheaded', 'facilitated') "
    "in every bullet. Use them sparingly (2-3 per job max). Prefer clear verbs "
    "like 'managed', 'resolved', 'delivered', 'supported', 'handled'.\n"
    "   - Add context: mention tools/systems from the candidate's actual "
    "skills where they naturally apply\n"
    "   - Quantify outcomes where the original provides numbers (e.g., 95% "
    "success rate)\n"
    "   - Mirror job posting terminology where truthful (e.g., if the job "
    "says 'troubleshoot hardware and software', use that phrase if the "
    "candidate did similar work)\n"
    "   - Do NOT invent metrics, responsibilities, outcomes, or technologies "
    "not in the original. Never add a number, percentage, or absolute "
    "quantifier ('100%', 'all', 'every', 'fully') the original does not state "
    "— keep the original's figures (e.g. '95%') but introduce NO new ones, and "
    "do not append an invented result ('improving X', 'ensuring seamless Y') "
    "the original does not claim. Rephrasing is fine; inflating is fabrication.\n"
    "   - Include ALL jobs from the original. Do not drop any.\n"
    "   - Write 3-5 bullets per job\n\n"
    "5. EDUCATION — include ALL education entries from the original. "
    "Preserve institution names, degrees, course names, and dates exactly. "
    "If the original uses a combined header such as 'Education & Certifications', "
    "preserve that combined header. Otherwise keep Education and Certifications as "
    "separate sections. Include EVERY entry even if it overlaps with Certifications "
    "(e.g., a 'Cert Preparation' course is Education, not a held Certification). "
    "If original has no education section, do not add one. "
    "Order entries REVERSE-CHRONOLOGICAL (most recent first).\n\n"
    "6. CERTIFICATIONS — include if present. Preserve names and status "
    "(e.g., 'In progress') exactly.\n\n"
    "7. LANGUAGES — include if present. Preserve language names and "
    "proficiency levels exactly.\n\n"
    "8. VOLUNTEER — include if present. Enhance bullets like experience "
    "section (action verbs, relevance framing). Can condense to 1-2 bullets "
    "if the role is not directly relevant to the job.\n\n"
    "9. REFERENCES — include if present. Single line is fine.\n\n"
    "TONE — when a TONE directive is provided in the user message:\n"
    "- Use the specified action verbs in experience bullets\n"
    "- Emphasize the listed themes in the summary and experience\n"
    "- Avoid the listed patterns in all sections\n"
    "- Mirror the job posting's vocabulary and sentence structure\n"
    "- If no tone directive is provided, use clear professional language\n\n"
    "FORMATTING:\n"
    "- **Bold** for section headers and job titles/company names\n"
    "- *Italics* for dates\n"
    "- Bullet points (•) for skills and experience items\n"
    "- REQUIRED STRUCTURE — the output MUST contain bold section headers on their own lines. "
    "Use the language requested by the user message for section headers. In English, use "
    "**Skills**, **Experience**, **Education** or **Education & Certifications**, "
    "**Certifications**, and **Languages**. In French, use **Compétences**, **Expérience**, "
    "**Formation** or **Formation et certifications**, **Certifications**, and **Langues**. "
    "Do not omit these headers. Place the skills header directly before the skills list and "
    "the experience header directly before the first job.\n\n"
    "ABSOLUTE RULES:\n"
    "- NEVER add skills, tools, or technologies not in the original resume\n"
    "- NEVER invent experience, metrics, or responsibilities\n"
    "- NEVER add education, certifications, or credentials not in original\n"
    "- NEVER describe the candidate with a credential or status word not in the "
    "original (e.g. 'accredited', 'certified', 'licensed', 'chartered') — it "
    "claims a qualification they may not hold\n"
    "- NEVER present the job posting's environment, tech stack, tools, or domain "
    "as the candidate's own experience — claim only what the original states; the "
    "candidate may 'seek' or 'transition to' the role, not have worked in it\n"
    "- NEVER remove or shorten job titles — preserve the COMPLETE original "
    "title including all qualifiers (e.g., 'Claims Specialist Dental & Medical' "
    "must stay as-is, not shortened to 'Claims Specialist')\n"
    "- If a job requirement is missing from the candidate's skills, do NOT "
    "add it — instead emphasize related skills they DO have\n\n"
    "EXAMPLES — before and after tailoring:\n\n"
    "BEFORE summary:\n"
    "'IT professional with experience in technical support and customer service.'\n\n"
    "AFTER summary (for a Help Desk Analyst job):\n"
    "'Experienced IT support professional with 5+ years resolving hardware, "
    "software, and network issues. Skilled in ServiceNow, Active Directory, "
    "and Windows environments. Seeking to leverage technical troubleshooting "
    "expertise as a Help Desk Analyst.'\n\n"
    "BEFORE bullet:\n"
    "'• Helped customers with technical issues'\n\n"
    "AFTER bullet:\n"
    "'• Resolved 40+ daily technical support tickets across Windows, macOS, "
    "and mobile platforms, maintaining a 95% first-call resolution rate'\n\n"
    "HONESTY — do NOT overclaim:\n"
    "BAD summary (invents a credential AND claims the job's environment as "
    "experience):\n"
    "'Accredited security professional with deep hands-on experience within a "
    "cloud-native, Mac-first environment.'\n"
    "GOOD summary (reframes the candidate's REAL background, seeks the role):\n"
    "'Operations professional with 10+ years of triage and escalation "
    "experience, now transitioning into security operations through hands-on "
    "cybersecurity training. Seeking to apply this foundation as a Security "
    "Analyst.'"
)

TAILOR_PROMPT_TEMPLATE = (
    "Tailor this resume for the following job:\n\n"
    "Job Title: {job_title}\n"
    "Company: {job_company}\n"
    "Location: {job_location}\n"
    "Description: {job_description}\n"
    "Requirements: {requirements}\n\n"
    "Current Resume:\n---\n{resume_text}\n---\n\n"
    "Candidate's verbatim skills (preserve these exactly, reorder only):\n"
    "{skills}\n\n"
    "Education source section that MUST appear in the output. Preserve the source "
    "header if it is combined, e.g. 'Education & Certifications':\n"
    "{education_entries}\n\n"
    "{tone_section}\n\n"
    "{user_instructions}\n\n"
    "Return the complete tailored resume text."
)

CHANGES_PROMPT_TEMPLATE = (
    "Given the original and tailored resume below, provide a concise "
    "bullet-point summary of what changed and why.\n\n"
    "Original (first 500 chars):\n---\n{original_preview}\n---\n\n"
    "Tailored (first 500 chars):\n---\n{tailored_preview}\n---\n\n"
    "Return ONLY 3-5 bullet points describing the CONCRETE changes (what was "
    "rephrased, reordered, or emphasized). Do NOT add assurances about honesty "
    "or accuracy (e.g. 'no information was invented', 'nothing was fabricated') "
    "— the tool does not guarantee that and the user reviews the result; only "
    "describe the changes. No thinking process, no explanation, just the bullets."
)

# Credential/status words a tailored summary must not claim unless the candidate already holds
# them. The tailor LLM (a 4B) will occasionally promote the candidate with a credential they lack
# ("Accredited security professional"); the section-level guards don't cover the freeform summary,
# so `_strip_unearned_credentials` is the deterministic backstop. Human review remains the final
# check — this catches the clearest class, not every possible overclaim.
_CREDENTIAL_TERMS = (
    "accredited",
    "certified",
    "licensed",
    "chartered",
    "credentialed",
    "accrédité",
    "accréditée",
    "accredite",
    "accreditee",
    "certifié",
    "certifiée",
    "certifie",
    "certifiee",
    "agréé",
    "agréée",
    "agree",
    "agreee",
    "licencié",
    "licenciée",
    "licencie",
    "licenciee",
)
# Summary-type headers that stay INSIDE the credential-scrub region. The scrub region is the
# LEADING block — top of the résumé down to the first NON-summary section header — so a labelled
# summary (under any of these) AND a header-less leading paragraph are both covered (a 4B that
# overclaims often omits/renames the header). Skills/experience/etc. end the region.
_SUMMARY_SYNONYMS = frozenset(
    {
        "summary",
        "professional summary",
        "career summary",
        "profile",
        "profil",
        "professional profile",
        "objective",
        "career objective",
        "about",
        "about me",
        "summary of qualifications",
    }
)
_CREDENTIAL_HEADER_RE = re.compile(
    r"^\*{0,2}\s*"
    r"(?:certifications?|licen[sc]es?|licences?|credentials?|accreditations?|accréditations?)"
    r"\s*\*{0,2}$",
    re.IGNORECASE,
)
_SKILLS_HEADER_RE = re.compile(
    r"^\*{0,2}\s*(?:(?:technical|core|key|professional|relevant|soft)\s+)?"
    r"(?:skills|competencies|proficiencies|compétences|competences)\s*\*{0,2}$",
    re.IGNORECASE,
)
_SUMMARY_SOURCE_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "incident resolution",
        "high-stakes client problem-solving",
        "high-stakes client problem-solving",
    ),
    ("process improvement", "operations management", "operations management"),
    ("process optimization", "operations management", "operations management"),
    ("client support", "client-facing work", "client-facing"),
    (
        "technical troubleshooting experience",
        "front-line technical support experience",
        "technical support",
    ),
    ("stakeholder coordination", "triage and escalation", "triage"),
    ("hands-on technical training and coursework", "cybersecurity coursework", "coursework"),
    ("hands-on technical training", "cybersecurity coursework", "coursework"),
    ("hands-on cybersecurity training", "cybersecurity coursework", "coursework"),
)


def _is_summary_header(stripped: str) -> bool:
    """Whether *stripped* is a summary-type header (markdown/ATX/colon-tolerant). These stay inside
    the credential-scrub leading region, unlike skills/experience/etc. which end it."""
    cleaned = re.sub(r"^[#*\s]+", "", stripped)
    cleaned = re.sub(r"[*\s:]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned in _SUMMARY_SYNONYMS


def _skills_match(skill_lower: str, orig: str) -> bool:
    """Check if a tailored skill matches an original skill.

    Uses exact match, token containment, and fuzzy matching.
    """
    from difflib import SequenceMatcher

    if skill_lower == orig:
        return True
    skill_tokens = set(skill_lower.split())
    orig_tokens = set(orig.split())
    shorter, longer = (
        (skill_tokens, orig_tokens)
        if len(skill_tokens) <= len(orig_tokens)
        else (orig_tokens, skill_tokens)
    )
    if shorter and shorter.issubset(longer):
        return True
    ratio = SequenceMatcher(None, skill_lower, orig).ratio()
    return ratio >= 0.85


class ResumeTailor:
    """Tailor resumes for specific job listings using LLM."""

    def __init__(self, config: LLMConfig, runtime: LLMRuntime | None = None) -> None:
        self._config = config
        # Shared per-command breaker (passed by the CLI; spans tailoring + cover-letter
        # so the app's largest LLM calls are finally breaker-protected).
        self._runtime = runtime or LLMRuntime.defaults(name="resume-tailor")
        self._breaker = self._runtime.breaker
        # Its OWN runtime/breaker (NOT this tailor's): a flaky verifier endpoint must never trip the
        # circuit that guards real tailoring (#4 fail-safe — a verifier problem never blocks
        # generation). Used ONLY by tailor_verified() — tailor() stays the pure primitive.
        self._verifier = GroundingVerifier(config)

    @async_retry(
        max_attempts=2, base_delay=1.0, exceptions=(LLMError,), exclude=(CircuitOpenError,)
    )
    async def tailor(
        self,
        resume: ResumeData,
        job: JobListing,
        user_instructions: str = "",
        style_guide: StyleGuide | None = None,
        tone_profile: ToneProfile | None = None,
        matcher: JobMatcher | None = None,
        match_result: MatchResult | None = None,
    ) -> TailoredResume:
        """Tailor a resume for a specific job.

        Args:
            resume: Original parsed resume
            job: Target job listing
            user_instructions: Optional user guidance for tailoring
            style_guide: Optional style guide to apply
            tone_profile: Optional pre-detected ToneProfile to avoid re-detection
            matcher: Optional configured JobMatcher instance to reuse
            match_result: Optional already-computed fit score to avoid duplicate matching

        Returns:
            TailoredResume with full metadata
        """
        if match_result is None:
            if matcher is None:
                raise ConfigError(
                    "Resume tailoring requires a configured JobMatcher or precomputed "
                    "MatchResult. Pass the command/TUI/batch matcher so embeddings use the "
                    "configured device instead of constructing a hidden fallback matcher."
                )
            match_result = await matcher.match_resume_to_job(resume, job)
        elif match_result.job != job:
            logger.debug(
                "Using caller-provided MatchResult for %s at %s; job object differs from "
                "the tailoring target.",
                match_result.job.title,
                match_result.job.company,
            )

        logger.info("Current match: %.0f%%", match_result.score * 100)

        instruction_section = ""
        if user_instructions:
            instruction_section = f"Additional instructions from user:\n{user_instructions}"
        else:
            instruction_section = "No additional instructions."

        edu_entries = self._extract_education_entries(resume.raw_text)

        from job_applicator.documents.tone_detector import ToneDetector

        if tone_profile is None:
            tone_detector = ToneDetector()
            tone_profile = tone_detector.detect(
                title=job.title,
                description=job.description,
                requirements=job.requirements,
            )
        tone_section = ToneDetector().format_for_prompt(tone_profile)

        prompt = TAILOR_PROMPT_TEMPLATE.format(
            job_title=job.title,
            job_company=job.company,
            job_location=job.location,
            job_description=job.description[:800],
            requirements=", ".join(job.requirements),
            resume_text=resume.raw_text[:5000],
            skills=", ".join(resume.skills),
            education_entries=edu_entries,
            tone_section=tone_section,
            user_instructions=instruction_section,
        )

        if style_guide:
            style_section = StyleAnalyzer.format_style_for_prompt(style_guide)
            prompt += f"\n\n{style_section}"

        # Output language: the CV must match the cover letter (both resolve from [llm] language +
        # the same job), so one application never mixes languages. Translate the PROSE (summary,
        # bullets, section headings); keep skill names, company/school names, course names, and
        # certifications verbatim (a French Québec CV keeps English technical terms).
        language = resolve_output_language(self._config.language, job.description)
        logger.info(
            "Tailoring résumé in %s (language setting=%s) for %s",
            language,
            self._config.language,
            job.title,
        )
        prompt += (
            f"\n\nIMPORTANT: Write the ENTIRE tailored résumé in {language} — every section "
            f"heading, the summary, and every bullet. Translate the prose, but keep skill names, "
            f"job titles, company and school names, course names, certifications, and technical "
            f"tools/terms VERBATIM (do not translate source-owned titles, proper nouns, or "
            f"technical terms). Keep skills/coursework as concise source-backed lists, not one "
            f"long translated prose claim."
        )

        tailored_text = await self._call_llm(prompt)
        tailored_text = self._require_nonempty(tailored_text)
        tailored_text = self._validate_skills(tailored_text, resume.skills)
        tailored_text = self._strip_hallucinated_tools(
            tailored_text, resume.raw_text, job.requirements
        )
        tailored_text = self._strip_malformed_tool_removal_sentences(tailored_text)
        tailored_text = self._ground_summary_phrases(tailored_text, resume.raw_text)
        tailored_text = self._ensure_source_backed_summary(tailored_text, resume.raw_text, language)
        tailored_text = self._strip_hallucinated_education(tailored_text, resume.raw_text)
        tailored_text = self._strip_unbacked_optional_sections(tailored_text, resume.raw_text)
        tailored_text = self._strip_unearned_credentials(tailored_text, resume.raw_text)
        tailored_text = self._strip_unsupported_metric_claims(tailored_text, resume.raw_text)
        tailored_text = self._strip_unverifiable_aspirations(tailored_text)
        tailored_text = self._strip_unbacked_responsibility_bullets(tailored_text, resume.raw_text)
        tailored_text = self._strip_misplaced_support_domain_bullets(
            tailored_text,
            resume.raw_text,
        )
        tailored_text = self._strip_low_evidence_bullets(tailored_text, resume.raw_text)
        tailored_text = self._normalize_date_range_dashes(tailored_text)
        tailored_text = self._strip_unbacked_references(tailored_text, resume.raw_text)
        tailored_text = self._preserve_source_required_sections(
            tailored_text, resume.raw_text, language
        )
        tailored_text = self._localize_standard_labels_for_language(tailored_text, language)
        tailored_text = self._polish_french_output(tailored_text, language)
        tailored_text = self._strip_duplicate_bullets(tailored_text)
        tailored_text = self._strip_non_source_bold_entry_headings(tailored_text, resume.raw_text)
        changes = await self._summarize_changes(resume.raw_text, tailored_text)

        return TailoredResume(
            original_path="",
            tailored_text=tailored_text,
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            match_score=match_result.score,
            semantic_score=match_result.semantic_score,
            skill_score=match_result.skill_score,
            matched_skills=match_result.matched_skills,
            missing_skills=match_result.missing_skills,
            changes_summary=changes,
            user_modifications=user_instructions,
        )

    async def tailor_verified(
        self,
        resume: ResumeData,
        job: JobListing,
        user_instructions: str = "",
        style_guide: StyleGuide | None = None,
        tone_profile: ToneProfile | None = None,
        matcher: JobMatcher | None = None,
        match_result: MatchResult | None = None,
    ) -> TailoredResume:
        """``tailor`` plus a language-agnostic grounding pass (spec §6).

        The tailored text is verified against the BASE résumé and the report is ATTACHED to the
        result for human review — claims are NEVER auto-stripped. This is the document of record:
        the user is its ground truth, and the verifier has a measured precision residual, so a flag
        is a question to the user, not a deletion. Non-blocking: a verifier failure is the fail-safe
        (#4) — the tailored résumé is returned with ``grounding_report=None``, never blocked and
        never reported as verified.
        """
        result = await self.tailor(
            resume, job, user_instructions, style_guide, tone_profile, matcher, match_result
        )
        return await self.verify_tailored(result, resume)

    async def verify_tailored(self, result: TailoredResume, resume: ResumeData) -> TailoredResume:
        """Attach the grounding report (tailored text vs the BASE résumé) for human review (spec
        §6). Non-blocking fail-safe (#4): a verifier failure leaves ``grounding_report=None`` and
        never blocks. Reusable so an interactive refine can be re-verified the same way."""
        try:
            result.grounding_report = await self._verifier.verify(result.tailored_text, resume)
        except GroundingUnavailableError as exc:
            logger.info("Résumé grounding verification skipped (verifier unavailable): %s", exc)
        return result

    async def refine_verified(
        self,
        original_resume: ResumeData,
        current_tailored: TailoredResume,
        user_feedback: str,
        job: JobListing,
        matcher: JobMatcher | None = None,
        tone_profile: ToneProfile | None = None,
        style_guide: StyleGuide | None = None,
    ) -> TailoredResume:
        """``refine`` plus the grounding pass (spec §6), so an interactively refined version carries
        the same honesty report as the primary — ATTACHED for human review, never auto-stripped.
        Non-blocking fail-safe (#4), like ``tailor_verified``."""
        result = await self.refine(
            original_resume,
            current_tailored,
            user_feedback,
            job,
            matcher,
            tone_profile,
            style_guide,
        )
        return await self.verify_tailored(result, original_resume)

    @async_retry(
        max_attempts=2, base_delay=1.0, exceptions=(LLMError,), exclude=(CircuitOpenError,)
    )
    async def refine(
        self,
        original_resume: ResumeData,
        current_tailored: TailoredResume,
        user_feedback: str,
        job: JobListing,
        matcher: JobMatcher | None = None,
        tone_profile: ToneProfile | None = None,
        style_guide: StyleGuide | None = None,
    ) -> TailoredResume:
        """Refine a tailored resume based on user feedback.

        Args:
            original_resume: The original resume
            current_tailored: The current tailored version
            user_feedback: User's feedback/instructions
            job: Target job listing
            matcher: Optional JobMatcher instance to reuse
            tone_profile: Optional tone profile to maintain across refinements
            style_guide: Optional style guide to maintain across refinements

        Returns:
            New TailoredResume with refinements applied
        """
        tone_directive = ""
        if tone_profile:
            from job_applicator.documents.tone_detector import ToneDetector

            tone_directive = (
                f"\n\n{ToneDetector().format_for_prompt(tone_profile)}\n"
                "Maintain this tone throughout the refined resume."
            )

        style_section = ""
        if style_guide:
            style_section = (
                f"\n\n{StyleAnalyzer.format_style_for_prompt(style_guide)}\n"
                "Maintain this writing style throughout the refined resume."
            )

        language = resolve_output_language(self._config.language, job.description)
        prompt = (
            f"The user wants changes to this tailored resume.\n\n"
            f"Job: {job.title} at {job.company}\n"
            f"Requirements: {', '.join(job.requirements)}\n\n"
            f"Candidate's ACTUAL skills (ONLY use these):\n"
            f"{', '.join(original_resume.skills)}\n\n"
            f"Current tailored resume:\n---\n"
            f"{current_tailored.tailored_text[:5000]}\n---\n\n"
            f"User feedback:\n{user_feedback}\n\n"
            f"Apply the user's feedback while keeping the resume tailored "
            f"for the job. Do NOT add skills not in the candidate's actual "
            f"skills list. Do NOT add Education if none exists in original. Write the refined "
            f"résumé in {language}. Translate prose only; keep source-owned job titles, company "
            f"and school names, course names, certifications, and technical tools/terms verbatim. "
            f"Keep skills/coursework as concise source-backed lists, not one long translated "
            f"prose claim."
            f"{tone_directive}"
            f"{style_section}\n"
            f"Return the complete updated resume text."
        )

        refined_text = await self._call_llm(prompt, temperature=0.3)
        refined_text = self._validate_skills(refined_text, original_resume.skills)
        refined_text = self._strip_hallucinated_tools(
            refined_text, original_resume.raw_text, job.requirements
        )
        refined_text = self._strip_malformed_tool_removal_sentences(refined_text)
        refined_text = self._ground_summary_phrases(refined_text, original_resume.raw_text)
        refined_text = self._ensure_source_backed_summary(
            refined_text, original_resume.raw_text, language
        )
        refined_text = self._strip_hallucinated_education(refined_text, original_resume.raw_text)
        refined_text = self._strip_unbacked_optional_sections(
            refined_text, original_resume.raw_text
        )
        refined_text = self._strip_unearned_credentials(refined_text, original_resume.raw_text)
        refined_text = self._strip_unsupported_metric_claims(refined_text, original_resume.raw_text)
        refined_text = self._strip_unverifiable_aspirations(refined_text)
        refined_text = self._strip_unbacked_responsibility_bullets(
            refined_text, original_resume.raw_text
        )
        refined_text = self._strip_misplaced_support_domain_bullets(
            refined_text,
            original_resume.raw_text,
        )
        refined_text = self._strip_low_evidence_bullets(refined_text, original_resume.raw_text)
        refined_text = self._normalize_date_range_dashes(refined_text)
        refined_text = self._strip_unbacked_references(refined_text, original_resume.raw_text)
        refined_text = self._preserve_source_required_sections(
            refined_text, original_resume.raw_text, language
        )
        refined_text = self._localize_standard_labels_for_language(refined_text, language)
        refined_text = self._polish_french_output(refined_text, language)
        refined_text = self._strip_duplicate_bullets(refined_text)
        refined_text = self._strip_non_source_bold_entry_headings(
            refined_text, original_resume.raw_text
        )
        refined_text = self._require_nonempty(refined_text)
        changes = await self._summarize_changes(current_tailored.tailored_text, refined_text)

        # Recompute match scores against the refined text
        if matcher is None:
            raise ConfigError(
                "Résumé refinement requires a configured JobMatcher. Pass the command/TUI/batch "
                "matcher so embeddings use the configured device instead of constructing a "
                "hidden fallback matcher."
            )

        synthetic_resume = ResumeData(
            raw_text=refined_text,
            name=original_resume.name,
            email=original_resume.email,
            phone=original_resume.phone,
            summary=original_resume.summary,
            skills=original_resume.skills,
            experience=original_resume.experience,
            education=original_resume.education,
        )
        new_match = await matcher.match_resume_to_job(synthetic_resume, job)

        return TailoredResume(
            original_path=current_tailored.original_path,
            tailored_text=refined_text,
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            match_score=new_match.score,
            semantic_score=new_match.semantic_score,
            skill_score=new_match.skill_score,
            matched_skills=new_match.matched_skills,
            missing_skills=new_match.missing_skills,
            changes_summary=changes,
            user_modifications=user_feedback,
            attempt=current_tailored.attempt + 1,
        )

    def _extract_education_entries(self, text: str) -> str:
        """Extract the source Education section for the prompt.

        Uses the shared section-header parser, so compound v1 headers like
        ``EDUCATION & CERTIFICATIONS`` are treated as Education instead of being missed.
        """
        source = _source_section(text, "Education")
        if source is None:
            return "None — do not add an Education section."

        header, body = source
        normalized_header = re.sub(r"\s+", " ", header.strip().strip("*").strip()).casefold()
        if normalized_header == "education":
            entries: list[str] = []
            current_entry: list[str] = []

            for line in body:
                stripped = line.strip()
                if not stripped:
                    continue
                is_date = bool(
                    re.search(r"\d{4}", stripped) and re.search(r"[-\u2013\u2014]", stripped)
                )
                if is_date:
                    current_entry.append(stripped)
                    entries.append(" ".join(current_entry).strip())
                    current_entry = []
                elif current_entry:
                    current_entry.append(stripped)
                else:
                    current_entry = [stripped]

            if current_entry:
                entries.append(" ".join(current_entry).strip())
            if not entries:
                return "None — do not add an Education section."
            return "\n".join(f"  {index}. {entry}" for index, entry in enumerate(entries, 1))

        return "\n".join([header, *(f"  {line.strip()}" for line in body if line.strip())])

    def _require_nonempty(self, text: str) -> str:
        """Guard against an empty tailored résumé reaching the caller (and then cover-letter
        generation + PDF rendering). Raises LLMError so @async_retry retries a transient empty
        completion before failing with a typed error (mirrors cover_letter._validate_output)."""
        if not text.strip():
            raise LLMError("Tailored résumé is empty")
        return text

    @staticmethod
    def _section_body(text: str, header_re: re.Pattern[str]) -> str:
        """Return the lines under the first header matching ``header_re``, up to the next
        section header (markdown-bold-aware)."""
        out: list[str] = []
        in_section = False
        for line in text.split("\n"):
            stripped = line.strip()
            if header_re.match(stripped):
                in_section = True
                continue
            if in_section and _looks_like_section_header(stripped):
                break
            if in_section:
                out.append(line)
        return "\n".join(out)

    @staticmethod
    def _remove_credential_words(line: str, terms: list[str]) -> str:
        """Strip credential words (plus an adjacent connector) from a line and repair the prose so
        the sentence stays readable (e.g. 'Accredited security pro' → 'Security pro'). Returns a
        line UNTOUCHED if it holds none of the terms (so we never reflow lines we didn't change);
        grammar repair is best-effort — rare residue (a/an agreement) is left for human review."""
        cleaned = line
        removed = False
        for term in terms:
            new = re.sub(rf"(?i)\b{re.escape(term)}\b(?:\s*,\s*|\s+and\s+|\s+)?", "", cleaned)
            if new != cleaned:
                removed = True
                cleaned = new
        if not removed:
            return line
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r",\s*,", ", ", cleaned)  # commas doubled by a removal
        cleaned = re.sub(r"\s+,", ",", cleaned)  # a space left before a comma
        cleaned = re.sub(r"^\s*(?:and|,)\s+", "", cleaned, flags=re.IGNORECASE)  # leading connector
        cleaned = cleaned.strip()
        if not re.search(r"[A-Za-z0-9]", cleaned):  # nothing but punctuation left → drop the line
            return ""
        if cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    @staticmethod
    def _lead_region(text: str) -> str:
        """The leading block (name/contact/summary) up to the first NON-summary section header."""
        out: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if _looks_like_section_header(stripped) and not _is_summary_header(stripped):
                break
            out.append(line)
        return "\n".join(out)

    def _strip_unearned_credentials(self, tailored: str, original: str) -> str:
        """Remove credential/status claims (e.g. 'accredited', 'certified') from the tailored
        summary that the candidate does not hold in the original.

        The scrub region is the LEADING block — top of the résumé to the first NON-summary section
        header — so it covers a labelled summary under ANY header AND a header-less leading
        paragraph (the layout a 4B that just overclaimed often emits). A term survives only if the
        ORIGINAL already claims it in a credential-plausible context (its leading/summary block,
        certifications, or skills section) — NOT a benign 'certified' verb in an experience bullet.
        Deterministic backstop for the freeform summary the section-level guards skip; HUMAN REVIEW
        remains the final check (this catches the clearest class, not every possible overclaim)."""
        earned_ctx = (
            self._lead_region(original)
            + "\n"
            + self._section_body(original, _CREDENTIAL_HEADER_RE)
            + "\n"
            + self._section_body(original, _SKILLS_HEADER_RE)
        ).lower()
        unearned = [t for t in _CREDENTIAL_TERMS if t not in earned_ctx]
        if not unearned:
            return tailored

        out: list[str] = []
        in_lead = True  # the leading region (name/contact/summary) before the first real section
        for line in tailored.split("\n"):
            stripped = line.strip()
            if (
                in_lead
                and _looks_like_section_header(stripped)
                and not _is_summary_header(stripped)
            ):
                in_lead = False  # reached skills/experience/etc. — past the summary
            if in_lead and stripped and not _is_summary_header(stripped):
                line = self._remove_credential_words(line, unearned)
            out.append(line)
        return "\n".join(out)

    def _validate_skills(self, text: str, original_skills: list[str]) -> str:
        """Strip skills from tailored text that aren't in the original resume.

        Parses the skills section, compares each skill against the original
        list using fuzzy matching, and removes hallucinated skills. Skill-name
        normalization lets "Python 3" match an original "Python" entry, while
        the hard-negative list drops generic traits such as "team player".
        """
        import re

        from job_applicator.skills import is_hard_negative, normalize_skill

        if not original_skills:
            return text

        # Normalize original skills for matching
        norm_skills = [normalize_skill(s).lower() for s in original_skills if s.strip()]

        lines = text.split("\n")
        result_lines = []
        in_skills_section = False

        for _i, line in enumerate(lines):
            stripped = line.strip()

            if section_header(stripped) == "Skills":
                in_skills_section = True
                result_lines.append(line)
                continue

            # Detect end of skills section (next section header)
            if in_skills_section and _looks_like_section_header(stripped):
                in_skills_section = False
                result_lines.append(line)
                continue

            # If in skills section, validate each skill line
            if in_skills_section and stripped:
                # Extract the skill text (remove bullets, markdown)
                bullet_match = re.match(r"^([\*\-\•\·\s]*)(.*)$", stripped)
                prefix = bullet_match.group(1) if bullet_match else ""
                skill_text = bullet_match.group(2).strip() if bullet_match else stripped

                if not skill_text or len(skill_text) < 3:
                    result_lines.append(line)
                    continue

                # Some LLMs emit the whole skills section as one comma-separated
                # line ("Python, FastAPI, ..."). Split on commas, validate each
                # token individually, and rebuild the line so valid skills are kept.
                # Ampersand / "and" are left intact because lines like
                # "Docker & Kubernetes" still match via token containment when one
                # of the skills is present in the original résumé.
                raw_tokens = _split_csv_outside_parentheses(skill_text)

                kept_tokens: list[str] = []
                for token in raw_tokens:
                    if len(token) < 3:
                        continue
                    if token.count("(") != token.count(")"):
                        logger.info("Stripped malformed skill token: %s", token)
                        continue
                    token_norm = normalize_skill(token).lower()

                    # Drop generic traits unless they were literally in the original resume.
                    if is_hard_negative(token_norm) and token_norm not in norm_skills:
                        logger.info("Stripped hard-negative skill: %s", token)
                        continue

                    if any(_skills_match(token_norm, orig) for orig in norm_skills):
                        kept_tokens.append(token)
                    else:
                        logger.info("Stripped hallucinated skill: %s", token)

                if kept_tokens:
                    result_lines.append(prefix + ", ".join(kept_tokens))
                continue
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

    def _strip_hallucinated_tools(
        self,
        tailored: str,
        original_text: str,
        job_requirements: list[str],
    ) -> str:
        """Remove tool/technology names from tailored text that aren't in
        the original resume.

        Scans for job requirement terms (like ServiceNow, Jira, etc.) that
        appear in the tailored output but NOT in the original resume, and
        replaces them with generic equivalents.
        """
        import re

        original_lower = original_text.lower()

        # Common tool replacements when hallucinated
        tool_replacements = {
            "servicenow": "ticketing systems",
            "jira": "project management tools",
            "confluence": "documentation platforms",
            "slack": "communication platforms",
            "zendesk": "support platforms",
            "freshdesk": "support platforms",
            "salesforce": "CRM systems",
            "hubspot": "CRM systems",
            "pagerduty": "incident management tools",
            "datadog": "monitoring tools",
            "splunk": "log analysis tools",
            "aws": "cloud platforms",
            "azure": "cloud platforms",
            "gcp": "cloud platforms",
        }

        result = tailored

        for req in job_requirements:
            req_lower = req.lower().strip()
            if not req_lower or len(req_lower) < 3:
                continue

            # Check if this tool/requirement is in the original resume.
            # Word-boundary match so "Go" doesn't count as present in "Good".
            if _alnum_boundary_pattern(req_lower).search(original_lower):
                continue  # Original has it, keep it

            # Check if it appears in the tailored text. Word-boundary match so
            # "Java" doesn't corrupt "JavaScript".
            pattern = _alnum_boundary_pattern(req)
            if pattern.search(result):
                # Hallucinated tool — replace with generic if available
                replacement = tool_replacements.get(req_lower)
                if replacement:
                    # Check if generic term already appears nearby
                    # to avoid "ticketing systems like ticketing systems"
                    result_lower = result.lower()
                    req_pos = result_lower.find(req_lower)
                    if req_pos >= 0:
                        context = result_lower[max(0, req_pos - 50) : req_pos + len(req_lower) + 50]
                        if replacement.lower() in context:
                            # Generic already nearby — just remove the tool
                            result = pattern.sub("", result)
                            logger.info(
                                "Removed hallucinated tool '%s' (generic '%s' already in context)",
                                req,
                                replacement,
                            )
                        else:
                            result = pattern.sub(replacement, result)
                            logger.info(
                                "Replaced hallucinated tool '%s' with '%s'",
                                req,
                                replacement,
                            )
                    else:
                        result = pattern.sub(replacement, result)
                        logger.info(
                            "Replaced hallucinated tool '%s' with '%s'",
                            req,
                            replacement,
                        )
                else:
                    # No generic replacement — just remove it
                    result = pattern.sub("", result)
                    logger.info("Removed hallucinated tool '%s'", req)

        # Pass 2: check tool_replacements keys against tailored vs original
        req_lower_set = {r.lower().strip() for r in job_requirements}
        for tool_key, replacement in tool_replacements.items():
            # Alphanumeric boundaries (consistent with Pass 1) so a short key
            # like "aws" isn't seen as present in "draws" or matched inside
            # another word in the tailored text.
            if _alnum_boundary_pattern(tool_key).search(original_lower):
                continue
            if tool_key in req_lower_set:
                continue
            tool_pattern = _alnum_boundary_pattern(tool_key)
            match = tool_pattern.search(result)
            if match:
                tool_pos = match.start()
                context = result.lower()[max(0, tool_pos - 50) : tool_pos + len(tool_key) + 50]
                if replacement.lower() in context:
                    result = tool_pattern.sub("", result)
                else:
                    result = tool_pattern.sub(replacement, result)
                logger.info("Pass2: replaced hallucinated tool '%s'", tool_key)

        # Clean up double spaces and broken phrases from removals
        result = re.sub(r"  +", " ", result)
        result = re.sub(r"\blike\s*,", ",", result)
        result = re.sub(r"\blike\s+\.", ".", result)
        result = re.sub(r"\bsuch as\s*,", ",", result)
        result = re.sub(r"\bsuch as\s+\.", ".", result)
        result = re.sub(r"\busing\s+,", ",", result)
        result = re.sub(r"\butilizing\s+,", ",", result)
        result = re.sub(r",\s*,", ",", result)
        result = re.sub(r"\s+\.", ".", result)
        return result.strip()

    @staticmethod
    def _strip_malformed_tool_removal_sentences(tailored: str) -> str:
        """Drop prose sentences broken by unsupported-tool removal.

        Replacing/removing hallucinated job tools can leave empty list slots such as ``and ,`` or
        ``in ,``. A broken sentence is worse than a shorter résumé, so drop only the malformed
        sentence and keep any clean sentence on the same line.
        """
        malformed = re.compile(
            r"(?:,\s*and\s*,|,\s*,|\bin\s*,|foundation\s+in\s*,|-\s*aligned|and\s*[.;])",
            re.IGNORECASE,
        )
        result_lines: list[str] = []
        for line in tailored.splitlines():
            stripped = line.strip()
            is_bullet = bool(re.match(r"^[•*\-]\s+", stripped))
            if not stripped or not malformed.search(line):
                result_lines.append(line)
                continue
            if is_bullet:
                logger.info("Dropped malformed bullet after tool removal: %s", stripped)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", line)
            kept = [
                sentence for sentence in sentences if sentence and not malformed.search(sentence)
            ]
            if kept:
                result_lines.append(" ".join(kept).strip())
            else:
                logger.info("Dropped malformed sentence after tool removal: %s", stripped)
        return "\n".join(result_lines).strip()

    @staticmethod
    def _ground_summary_phrases(tailored: str, original: str) -> str:
        """Rewrite high-risk generated summary phrases to source-backed wording."""
        original_lower = original.lower()
        result_lines: list[str] = []
        in_lead = True
        for line in tailored.splitlines():
            stripped = line.strip()
            if (
                in_lead
                and _looks_like_section_header(stripped)
                and not _is_summary_header(stripped)
            ):
                in_lead = False
            if in_lead and stripped and not _is_summary_header(stripped):
                line = ResumeTailor._replace_summary_phrases(line, original_lower)
            result_lines.append(line)
        return "\n".join(result_lines).strip()

    @staticmethod
    def _replace_summary_phrases(line: str, original_lower: str) -> str:
        result = line
        for generated, replacement, evidence in _SUMMARY_SOURCE_REPLACEMENTS:
            if generated in original_lower or evidence not in original_lower:
                continue
            result = re.sub(
                rf"\b{re.escape(generated)}\b",
                replacement,
                result,
                flags=re.IGNORECASE,
            )
        result = re.sub(
            r"\boperations management,\s+and\s+operations management\b",
            "operations management",
            result,
        )
        result = re.sub(
            r"\bhigh-stakes client problem-solving,\s*problem-solving,\s*and\s*"
            r"triage and escalation\b",
            "high-stakes client problem-solving, triage, and escalation",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            r"\bcybersecurity coursework\s+and\s+coursework\b",
            "cybersecurity coursework",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            r",?\s+now transitioning into IT support through\s+",
            ", with ",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(r"\s{2,}", " ", result)
        return result

    @staticmethod
    def _ensure_source_backed_summary(
        tailored: str, original: str, language: str = "English"
    ) -> str:
        """Use a conservative source-backed summary when one can be built."""
        lines = tailored.splitlines()
        bounds = _section_bounds(lines, "Summary")
        if bounds is None:
            return tailored
        start, end = bounds
        fallback = ResumeTailor._source_backed_summary(original, language)
        if not fallback:
            return tailored
        lines = [*lines[: start + 1], fallback, "", *lines[end:]]
        return ResumeTailor._join_resume_lines(lines)

    @staticmethod
    def _source_backed_summary(original: str, language: str = "English") -> str:
        source = original.lower()
        sentences: list[str] = []
        french = language == "French"
        if (
            "10+ years" in source
            and "operations management" in source
            and "high-stakes client problem-solving" in source
            and "triage" in source
            and "escalation" in source
        ):
            if french:
                sentences.append(
                    "Professionnel des opérations avec plus de 10 ans d'expérience en gestion "
                    "des opérations, résolution de problèmes clients à fort enjeu, triage et "
                    "escalade."
                )
            else:
                sentences.append(
                    "Operations professional with 10+ years of operations management, "
                    "high-stakes client problem-solving, triage, and escalation experience."
                )
        if "technical support" in source and "coursework" in source:
            if french:
                technical = (
                    "Apporte une expérience de support technique de première ligne ainsi qu'une "
                    "formation en cybersécurité et en réseautique"
                )
                if all(
                    term in source
                    for term in ("hands-on", "siem", "soc operations", "incident response")
                ):
                    technical += (
                        " avec des laboratoires pratiques en SIEM, opérations SOC et réponse "
                        "aux incidents"
                    )
            else:
                technical = (
                    "Brings front-line technical support experience plus cybersecurity and "
                    "networking coursework"
                )
                if all(
                    term in source
                    for term in ("hands-on", "siem", "soc operations", "incident response")
                ):
                    technical += (
                        " with hands-on labs in SIEM, SOC operations, and incident response"
                    )
            sentences.append(f"{technical}.")
        return " ".join(sentences)

    def _strip_unsupported_metric_claims(self, tailored: str, original: str) -> str:
        """Remove common LLM metric embellishments not present in the base résumé.

        The grounding verifier catches these after generation, but unattended ``--yes`` should not
        depend on the model obeying the prompt. This sanitizer keeps source-backed numbers, restores
        abbreviated source metrics (``2B``/``5M``), and removes invented percentage-result clauses.
        """
        original_percentages = set(re.findall(r"\b\d+(?:\.\d+)?%", original))
        result = self._restore_source_metric_abbreviations(tailored, original)
        result = self._strip_unbacked_plus_quantifiers(result, original)

        lines: list[str] = []
        for line in result.splitlines():
            percentages = set(re.findall(r"\b\d+(?:\.\d+)?%", line))
            unsupported = percentages - original_percentages
            if not unsupported:
                lines.append(line)
                continue

            cleaned = line
            for pct in sorted(unsupported, key=len, reverse=True):
                pct_pattern = re.escape(pct)
                pct_token = rf"(?<!\d){pct_pattern}(?!\d)"
                cleaned = re.sub(
                    rf"\s*,?\s*(?:improving|improved|reducing|reduced|increasing|increased|"
                    rf"decreasing|decreased|boosting|boosted|cutting|cut|saving|saved|"
                    rf"lowering|lowered|raising|raised|resulting in|leading to|achieving|"
                    rf"achieved|delivering|delivered)[^.;\n]*?{pct_token}[^.;\n]*",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                cleaned = re.sub(
                    rf"\s*\([^)]*{pct_token}[^)]*\)",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                cleaned = re.sub(
                    rf"\s+(?:by|to|at)\s+{pct_token}",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )

            if set(re.findall(r"\b\d+(?:\.\d+)?%", cleaned)) - original_percentages:
                logger.info("Dropped line with unsupported metric claim: %s", line.strip())
                continue
            cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
            cleaned = re.sub(r",\s*[.;]", ".", cleaned)
            lines.append(cleaned.rstrip())

        return "\n".join(lines).strip()

    @staticmethod
    def _restore_source_metric_abbreviations(tailored: str, original: str) -> str:
        result = tailored
        for num in re.findall(r"\b\d+(?=B\b)", original, flags=re.IGNORECASE):
            result = re.sub(rf"\b{re.escape(num)}\s+billion\b", f"{num}B", result, flags=re.I)
        for num in re.findall(r"\b\d+(?=M\b)", original, flags=re.IGNORECASE):
            result = re.sub(rf"\b{re.escape(num)}\s+million\b", f"{num}M", result, flags=re.I)
        return result

    @staticmethod
    def _strip_unbacked_plus_quantifiers(tailored: str, original: str) -> str:
        result = tailored
        for match in re.finditer(r"\b\d+\+", tailored):
            token = match.group(0)
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", original):
                continue
            result = result.replace(token, token.rstrip("+"))
        return result

    def _strip_unbacked_references(self, tailored: str, original: str) -> str:
        """Remove a References section when the base résumé did not contain one."""
        if re.search(r"\breferences\b", original, re.IGNORECASE):
            return tailored

        lines = tailored.split("\n")
        result_lines: list[str] = []
        in_references = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"^\*{0,2}\s*references\s*\*{0,2}\s*$", stripped, re.IGNORECASE):
                in_references = True
                continue
            if in_references and _looks_like_section_header(stripped):
                in_references = False
                result_lines.append(line)
                continue
            if in_references:
                continue
            result_lines.append(line)
        return "\n".join(result_lines).strip()

    def _preserve_source_required_sections(
        self, tailored: str, original: str, language: str = "English"
    ) -> str:
        """Restore source-owned sections the LLM must not rewrite materially.

        Education/certification history and language fluency are facts of record. If the model
        drops, renames, or weakens them, replace those sections with the source text so unattended
        output cannot degrade the candidate's credentials or language claims.
        """
        result = self._preserve_source_section(
            tailored,
            original,
            "Education",
            before=("Languages", "References"),
            allow_combined_certifications=True,
            language=language,
        )
        result = self._preserve_source_section(
            result,
            original,
            "Projects",
            before=("Languages", "References"),
            allow_combined_certifications=False,
            language=language,
        )
        return self._preserve_source_section(
            result,
            original,
            "Languages",
            before=("References",),
            allow_combined_certifications=False,
            language=language,
        )

    @staticmethod
    def _preserve_source_section(
        tailored: str,
        original: str,
        canonical: str,
        *,
        before: tuple[str, ...],
        allow_combined_certifications: bool,
        language: str = "English",
    ) -> str:
        source = _source_section(original, canonical)
        if source is None:
            return tailored

        source_header, source_body = source
        source_body = ResumeTailor._localize_source_section_body(
            source_body, canonical=canonical, language=language
        )
        if language == "French":
            formatted_header = ResumeTailor._localize_french_source_section_header(
                source_header, canonical=canonical
            )
            if _has_bold_section_headers(tailored):
                formatted_header = f"**{formatted_header}**"
        else:
            formatted_header = _format_source_section_header(
                source_header,
                bold=_has_bold_section_headers(tailored),
            )
        replacement = [
            formatted_header,
            *source_body,
        ]
        lines = tailored.splitlines()
        bounds = _section_bounds(lines, canonical)
        if (
            bounds is None
            and allow_combined_certifications
            and "certification" in source_header.lower()
        ):
            bounds = _section_bounds(lines, "Certifications")

        if bounds is None:
            insert_at = _insertion_index(lines, before)
            prefix = [""] if insert_at > 0 and lines[insert_at - 1].strip() else []
            suffix = [""] if insert_at < len(lines) and lines[insert_at].strip() else []
            lines = lines[:insert_at] + prefix + replacement + suffix + lines[insert_at:]
        else:
            start, end = bounds
            lines = lines[:start] + replacement + lines[end:]
        return ResumeTailor._join_resume_lines(lines)

    @staticmethod
    def _localize_french_source_section_header(source_header: str, *, canonical: str) -> str:
        normalized = ResumeTailor._ascii_fold(source_header).casefold()
        if canonical == "Education":
            if "certification" in normalized:
                return "Formation et certifications"
            return "Formation"
        if canonical == "Projects":
            if "home lab" in normalized or "laboratoire" in normalized:
                return "Projets & laboratoire à domicile"
            return "Projets"
        if canonical == "Languages":
            return "Langues"
        return source_header

    @staticmethod
    def _localize_source_section_body(
        lines: list[str], *, canonical: str, language: str
    ) -> list[str]:
        if language != "French":
            return lines
        if canonical == "Languages":
            language_names = {"spanish": "espagnol"}
            localized: list[str] = []
            for line in lines:
                match = re.fullmatch(
                    r"Fluent in French and English, plus ([A-Za-z]+)\.",
                    line.strip(),
                )
                if match:
                    third_language = language_names.get(
                        match.group(1).casefold(),
                        match.group(1).casefold(),
                    )
                    localized.append(f"Français et anglais courants; {third_language}.")
                else:
                    localized.append(line)
            return localized
        if canonical == "Education":
            return [ResumeTailor._localize_french_education_line(line) for line in lines]
        if canonical == "Projects":
            return [ResumeTailor._localize_french_project_line(line) for line in lines]
        return lines

    @staticmethod
    def _localize_french_education_line(line: str) -> str:
        localized = line
        replacements = (
            ("Undergraduate Certificate", "Certificat universitaire"),
            ("Analysis & Operational Cybersecurity", "Analyse et cybersécurité opérationnelle"),
            ("2024 – Present", "2024 - Présent"),
            ("2024 - Present", "2024 - Présent"),
            ("Cisco CCNA & CompTIA Linux+ Coursework", "Cours Cisco CCNA et CompTIA Linux+"),
            ("Completed:", "Cours complétés :"),
            (
                "Server Security, and Networking & Security",
                "Server Security et Networking & Security",
            ),
            (" — incl. a hands-on SIEM lab, ", " — incluant un laboratoire pratique SIEM, "),
            ("SOC operations & monitoring", "opérations et surveillance SOC"),
            ("incident response", "réponse aux incidents"),
            ("threat intelligence", "renseignement sur les menaces"),
            (", and ", " et "),
            (" and ", " et "),
            ("Full CCNA networking curriculum", "Programme complet de réseautique CCNA"),
            ("network components", "composants réseau"),
            ("VLSM/subnetting", "VLSM/sous-réseaux"),
            ("routing & switching", "routage et commutation"),
            ("certification exam pending", "examen de certification en attente"),
            ("plus Linux administration", "plus administration Linux"),
            (" in Fedora", " sous Fedora"),
            ("(CLI, scripting)", "(CLI, scripts)"),
            ("B.A., Accounting", "B.A., Comptabilité"),
            ("Administration profile", "profil Administration"),
        )
        for source, replacement in replacements:
            localized = localized.replace(source, replacement)
        return localized

    @staticmethod
    def _localize_french_project_line(line: str) -> str:
        localized = line
        replacements = (
            (
                "Home cybersecurity lab & pen-test sandbox",
                "Laboratoire de cybersécurité à domicile et sandbox de test de pénétration",
            ),
            ("a multi-VM environment", "environnement multi-VM"),
            ("Kali attacker against target hosts", "Kali attaquant contre des hôtes cibles"),
            (
                "for hands-on attack/defense and detection practice",
                "pour la pratique attaque/défense et détection",
            ),
            ("Self-hosted BIND9 DNS server", "Serveur DNS BIND9 autohébergé"),
            (
                "configured, secured, and administered end-to-end",
                "configuré, sécurisé et administré de bout en bout",
            ),
            ("Hands-on security tooling", "Outils de sécurité en pratique"),
            ("packet analysis", "analyse de paquets"),
            ("network scanning", "scan de réseau"),
            ("network monitoring", "surveillance réseau"),
            ("and Kali Linux", "et Kali Linux"),
            (
                "through coursework labs and TryHackMe",
                "via les laboratoires de cours et TryHackMe",
            ),
        )
        for source, replacement in replacements:
            localized = localized.replace(source, replacement)
        return localized.replace(", and ", " et ")

    @staticmethod
    def _polish_french_output(tailored: str, language: str) -> str:
        if language != "French":
            return tailored
        replacements = (
            ("Prendre en charge 100 %", "Pris en charge 100 %"),
            ("Géré plus de 100 %", "Pris en charge 100 %"),
            ("Réalisé les contrats", "Planifié les contrats"),
            (
                "dans une opération rapide et à faible délai",
                "dans un environnement rapide à délais serrés",
            ),
            (
                "dans une opération rapide et exigeante en termes de temps",
                "dans un environnement rapide à délais serrés",
            ),
            (
                "dans une opération rapide et exigeante en temps",
                "dans un environnement rapide à délais serrés",
            ),
            (
                "dans une opération rapide et à haute priorité",
                "dans un environnement rapide à délais serrés",
            ),
            ("fournisseur externe d'IT", "fournisseur TI externe"),
            ("fournisseur externe d\u2019IT", "fournisseur TI externe"),
            ("entreprise à contrats bloqués", "entreprise à contrats fixes"),
            ("entreprise à contrat verrouillé", "entreprise à contrats fixes"),
            ("rétention client", "fidélisation des clients"),
            ("demandes de réclamation de bénéfices", "demandes de prestations"),
            ("demandes de bénéfices", "demandes de prestations"),
            ("en téléphone et par e-mail", "par téléphone et par courriel"),
            ("en téléphone et par courriel", "par téléphone et par courriel"),
            ("par téléphone, chat et e-mail", "par téléphone, chat et courriel"),
            (
                "par téléphone et courriel en résolution à la première tentative",
                "par téléphone et par courriel avec résolution au premier appel",
            ),
            (
                "par téléphone et courriel en résolution en premier appel",
                "par téléphone et par courriel avec résolution au premier appel",
            ),
            (
                "pour une résolution en première instance",
                "avec résolution au premier appel",
            ),
            (
                "taux de résolution en première instance",
                "taux de résolution au premier appel",
            ),
            (
                "résolution à la première tentative",
                "résolution au premier appel",
            ),
            ("Déboguait les problèmes", "Dépanné les problèmes"),
            ("Troublé les problèmes", "Dépanné les problèmes"),
            ("Apporté un support technique", "Fourni un support technique"),
            (
                "en diagnostic et résolution à distance des problèmes",
                "en diagnostiquant et résolvant à distance les problèmes",
            ),
            (
                "diagnostic et résolution à distance des problèmes",
                "diagnostiquant et résolvant à distance les problèmes",
            ),
            ("diagnosticant", "diagnostiquant"),
            ("Trié et escalada", "Trié et escaladé"),
        )
        result = tailored
        for source, replacement in replacements:
            result = result.replace(source, replacement)
        return result

    @staticmethod
    def _strip_duplicate_bullets(tailored: str) -> str:
        lines = tailored.splitlines()
        bullet_indices: dict[str, list[int]] = {}
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith(("•", "-", "*")):
                continue
            normalized = re.sub(r"^\s*[•*-]\s*", "", stripped)
            normalized = re.sub(r"[*_`]", "", normalized)
            normalized = re.sub(r"\s+", " ", normalized.casefold()).strip(" .;:")
            if len(normalized.split()) < 6:
                continue
            bullet_indices.setdefault(normalized, []).append(index)
        drop = {
            index
            for indices in bullet_indices.values()
            if len(indices) > 1
            for index in indices[:-1]
        }
        if not drop:
            return tailored
        return ResumeTailor._join_resume_lines(
            [line for index, line in enumerate(lines) if index not in drop]
        )

    @staticmethod
    def _join_resume_lines(lines: list[str]) -> str:
        result: list[str] = []
        previous_blank = False
        for line in lines:
            stripped = line.rstrip()
            if stripped:
                result.append(stripped)
                previous_blank = False
            elif result and not previous_blank:
                result.append("")
                previous_blank = True
        while result and not result[-1].strip():
            result.pop()
        return "\n".join(result).strip()

    @staticmethod
    def _strip_unverifiable_aspirations(tailored: str) -> str:
        """Remove target-role aspiration sentences before grounding.

        Aspirations such as "Seeking to leverage..." are not résumé-source facts, and the grounding
        verifier correctly cannot cite them. They are useful in a cover letter, but noisy in an
        unattended tailored CV.
        """
        result = re.sub(
            r"(?:(?<=^)|(?<=[.!?])\s+)"
            r"(?:Seeking|Looking|Aiming)\s+to\s+[^.!?]*(?:\.|$)",
            "",
            tailored,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        result = re.sub(
            r",?\s+(?:seeking|looking|aiming)\s+to\s+[^.!?]*(?=\.|$)",
            "",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            r"(?:(?<=^)|(?<=[.!?])\s+)"
            r"(?:Recherche|Souhaite|Vise)\s+(?:un\s+)?(?:r[oô]le|poste|emploi)\s+[^.!?]*(?:\.|$)",
            "",
            result,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return re.sub(r" {2,}", " ", result).strip()

    def _strip_unbacked_responsibility_bullets(self, tailored: str, original: str) -> str:
        """Drop high-risk responsibility bullets whose actors/domains are absent from the source."""
        original_lower = self._ascii_fold(original).lower()
        risky_terms = (
            "collaborat",
            "data scientist",
            "stakeholder",
            "architect",
            "architecture",
            "microservice",
            "deployed",
            "deployment",
            "uptime",
            "rollout",
            "workflow",
            "optimization",
            "optimized",
            "optimizing",
            "real-time",
            "high-concurrency",
            "high-traffic",
            "user requests",
            "ci/cd",
            "testing",
            "cloud-native",
            "fault tolerance",
            "scaled",
            "scalable",
            "scalability",
            "database load",
            "global user growth",
            "internal teams",
            "technical guidance",
            "urgence",
            "incidents techniques",
            "technical incidents",
            "continuite",
            "temps reel",
            "seminaire",
            "seminaires",
            "fluidite operationnelle",
            "fluidite des processus",
            "interactions clients",
            "operational continuity",
        )
        result_lines: list[str] = []
        for line in tailored.splitlines():
            stripped = line.strip()
            is_bullet = bool(re.match(r"^[•*\-]\s+", stripped))
            lower = self._ascii_fold(stripped).lower()
            missing_terms = [
                term for term in risky_terms if term in lower and term not in original_lower
            ]
            if stripped and missing_terms and not _looks_like_section_header(stripped):
                logger.info(
                    "Dropped responsibility line with unbacked term(s) %s: %s",
                    ", ".join(missing_terms),
                    stripped,
                )
                continue
            if is_bullet:
                if re.search(r"\bthat\s*[.;]?$", stripped, flags=re.IGNORECASE):
                    logger.info("Dropped incomplete responsibility bullet: %s", stripped)
                    continue
            result_lines.append(line)
        return "\n".join(result_lines).strip()

    @staticmethod
    def _support_domain_contexts(text: str, support_domain_terms: tuple[str, ...]) -> set[str]:
        contexts: set[str] = set()
        recent_headers: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            is_bullet = bool(re.match(r"^[•*\-]\s+", stripped))
            folded = ResumeTailor._ascii_fold(stripped.strip("*_`")).lower()
            if (
                not is_bullet
                and not _looks_like_section_header(stripped)
                and not re.search(r"\b\d{4}\b", folded)
                and len(folded.split()) <= 8
            ):
                recent_headers.append(folded)
                recent_headers = recent_headers[-4:]
            if any(term in folded for term in support_domain_terms):
                contexts.update(recent_headers)
        return contexts

    @staticmethod
    def _context_matches_source(current_context: str, source_contexts: set[str]) -> bool:
        if not current_context:
            return False
        current_tokens = set(re.findall(r"[a-z0-9]{3,}", current_context))
        for source_context in source_contexts:
            source_tokens = set(re.findall(r"[a-z0-9]{3,}", source_context))
            if current_tokens and source_tokens and len(current_tokens & source_tokens) >= 2:
                return True
            if current_context in source_context or source_context in current_context:
                return True
        return False

    @staticmethod
    def _strip_misplaced_support_domain_bullets(tailored: str, original: str) -> str:
        """Drop signal/receiver/connectivity bullets when the source employer context differs."""
        support_domain_terms = ("signal", "recepteur", "receiver", "connectivite", "connectivity")
        source_contexts = ResumeTailor._support_domain_contexts(original, support_domain_terms)
        if not source_contexts:
            return tailored

        current_context = ""
        result_lines: list[str] = []
        for line in tailored.splitlines():
            stripped = line.strip()
            is_bullet = bool(re.match(r"^[•*\-]\s+", stripped))
            folded = ResumeTailor._ascii_fold(stripped.strip("*_`")).lower()
            if not is_bullet and folded:
                if (
                    not _looks_like_section_header(stripped)
                    and not re.search(r"\b\d{4}\b", folded)
                    and len(folded.split()) <= 8
                ):
                    current_context = folded
                result_lines.append(line)
                continue
            if is_bullet and any(term in folded for term in support_domain_terms):
                if not ResumeTailor._context_matches_source(current_context, source_contexts):
                    logger.info(
                        "Dropped support-domain bullet outside its source employer context: %s",
                        stripped,
                    )
                    continue
            result_lines.append(line)
        return "\n".join(result_lines).strip()

    def _strip_low_evidence_bullets(self, tailored: str, original: str) -> str:
        """Drop generated bullets that are weakly supported by source résumé tokens."""
        source_tokens = self._evidence_tokens(original)
        if not source_tokens:
            return tailored

        result_lines: list[str] = []
        for line in tailored.splitlines():
            stripped = line.strip()
            if not re.match(r"^[•*\-]\s+", stripped):
                result_lines.append(line)
                continue
            bullet_tokens = self._evidence_tokens(stripped)
            if len(bullet_tokens) < 4:
                result_lines.append(line)
                continue
            overlap = len(bullet_tokens & source_tokens) / len(bullet_tokens)
            if overlap < 0.30:
                logger.info(
                    "Dropped low-evidence tailored bullet (%.0f%% source overlap): %s",
                    overlap * 100,
                    stripped,
                )
                continue
            result_lines.append(line)
        return "\n".join(result_lines).strip()

    @staticmethod
    def _localize_standard_labels_for_language(tailored: str, language: str) -> str:
        """Normalize standard résumé labels when the requested output language is French."""
        if language != "French":
            return tailored

        replacements = {
            "summary": "Profil",
            "professional summary": "Profil",
            "resume": "Profil",
            "profil": "Profil",
            "skills": "Compétences",
            "technical skills": "Compétences",
            "competences": "Compétences",
            "experience": "Expérience",
            "professional experience": "Expérience",
            "work experience": "Expérience",
            "experience professionnelle": "Expérience",
            "education": "Formation",
            "education & certifications": "Formation et certifications",
            "education and certifications": "Formation et certifications",
            "certifications": "Certifications",
            "languages": "Langues",
            "langues": "Langues",
            "projects": "Projets",
            "project": "Projets",
            "projects & home lab": "Projets & laboratoire à domicile",
            "projets & laboratoire a domicile": "Projets & laboratoire à domicile",
        }
        lines: list[str] = []
        for line in tailored.splitlines():
            stripped = line.strip()
            bold = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
            label = bold.group(1).strip() if bold else stripped
            normalized_label = ResumeTailor._ascii_fold(label).casefold()
            localized = replacements.get(normalized_label)
            if localized:
                prefix = line[: len(line) - len(line.lstrip())]
                lines.append(f"{prefix}**{localized}**" if bold else f"{prefix}{localized}")
                continue
            language_match = re.fullmatch(
                r"Fluent in French and English, plus ([A-Za-z]+)\.",
                stripped,
            )
            if language_match:
                language_names = {"spanish": "espagnol"}
                third_language = language_names.get(
                    language_match.group(1).casefold(),
                    language_match.group(1).casefold(),
                )
                prefix = line[: len(line) - len(line.lstrip())]
                lines.append(f"{prefix}Français et anglais courants; {third_language}.")
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _strip_non_source_bold_entry_headings(tailored: str, original: str) -> str:
        """Remove rewritten bold entry headings that are not source-owned labels.

        Job titles, credential titles, school names, and similar entry labels are facts of record.
        The prompt asks the LLM to keep them verbatim, but translated French drafts can still turn
        source titles into new bold headings. Keep exact source headings and known section labels;
        drop the rest so the verifier does not have to certify a rewritten credential/title.
        """
        normalized_source = _normalize_source_owned_fragment(original)
        lines: list[str] = []
        for line in tailored.splitlines():
            stripped = line.strip()
            bold = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
            if bold and not _looks_like_section_header(stripped):
                label = bold.group(1).strip()
                if _normalize_source_owned_fragment(label) not in normalized_source:
                    logger.info("Dropped non-source bold entry heading: %s", stripped)
                    continue
            lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _ascii_fold(text: str) -> str:
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    @staticmethod
    def _evidence_tokens(text: str) -> set[str]:
        stop = {
            "and",
            "the",
            "with",
            "using",
            "used",
            "for",
            "from",
            "that",
            "this",
            "into",
            "across",
            "based",
            "through",
            "while",
            "per",
            "day",
            "services",
            "systems",
            "application",
            "applications",
            "a",
            "au",
            "aux",
            "avec",
            "ce",
            "ces",
            "contre",
            "de",
            "des",
            "du",
            "en",
            "et",
            "la",
            "le",
            "les",
            "pour",
            "sur",
            "un",
            "une",
        }
        aliases = {
            "asynchronous": "async",
            "pipelines": "pipeline",
            "schemas": "schema",
            "models": "model",
            "workflows": "workflow",
            "operations": "operation",
            "operational": "operation",
            "operationnels": "operation",
            "operationnelles": "operation",
            "gere": "managed",
            "gerer": "managed",
            "gestion": "managed",
            "quotidien": "daily",
            "quotidiens": "daily",
            "quotidienne": "daily",
            "quotidiennes": "daily",
            "livraison": "delivery",
            "conducteur": "driver",
            "conducteurs": "drivers",
            "chauffeur": "driver",
            "chauffeurs": "drivers",
            "unique": "single",
            "escalade": "escalation",
            "escaladee": "escalation",
            "escalader": "escalation",
            "escalated": "escalation",
            "probleme": "issues",
            "problemes": "issues",
            "client": "client",
            "clients": "client",
            "differends": "disputes",
            "desaccords": "disputes",
            "reclamation": "claims",
            "reclamations": "claims",
            "articles": "item",
            "manquants": "missing",
            "horaire": "scheduling",
            "horaires": "scheduling",
            "coordination": "coordination",
            "coordonne": "coordinated",
            "coordonnes": "coordinated",
            "coordinated": "coordination",
            "realise": "booked",
            "realises": "booked",
            "contrat": "contracts",
            "contrats": "contracts",
            "assigne": "assigned",
            "assignes": "assigned",
            "operateur": "operators",
            "operateurs": "operators",
            "planning": "schedules",
            "plannings": "schedules",
            "hebdomadaire": "weekly",
            "hebdomadaires": "weekly",
            "mensuel": "monthly",
            "mensuels": "monthly",
            "rapide": "fast",
            "exigeante": "critical",
            "exigeant": "critical",
            "temps": "time",
            "mecanique": "mechanical",
            "disponibilite": "operational",
            "parc": "fleet",
            "resolu": "resolved",
            "resolues": "resolved",
            "demandes": "inquiries",
            "benefices": "benefit",
            "prestations": "benefit",
            "telephone": "phone",
            "courriel": "email",
            "premier": "first",
            "appel": "call",
            "diagnostique": "troubleshot",
            "diagnostiques": "troubleshot",
            "site": "website",
            "web": "website",
            "pour": "across",
            "fourni": "delivered",
            "soutien": "support",
            "support": "support",
            "niveau": "tier",
            "clavardage": "chat",
            "diagnostic": "diagnosing",
            "resolution": "resolution",
            "distance": "remotely",
            "recepteur": "receiver",
            "connectivite": "connectivity",
            "maintenu": "maintained",
            "taux": "rate",
            "environ": "roughly",
            "triage": "triaged",
            "trie": "triaged",
            "tries": "triaged",
            "complexes": "complex",
            "niveaux": "tiers",
            "superieurs": "higher",
            "procedure": "procedures",
            "procedures": "procedures",
            "documentees": "documented",
            "projets": "projects",
            "projet": "projects",
            "laboratoire": "lab",
            "laboratoires": "labs",
            "domicile": "home",
            "cybersecurite": "cybersecurity",
            "test": "test",
            "penetration": "pen-test",
            "sandbox": "sandbox",
            "environnement": "environment",
            "multi": "multi",
            "multi-vm": "multi-vm",
            "vm": "vm",
            "attaquant": "attacker",
            "hote": "host",
            "hotes": "hosts",
            "cible": "target",
            "cibles": "targets",
            "attaque": "attack",
            "defense": "defense",
            "detection": "detection",
            "pratique": "practice",
            "pratiques": "practice",
            "serveur": "server",
            "auto": "self",
            "heberge": "hosted",
            "auto-heberge": "self-hosted",
            "configure": "configured",
            "securise": "secured",
            "administre": "administered",
            "bout": "end",
            "outils": "tooling",
            "outil": "tooling",
            "securite": "security",
            "paquet": "packet",
            "paquets": "packet",
            "analyse": "analysis",
            "scan": "scanning",
            "reseau": "network",
            "surveillance": "monitoring",
            "cours": "coursework",
        }
        normalized = ResumeTailor._ascii_fold(text).lower()
        tokens = set(re.findall(r"[a-z0-9][a-z0-9+.-]{2,}", normalized))
        return {aliases.get(token, token) for token in tokens if token not in stop}

    @staticmethod
    def _normalize_date_range_dashes(tailored: str) -> str:
        """Use plain hyphens in year ranges to match common source résumé syntax."""
        return re.sub(
            r"(\b\d{4})[–—](Present|Current|présent|actuel|\d{4}\b)",
            r"\1-\2",
            tailored,
            flags=re.IGNORECASE,
        )

    def _strip_hallucinated_education(self, tailored: str, original: str) -> str:
        """Remove Education section if it wasn't in the original resume."""
        if _source_section(original, "Education") is not None:
            return tailored

        lines = tailored.split("\n")
        result_lines: list[str] = []
        in_education = False

        for line in lines:
            stripped = line.strip()

            if section_header(stripped) == "Education":
                in_education = True
                continue

            # Detect next section header (end of education)
            if in_education and _looks_like_section_header(stripped):
                in_education = False
                result_lines.append(line)
                continue

            # Skip education content
            if in_education:
                continue

            result_lines.append(line)

        return "\n".join(result_lines)

    def _strip_unbacked_optional_sections(self, tailored: str, original: str) -> str:
        """Remove optional sections if they weren't in the original resume."""
        sections_to_strip = {
            section
            for section in ("Certifications", "Languages", "Volunteer", "Awards", "Interests")
            if _source_section(original, section) is None
        }
        if not sections_to_strip:
            return tailored

        lines = tailored.split("\n")
        result_lines: list[str] = []
        in_section = False

        for line in lines:
            stripped = line.strip()

            if section_header(stripped) in sections_to_strip:
                in_section = True
                continue

            # Detect next section header (end of the empty section)
            if in_section and _looks_like_section_header(stripped):
                in_section = False
                result_lines.append(line)
                continue

            # Skip empty-section content
            if in_section:
                continue

            result_lines.append(line)

        return "\n".join(result_lines)

    def _strip_empty_certifications_languages(self, tailored: str, original: str) -> str:
        """Backward-compatible wrapper for the broader optional-section stripper."""
        return self._strip_unbacked_optional_sections(tailored, original)

    async def _call_llm(self, prompt: str, temperature: float = 0.4) -> str:
        """Call the LLM and return response text — circuit-breaker protected.

        All résumé-tailoring LLM calls (tailor, refine, change-summary) funnel
        through here, so one breaker guards them all.
        """

        async def _do() -> str:
            try:
                quiet_litellm()
                from litellm import acompletion

                model = litellm_model(self._config)

                response = await acompletion(
                    model=model,
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    messages=[
                        {"role": "system", "content": TAILOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    **litellm_completion_kwargs(self._config, temperature=temperature),
                )

                from job_applicator.utils.llm import strip_thinking_process

                content = strip_thinking_process(response.choices[0].message.content)
                return content.strip()

            except Exception as exc:
                from job_applicator.utils.llm import llm_call_error

                raise llm_call_error(exc, self._config.api_base) from exc

        return await self._breaker.call(_do)

    async def _summarize_changes(self, original: str, tailored: str) -> str:
        """Generate a summary of changes between original and tailored.

        A failure here RAISES (via _call_llm's typed error) rather than fabricating a summary
        string — a made-up "(summary generation failed)" value masks the failure as a real result.
        """
        prompt = CHANGES_PROMPT_TEMPLATE.format(
            original_preview=original[:500],
            tailored_preview=tailored[:500],
        )
        return await self._call_llm(prompt, temperature=0.2)
