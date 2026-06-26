"""LLM-powered resume tailoring — rewrites resume content for a specific job."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from job_applicator.config import LLMConfig
from job_applicator.documents.style_analyzer import StyleAnalyzer
from job_applicator.exceptions import LLMError
from job_applicator.models import (
    DateAuditResult,
    JobListing,
    ResumeData,
    StyleGuide,
    TailoredResume,
)
from job_applicator.utils.llm import CircuitOpenError, LLMRuntime, quiet_litellm
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

if TYPE_CHECKING:
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.embeddings.matching import JobMatcher

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
    }
)

_COLON_HEADER_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s*$")


def _is_section_header(stripped: str) -> bool:
    """Return True if *stripped* line looks like a resume section header."""
    normalized = re.sub(r"\s+", " ", stripped.strip())
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


MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


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
    """Parse and validate dates from resume text.

    Checks for:
    - Chronological ordering (most recent first within each section)
    - Timeline coherence (no impossible overlaps, gap detection)
    - Staleness (entries that suggest the CV is outdated)
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

        # Check for "Present" entries that might be stale
        for e in entries:
            if e.is_current:
                # This is fine — still active
                pass

        # Check education dates for staleness
        edu_entries = [e for e in entries if e.section == "Education"]
        for e in edu_entries:
            if e.end_year and not e.is_current:
                years_since = self._now.year - e.end_year
                if years_since > 10:
                    staleness_issues.append(
                        f"Education '{e.label}' ended {years_since} years ago "
                        f"({e.end_year}). Consider if this is still relevant."
                    )

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
            is_stale=bool(staleness_issues),
            is_ordered=not ordering_issues,
            latest_date=latest_str,
            earliest_date=earliest_str,
        )

    def _parse_all_dates(self, text: str) -> list[ParsedDate]:
        """Extract all date entries from resume text."""
        entries: list[ParsedDate] = []
        lines = text.split("\n")
        current_section = ""

        # Pattern: "Month Year - Month Year" or "Month Year - Present"
        # Also: "YYYY - YYYY" or "YYYY - Present"
        date_pattern = re.compile(
            r"(?:"
            r"(?:(\w+)\s+)?(\d{4})\s*[-\u2013\u2014]\s*"
            r"(?:(\w+)\s+)?(\d{4}|[Pp]resent)"
            r"|"
            r"(\d{4})\s*[-\u2013\u2014]\s*(\d{4}|[Pp]resent)"
            r")"
        )

        # Section headers
        section_pattern = re.compile(
            r"^\*{0,2}\s*(EXPERIENCE|EDUCATION|EMPLOYMENT|VOLUNTEER"
            r"|CERTIFICATIONS|INTERNSHIP)\s*\*{0,2}$",
            re.IGNORECASE,
        )

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Track current section
            sec_match = section_pattern.match(stripped)
            if sec_match:
                current_section = sec_match.group(1).title()
                continue

            # Look for dates
            date_match = date_pattern.search(stripped)
            if not date_match:
                continue

            # Find the label (previous non-empty line that looks like a title)
            label = self._find_label(lines, i)

            # Parse the date
            if date_match.group(5):  # YYYY - YYYY format
                start_year = int(date_match.group(5))
                start_month = None
                end_str = date_match.group(6)
            else:  # Month Year - Month Year format
                start_month = self._parse_month(date_match.group(1))
                start_year = int(date_match.group(2))
                end_str = date_match.group(4)

            is_current = end_str.lower() == "present"
            end_year: int | None = None
            end_month: int | None = None
            if not is_current:
                if date_match.group(5):
                    end_year = int(end_str)
                else:
                    end_month = self._parse_month(date_match.group(3))
                    end_year = int(end_str)

            entries.append(
                ParsedDate(
                    label=label,
                    section=current_section or "Unknown",
                    start_year=start_year,
                    start_month=start_month,
                    end_year=end_year,
                    end_month=end_month,
                    is_current=is_current,
                )
            )

        return entries

    def _find_label(self, lines: list[str], date_line_idx: int) -> str:
        """Find the label for a date entry by looking at preceding lines."""
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

    def _parse_month(self, month_str: str | None) -> int | None:
        """Parse month name to number."""
        if not month_str:
            return None
        return MONTH_MAP.get(month_str.lower().strip())

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
    "section to tailor. Mirror the job posting's language. Mention the "
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
    "   - Do NOT invent metrics, responsibilities, or technologies not in "
    "the original\n"
    "   - Include ALL jobs from the original. Do not drop any.\n"
    "   - Write 3-5 bullets per job\n\n"
    "5. EDUCATION — include ALL education entries from the original. "
    "Preserve institution names, degrees, course names, and dates exactly. "
    "Do NOT merge education entries with Certifications — they are separate "
    "sections. Include EVERY entry even if it overlaps with Certifications "
    "(e.g., a 'Cert Preparation' course is Education, not a Certification). "
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
    "- REQUIRED STRUCTURE — the output MUST contain these exact bold section "
    "headers on their own lines:\n"
    "  **Skills**\n"
    "  **Experience**\n"
    "  **Education** (if present in original)\n"
    "  **Certifications** (if present in original)\n"
    "  **Languages** (if present in original)\n"
    "  Do not omit these headers. Place the **Skills** header directly before "
    "the skills list and the **Experience** header directly before the first "
    "job.\n\n"
    "ABSOLUTE RULES:\n"
    "- NEVER add skills, tools, or technologies not in the original resume\n"
    "- NEVER invent experience, metrics, or responsibilities\n"
    "- NEVER add education, certifications, or credentials not in original\n"
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
    "and mobile platforms, maintaining a 95% first-call resolution rate'"
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
    "Education entries that MUST appear in the output (do not merge with "
    "Certifications — they are separate sections):\n"
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
    "Return ONLY 3-5 bullet points describing the key changes. "
    "No thinking process, no explanation, just the bullet points."
)


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
    ) -> TailoredResume:
        """Tailor a resume for a specific job.

        Args:
            resume: Original parsed resume
            job: Target job listing
            user_instructions: Optional user guidance for tailoring
            style_guide: Optional style guide to apply
            tone_profile: Optional pre-detected ToneProfile to avoid re-detection
            matcher: Optional JobMatcher instance to reuse (avoids re-creating)

        Returns:
            TailoredResume with full metadata
        """
        from job_applicator.config import EmbeddingConfig
        from job_applicator.embeddings.matching import JobMatcher

        if matcher is None:
            matcher = JobMatcher(
                EmbeddingConfig(device="cpu", memory_limit_gb=0.5),
                self._config,
                self._runtime,
            )
        match_result = await matcher.match_resume_to_job(resume, job)

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

        tailored_text = await self._call_llm(prompt)
        tailored_text = self._validate_skills(tailored_text, resume.skills)
        tailored_text = self._strip_hallucinated_tools(
            tailored_text, resume.raw_text, job.requirements
        )
        tailored_text = self._strip_hallucinated_education(tailored_text, resume.raw_text)
        tailored_text = self._strip_empty_certifications_languages(tailored_text, resume.raw_text)
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
            f"skills list. Do NOT add Education if none exists in original."
            f"{tone_directive}"
            f"{style_section}\n"
            f"Return the complete updated resume text."
        )

        refined_text = await self._call_llm(prompt, temperature=0.3)
        refined_text = self._validate_skills(refined_text, original_resume.skills)
        refined_text = self._strip_hallucinated_tools(
            refined_text, original_resume.raw_text, job.requirements
        )
        refined_text = self._strip_hallucinated_education(refined_text, original_resume.raw_text)
        refined_text = self._strip_empty_certifications_languages(
            refined_text, original_resume.raw_text
        )
        changes = await self._summarize_changes(current_tailored.tailored_text, refined_text)

        # Recompute match scores against the refined text
        from job_applicator.config import EmbeddingConfig
        from job_applicator.embeddings.matching import JobMatcher

        if matcher is None:
            matcher = JobMatcher(
                EmbeddingConfig(device="cpu", memory_limit_gb=0.5),
                self._config,
                self._runtime,
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
        """Extract education entries as a numbered checklist for the prompt.

        Parses the Education section and returns each entry as a numbered
        item so the LLM can't silently drop any.
        """
        lines = text.split("\n")
        entries: list[str] = []
        in_edu = False
        current_entry: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Detect education section start
            if re.match(r"^\*{0,2}\s*EDUCATION\s*\*{0,2}$", stripped, re.IGNORECASE):
                in_edu = True
                continue

            # Detect next section (end of education)
            if in_edu and _looks_like_section_header(stripped):
                if current_entry:
                    entries.append(" ".join(current_entry).strip())
                    current_entry = []
                in_edu = False
                continue

            if in_edu and stripped:
                # Date line = end of current entry
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

        numbered = []
        for i, entry in enumerate(entries, 1):
            numbered.append(f"  {i}. {entry}")
        return "\n".join(numbered)

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

            # Detect skills section header
            # Header set kept aligned with the parser's
            # ResumeLoader._extract_skills_section() regex.
            if re.match(
                r"^\*{0,2}\s*"
                r"(?:(?:Technical|Core|Key|Professional|Relevant|Soft)\s+)?"
                r"(?:Skills|Competencies|Proficiencies)"
                r"\s*\*{0,2}$",
                stripped,
                re.IGNORECASE,
            ):
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
                skill_text = re.sub(r"^[\*\-\•\·\s]+", "", stripped)
                skill_text = skill_text.strip()

                if not skill_text or len(skill_text) < 3:
                    result_lines.append(line)
                    continue

                skill_norm = normalize_skill(skill_text).lower()

                # Drop generic traits unless they were literally in the original resume.
                if is_hard_negative(skill_norm) and skill_norm not in norm_skills:
                    logger.info("Stripped hard-negative skill: %s", skill_text)
                    continue

                # Check if this skill matches any original skill
                is_original = any(_skills_match(skill_norm, orig) for orig in norm_skills)

                if is_original:
                    result_lines.append(line)
                else:
                    # Hallucinated skill — skip it
                    logger.info("Stripped hallucinated skill: %s", skill_text)
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

    def _strip_hallucinated_education(self, tailored: str, original: str) -> str:
        """Remove Education section if it wasn't in the original resume."""
        import re

        # Check if original has an education section (case-insensitive word boundary)
        if re.search(r"\bEDUCATION\b", original, re.IGNORECASE):
            return tailored  # Original has education, keep it

        # Find and remove education section from tailored text
        lines = tailored.split("\n")
        result_lines: list[str] = []
        in_education = False

        for line in lines:
            stripped = line.strip()

            # Detect education section header (with optional markdown bold)
            if re.match(r"^\*{0,2}\s*EDUCATION\s*\*{0,2}$", stripped, re.IGNORECASE):
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

    def _strip_empty_certifications_languages(self, tailored: str, original: str) -> str:
        """Remove Certifications/Languages sections if they weren't in the original resume."""
        import re

        sections_to_strip: list[str] = []
        if not re.search(r"\bCERTIFICATIONS\b", original, re.IGNORECASE):
            sections_to_strip.append("CERTIFICATIONS")
        if not re.search(r"\bLANGUAGES\b", original, re.IGNORECASE):
            sections_to_strip.append("LANGUAGES")
        if not sections_to_strip:
            return tailored

        lines = tailored.split("\n")
        result_lines: list[str] = []
        in_section = False

        for line in lines:
            stripped = line.strip()

            # Detect one of the target section headers (with optional markdown bold)
            if any(
                re.match(rf"^\*{{0,2}}\s*{section}\s*\*{{0,2}}$", stripped, re.IGNORECASE)
                for section in sections_to_strip
            ):
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

    async def _call_llm(self, prompt: str, temperature: float = 0.4) -> str:
        """Call the LLM and return response text — circuit-breaker protected.

        All résumé-tailoring LLM calls (tailor, refine, change-summary) funnel
        through here, so one breaker guards them all.
        """

        async def _do() -> str:
            try:
                quiet_litellm()
                from litellm import acompletion

                model = (
                    f"openai/{self._config.model}" if self._config.api_base else self._config.model
                )

                response = await acompletion(
                    model=model,
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    messages=[
                        {"role": "system", "content": TAILOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self._config.max_tokens,
                    temperature=temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

                from job_applicator.utils.llm import strip_thinking_process

                content = strip_thinking_process(response.choices[0].message.content)
                return content.strip()

            except Exception as exc:
                from job_applicator.utils.llm import llm_call_error

                raise llm_call_error(exc, self._config.api_base) from exc

        return await self._breaker.call(_do)

    async def _summarize_changes(self, original: str, tailored: str) -> str:
        """Generate a summary of changes between original and tailored."""
        try:
            prompt = CHANGES_PROMPT_TEMPLATE.format(
                original_preview=original[:500],
                tailored_preview=tailored[:500],
            )
            return await self._call_llm(prompt, temperature=0.2)
        except Exception:
            return "Changes applied (summary generation failed)"
