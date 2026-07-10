"""LLM-powered resume tailoring — rewrites resume content for a specific job."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import TYPE_CHECKING

from job_applicator.config import LLMConfig
from job_applicator.documents.resume import MONTH_MAP, parse_date_range, section_header
from job_applicator.documents.resume_document import ResumeDocument
from job_applicator.documents.resume_overlay import ResumeOverlayGenerator
from job_applicator.documents.source_facts import build_source_fact_catalog
from job_applicator.documents.source_realization import realize_resume_statement
from job_applicator.exceptions import ConfigError, LLMError, TailorIntegrityError
from job_applicator.models import (
    DateAuditResult,
    GroundingReport,
    JobListing,
    ResumeData,
    StyleGuide,
    TailoredResume,
)
from job_applicator.utils.language import detect_language, resolve_output_language
from job_applicator.utils.llm import LLMRuntime
from job_applicator.utils.logging import get_logger

if TYPE_CHECKING:
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.embeddings.matching import JobMatcher, MatchResult

logger = get_logger("documents.resume_tailor")


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


class ResumeTailor:
    """Target a résumé through one bounded summary overlay."""

    def __init__(
        self,
        config: LLMConfig,
        runtime: LLMRuntime | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime or LLMRuntime.defaults(name="resume-tailor")
        self._overlay_generator = ResumeOverlayGenerator(
            config,
            self._runtime,
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
        target_language = resolve_output_language(self._config.language, job.description)
        self._require_matching_source_language(resume.raw_text, target_language)

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
        del tone_profile

        logger.info(
            "Generating source-preserving résumé overlay in %s (language setting=%s) for %s",
            target_language,
            self._config.language,
            job.title,
        )
        tailored_text, overlay = await self._overlay_generator.generate(
            resume=resume,
            job=job,
            language=target_language,
            style_guide=style_guide,
            user_instructions=user_instructions.strip(),
        )

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
            changes_summary=(
                "Replaced the summary with three source-backed sentences; preserved every "
                "non-summary source section."
            ),
            user_modifications=user_instructions,
            prompt_version=overlay.architecture_version,
            grounding_report=GroundingReport(),
            overlay=overlay,
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
        """Generate an overlay and verify the immutable-body boundary."""
        result = await self.tailor(
            resume, job, user_instructions, style_guide, tone_profile, matcher, match_result
        )
        return await self.verify_tailored(result, resume)

    async def verify_tailored(self, result: TailoredResume, resume: ResumeData) -> TailoredResume:
        """Fail closed unless only the declared summary overlay changed."""

        if result.overlay is None:
            raise TailorIntegrityError("Tailored résumé is missing source-overlay provenance.")
        cited_ids = [
            fact_id
            for sentence in result.overlay.summary_sentences
            for fact_id in sentence.fact_ids
        ]
        if any(len(sentence.fact_ids) != 1 for sentence in result.overlay.summary_sentences):
            raise TailorIntegrityError(
                "Each generated résumé summary sentence must cite exactly one source fact."
            )
        if len(set(cited_ids)) != 3:
            raise TailorIntegrityError(
                "Generated résumé summary provenance must cite three distinct source facts."
            )
        facts_by_id = {fact.fact_id: fact for fact in build_source_fact_catalog(resume).facts}
        if not set(cited_ids) <= facts_by_id.keys():
            raise TailorIntegrityError(
                "Generated résumé summary provenance cites facts absent from the source résumé."
            )
        if any(
            sentence.text != realize_resume_statement(facts_by_id[sentence.fact_ids[0]]).text
            for sentence in result.overlay.summary_sentences
        ):
            raise TailorIntegrityError(
                "Generated résumé summary differs from deterministic source realization."
            )
        source_document = ResumeDocument.parse(resume.raw_text)
        result_document = ResumeDocument.parse(result.tailored_text)
        source_digest = source_document.non_summary_sha256()
        if result.overlay.source_body_sha256 != source_digest:
            raise TailorIntegrityError(
                "Tailored résumé provenance does not match the current source résumé."
            )
        if result_document.non_summary_sha256() != source_digest:
            raise TailorIntegrityError(
                "Tailored résumé changed content outside the generated summary."
            )
        summaries = result_document.summary_sections()
        if len(summaries) != 1:
            raise TailorIntegrityError("Tailored résumé must contain exactly one summary section.")
        actual_summary = re.sub(r"\s+", " ", " ".join(summaries[0].body_lines)).strip()
        declared_summary = re.sub(
            r"\s+",
            " ",
            " ".join(sentence.text for sentence in result.overlay.summary_sentences),
        ).strip()
        if actual_summary != declared_summary:
            raise TailorIntegrityError(
                "Tailored résumé summary does not match its source-backed overlay provenance."
            )
        result.grounding_report = GroundingReport()
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
        """Regenerate the bounded overlay and verify the immutable source body."""
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
        target_language = resolve_output_language(self._config.language, job.description)
        self._require_matching_source_language(original_resume.raw_text, target_language)

        del tone_profile
        refined_text, overlay = await self._overlay_generator.generate(
            resume=original_resume,
            job=job,
            language=target_language,
            style_guide=style_guide,
            user_instructions=user_feedback.strip(),
        )

        if matcher is None:
            raise ConfigError(
                "Résumé refinement requires a configured JobMatcher. Pass the command/TUI/batch "
                "matcher so embeddings use the configured device instead of constructing a "
                "hidden fallback matcher."
            )

        synthetic_resume = original_resume.model_copy(update={"raw_text": refined_text})
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
            changes_summary=(
                "Regenerated the three source-backed summary sentences from the original résumé; "
                "preserved every non-summary source section."
            ),
            user_modifications=user_feedback,
            attempt=current_tailored.attempt + 1,
            prompt_version=overlay.architecture_version,
            grounding_report=GroundingReport(),
            overlay=overlay,
        )

    @staticmethod
    def _require_matching_source_language(source_text: str, target_language: str) -> None:
        """Fail closed when tailoring would require unverified machine translation."""

        source_language = "French" if detect_language(source_text) == "fr" else "English"
        if source_language != target_language:
            raise LLMError(
                "Cross-language resume tailoring is unavailable: the source resume is "
                f"{source_language}, but the requested output is {target_language}. Provide a "
                f"{target_language} source resume so tailoring and grounding stay in one language."
            )
