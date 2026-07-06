"""Shared Pydantic models — typed data contracts between layers."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, computed_field


class JobBoard(StrEnum):
    """Supported job boards."""

    LINKEDIN = "linkedin"
    INDEED = "indeed"

    @property
    def display_name(self) -> str:
        """Human-facing board name (proper casing) for UI/messages — the enum *value*
        ('linkedin') is the wire form; this is what a user should see ('LinkedIn')."""
        return {"linkedin": "LinkedIn", "indeed": "Indeed"}.get(self.value, self.value.title())


class ApplicationStatus(StrEnum):
    """Outcome of an application attempt."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FAILED = "failed"
    SKIPPED = "skipped"
    ALREADY_APPLIED = "already_applied"


class FunnelStatus(StrEnum):
    """Where a job sits in the application funnel.

    The funnel *head* (found → matched → tailored → cover_letter) lives in
    ``jobs_store.JobStore``; ``APPLIED`` is authoritative in
    ``state.ApplicationState`` (it drives the daily cap), so the store never
    forks it — the ``status`` view composes both.
    """

    FOUND = "found"
    MATCHED = "matched"
    TAILORED = "tailored"
    COVER_LETTER = "cover_letter"
    APPLIED = "applied"


class Format(StrEnum):
    """Valid --format values for artifact output."""

    TXT = "txt"
    PDF = "pdf"
    BOTH = "both"


class JobListing(BaseModel):
    """Scraped job data from a job board."""

    title: str
    company: str
    url: HttpUrl
    description: str = ""
    location: str = ""
    salary: str | None = None
    requirements: list[str] = Field(default_factory=list)
    board: JobBoard
    seniority: str | None = Field(
        default=None,
        description="Detected seniority level: junior, mid, senior, lead, principal, staff",
    )
    posted_at: datetime | None = None
    scraped_at: datetime = Field(default_factory=datetime.now)

    model_config = {"extra": "forbid"}


class StoredJob(BaseModel):
    """A job persisted in the funnel store: the listing plus funnel metadata.

    The cross-boundary contract from ``jobs_store.JobStore`` to its callers (the
    ``status`` view, ``--from`` resolution, the TUI). ``job`` is the reconstructed
    listing; the remaining fields are funnel state the store owns.
    """

    id: int
    job: JobListing
    funnel_status: FunnelStatus = FunnelStatus.FOUND
    match_score: float | None = None
    semantic_score: float | None = None
    skill_score: float | None = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    tailored_resume_path: str = ""
    cover_letter_path: str = ""
    pdf_path: str = ""
    cover_letter_pdf_path: str = ""
    source_query: str = ""
    first_seen_at: datetime
    updated_at: datetime

    model_config = {"extra": "forbid"}


_SENIORITY_KEYWORDS: dict[str, list[str]] = {
    "intern": ["intern", "internship", "co-op"],
    "junior": ["junior", "jr", "entry level", "entry-level", "associate"],
    "mid": ["mid-level", "mid level", "intermediate"],
    "senior": ["senior", "sr", "sr."],
    "lead": ["lead", "team lead"],
    "principal": ["principal"],
    "staff": ["staff"],
    "director": ["director", "vp", "vice president"],
}


def detect_seniority(title: str, description: str = "") -> str | None:
    """Detect seniority level from job title and description.

    Returns one of: intern, junior, mid, senior, lead, principal, staff, director, None.

    The title is the strongest signal and takes precedence; the description is
    only consulted when the title is inconclusive (titles are terse and
    unambiguous, whereas descriptions are noisier).
    """
    for text in (title, description):
        text_lower = text.lower()
        for level, keywords in _SENIORITY_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                    return level
    return None


# Currency figures in a salary string, e.g. "$86,000", "$50K", "$1.5M" — the leading "$" is
# required so stray numbers (a "10%" bonus, a street number) aren't mistaken for pay.
# The K/M multiplier must sit RIGHT ON the number and not start a word — so "$1.5M" scales but
# the "M" of "$5 Main St" does not.
_SALARY_FIGURE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)([kKmM])?(?![A-Za-z])")
_SALARY_SUFFIX = {"k": 1_000, "m": 1_000_000}
# Pay-period detectors → annualization factor (US/CA work-year norms). Anchored on word
# boundaries so "day" does NOT fire on "Saturday"/"payday" (a measured 260x blow-up) while
# "/hr"-style tokens still match; the first period that matches wins.
_SALARY_PERIODS: list[tuple[tuple[str, ...], int]] = [
    ((r"\bhour\b", r"\bhourly\b", r"/hr\b", r"\bper hour\b", r"\ban hour\b"), 2080),
    ((r"\bweek\b", r"\bweekly\b", r"/wk\b", r"\bper week\b"), 52),
    ((r"\bmonth\b", r"\bmonthly\b", r"/mo\b", r"\bper month\b"), 12),
    ((r"\bday\b", r"\bdaily\b", r"\bper day\b"), 260),
]
# Below this an "annual" figure is noise (a stray "$5", a typo) rather than real pay → unknown.
_MIN_PLAUSIBLE_ANNUAL = 1_000


def parse_salary_to_annual_min(text: str | None) -> int | None:
    """Parse a free-text salary into a conservative ANNUAL minimum, for sort/filter.

    Returns the lower bound of any range, annualized by the pay period (hourly x 2080, etc.);
    a ``$50K`` figure expands to 50000 and ``$1.5M`` to 1500000. Returns ``None`` when nothing
    parseable (or only an implausibly small figure) is found — an unknown salary is never forced
    to a number, so callers treat it as "unlisted". Purely numeric: it does NOT convert
    currencies (a ca.indeed.com figure is read as-is, in CAD). Expects a salary string, not a
    full posting (it keys off ``$`` figures, so a stray "$"-amount in prose can mislead it).
    """
    if not text:
        return None
    figures: list[float] = []
    for match in _SALARY_FIGURE.finditer(text):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        suffix = match.group(2)
        if suffix:  # "K" → thousands, "M" → millions
            value *= _SALARY_SUFFIX[suffix.lower()]
        figures.append(value)
    if not figures:
        return None
    amount = min(figures)  # the range's lower bound is the conservative floor
    low = text.lower()
    for patterns, factor in _SALARY_PERIODS:
        if any(re.search(pattern, low) for pattern in patterns):
            amount *= factor
            break
    annual = int(amount)
    return annual if annual >= _MIN_PLAUSIBLE_ANNUAL else None


def coverage_measured(matched_skills: list[str], missing_skills: list[str]) -> bool:
    """Whether a match had requirements to measure skill coverage against.

    True  → ``skill_score`` is a real coverage fraction (matched / total requirements).
    False → the *semantic-only* case: the JD listed (and the extractor found) no requirements,
    so the score is semantic similarity alone and ``skill_score`` is 0.0 *by convention* — NOT
    because the candidate matched none of them. Renderers MUST NOT show that 0.0 as
    "0% of skills matched"; it means coverage was not measured. Single source of the
    semantic-only predicate, shared by the scorer (``JobMatcher._combined_score``) and the
    CLI/TUI renderers so the convention has exactly one definition."""
    return bool(matched_skills or missing_skills)


class UserProfile(BaseModel):
    """User data for form filling."""

    first_name: str
    last_name: str
    email: str
    phone: str
    location: str = ""
    linkedin_url: HttpUrl | None = None
    portfolio_url: HttpUrl | None = None
    resume_path: str = ""
    cover_letter_template: str = ""

    model_config = {"extra": "forbid"}


class ExperienceEntry(BaseModel):
    """A single work experience entry from a resume."""

    title: str = Field(default="", description="Job title")
    company: str = Field(default="", description="Company name")
    location: str = Field(default="", description="Job location")
    start_date: str = Field(default="", description="Start date (e.g. '2020' or 'January 2020')")
    end_date: str = Field(default="", description="End date (e.g. '2024' or 'Present')")
    bullets: list[str] = Field(
        default_factory=list, description="Achievement/responsibility bullets"
    )

    model_config = {"extra": "forbid"}


class EducationEntry(BaseModel):
    """A single education entry from a resume."""

    institution: str = Field(default="", description="School/university name")
    degree: str = Field(default="", description="Degree or program name")
    location: str = Field(default="", description="Institution location")
    start_date: str = Field(default="", description="Start date")
    end_date: str = Field(default="", description="End date")

    model_config = {"extra": "forbid"}


class DateEntry(BaseModel):
    """A parsed date entry from a resume audit."""

    label: str = Field(description="Entry label (e.g. job title or degree)")
    section: str = Field(description="Resume section (e.g. Experience, Education)")
    start: str = Field(description="Formatted start date")
    end: str = Field(description="Formatted end date or 'Present'")
    is_current: bool = Field(default=False, description="True if entry is ongoing")

    model_config = {"extra": "forbid"}


class ResumeData(BaseModel):
    """Parsed resume content."""

    raw_text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list, description="Cached embedding vector")
    parse_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Heuristic confidence of the parse"
    )
    parse_method: str = Field(default="", description="Parser that produced raw_text")

    model_config = {"extra": "forbid"}


class StyleGuide(BaseModel):
    """Writing style patterns extracted from example resumes/cover letters."""

    model_config = {"extra": "forbid"}

    # Core style dimensions
    tone: str = Field(description="Overall tone")
    sentence_structure: str = Field(description="Typical sentence patterns")
    vocabulary_level: str = Field(description="Vocabulary complexity")
    paragraph_style: str = Field(description="How paragraphs are structured")

    # Phrase patterns
    key_phrases: list[str] = Field(description="Frequently used phrases", default_factory=list)
    avoid_phrases: list[str] = Field(description="Phrases to avoid", default_factory=list)
    power_words: list[str] = Field(description="Strong action verbs used", default_factory=list)
    industry_jargon: list[str] = Field(description="Domain-specific terms", default_factory=list)

    # Structural patterns
    greeting_style: str = Field(default="", description="How greetings/openings are handled")
    closing_style: str = Field(default="", description="How closings/sign-offs are handled")
    use_of_metrics: str = Field(default="", description="How numbers/achievements are presented")
    storytelling_approach: str = Field(default="", description="Narrative vs bullet-point style")
    sentence_variety: str = Field(default="", description="Mix of sentence lengths and structures")
    personal_touch: str = Field(default="", description="How personality comes through")

    # Formatting
    formatting_notes: str = Field(description="Any specific formatting patterns observed")
    sample_paragraph: str = Field(description="A sample paragraph showing the style")


class DryRunValidation(BaseModel):
    """Details captured during a dry-run application (submit=False).

    Lets users verify that the automation reached the submit step and filled
    the expected fields without sending a real application.
    """

    reached_submit: bool = False
    easy_apply_button_found: bool = False
    fields_filled: list[str] = Field(default_factory=list)
    fill_errors: list[str] = Field(default_factory=list)  # fields present but could not be filled
    resume_uploaded: bool = False
    cover_letter_field_found: bool = False
    advance_steps: int = 0
    advance_selectors: list[str] = Field(default_factory=list)
    submit_selector: str = ""
    modal_title: str = ""
    required_empty_fields: list[str] = Field(default_factory=list)
    disabled_submit_reason: str = ""
    debug_artifacts: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ApplicationResult(BaseModel):
    """Outcome of an application attempt."""

    job: JobListing
    status: ApplicationStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cover_letter: str | None = None
    error_message: str | None = None
    notes: str = ""
    dry_run: DryRunValidation | None = None

    model_config = {"extra": "forbid"}


class BatchRunSpec(BaseModel):
    """Identity + parameters of a batch run — one validated payload (not loose
    scalars) across the batch-state boundary. ``run_id()`` derives the run's
    deterministic id from its identity fields, so the id and the resume-match key
    share one source and cannot drift apart.
    """

    site: str
    query: str | None = None
    jobs_file: str | None = None
    resume_path: str
    top_k: int
    min_score: float
    cover_letter: bool

    model_config = {"extra": "forbid"}

    def run_id(self) -> str:
        """Deterministic 16-char run id from the identity fields."""
        import hashlib

        key = (
            f"{self.site}|{self.query or ''}|{self.jobs_file or ''}|"
            f"{self.resume_path}|{self.top_k}|{self.min_score}|{self.cover_letter}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class ClaimCheck(BaseModel):
    """One factual claim found in a generated document, with the verifier's grounding verdict and
    the verbatim source line it cited — so the verdict can be deterministically audited."""

    model_config = {"extra": "forbid"}

    claim: str = Field(description="The factual claim, quoted from the generated document")
    grounded: bool = Field(description="True if the SOURCE résumé supports the claim")
    source_quote: str = Field(
        default="", description="Verbatim SOURCE text supporting it (empty if not grounded)"
    )
    note: str = Field(default="", description="Why it is not grounded (contradiction or silence)")


class VerificationReport(BaseModel):
    """Raw structured output of the grounding verifier — one ClaimCheck per enumerated claim."""

    model_config = {"extra": "forbid"}

    claims: list[ClaimCheck] = Field(default_factory=list)


class GroundingReport(BaseModel):
    """The AUDITED result surfaced to the user: claims the source does not support (model-flagged
    OR deterministically overridden by audit) plus sentences the verifier never enumerated
    (coverage gaps). ``complete`` is False when the enumeration missed content; ``clean`` only when
    nothing is unsupported AND coverage is complete."""

    # 'ignore' (not 'forbid') so the computed clean/complete below serialize into `tailor --json`
    # AND survive the TailoredResume.model_validate_json round-trip (cli meta reload): on load the
    # dumped computed keys are ignored and RECOMPUTED, never trusted from input. Safe because this
    # is a verifier-OUTPUT model (never user input), so 'forbid' caught no real schema-drift here.
    model_config = {"extra": "ignore"}

    unsupported: list[ClaimCheck] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def complete(self) -> bool:
        return not self.coverage_gaps

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clean(self) -> bool:
        return not self.unsupported and self.complete


class TailoredResume(BaseModel):
    """A resume tailored for a specific job, with full metadata."""

    original_path: str = Field(description="Path to original resume")
    tailored_text: str = Field(description="Full tailored resume text")
    job_title: str
    job_company: str
    job_url: str = ""
    match_score: float = Field(description="Combined match score at tailoring time")
    semantic_score: float = Field(description="Semantic similarity score")
    skill_score: float = Field(description="Skill coverage score")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    changes_summary: str = Field(description="LLM-generated summary of changes made")
    user_modifications: str = Field(default="", description="User's custom input that was applied")
    attempt: int = Field(default=1, description="Which attempt this is (1 = first)")
    prompt_version: str = Field(default="1.0", description="Prompt version used for this attempt")
    created_at: datetime = Field(default_factory=datetime.now)
    output_path: str = Field(default="", description="Path where tailored resume was saved")
    cover_letter_path: str = Field(default="", description="Path to generated cover letter, if any")
    pdf_path: str = Field(default="", description="Path to generated PDF résumé, if any")
    grounding_report: GroundingReport | None = Field(
        default=None,
        description="Honesty check of the tailored text vs the BASE résumé (None = not run or "
        "verifier unavailable). Surfaced for human review — claims are never auto-stripped.",
    )

    model_config = {"extra": "forbid"}


class TailorSession:
    """Tracks all tailoring attempts for a resume/job pair."""

    def __init__(
        self,
        original_text: str,
        job_title: str,
        job_company: str,
    ) -> None:
        self.original_text = original_text
        self.job_title = job_title
        self.job_company = job_company
        self.attempts: list[TailoredResume] = []
        self.current_index: int = -1

    def add_attempt(self, result: TailoredResume) -> None:
        """Add a new attempt and set it as current."""
        self.attempts.append(result)
        self.current_index = len(self.attempts) - 1

    @property
    def current(self) -> TailoredResume:
        """Get the currently selected attempt."""
        if not self.attempts or self.current_index < 0:
            raise IndexError("No attempts in session")
        return self.attempts[self.current_index]

    def select(self, index: int) -> None:
        """Select a previous attempt by index."""
        if index < 0 or index >= len(self.attempts):
            raise IndexError(f"Attempt index {index} out of range (0-{len(self.attempts) - 1})")
        self.current_index = index


class CoverLetterResult(BaseModel):
    """A generated cover letter with metadata."""

    job_title: str
    job_company: str
    job_url: str = ""
    cover_letter_text: str
    user_modifications: str = ""
    attempt: int = 1
    prompt_version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)
    output_path: str = ""
    pdf_path: str = Field(default="", description="Path to generated PDF cover letter, if any")

    model_config = {"extra": "forbid"}


class CoverLetterSession:
    """Tracks cover letter generation attempts."""

    def __init__(self, job_title: str, job_company: str) -> None:
        self.job_title = job_title
        self.job_company = job_company
        self.attempts: list[CoverLetterResult] = []
        self.current_index: int = -1

    def add_attempt(self, result: CoverLetterResult) -> None:
        """Add a new attempt and set it as current."""
        self.attempts.append(result)
        self.current_index = len(self.attempts) - 1

    @property
    def current(self) -> CoverLetterResult:
        """Get the currently selected attempt."""
        if not self.attempts or self.current_index < 0:
            raise IndexError("No attempts in session")
        return self.attempts[self.current_index]

    def select(self, index: int) -> None:
        """Select a previous attempt by index."""
        if index < 0 or index >= len(self.attempts):
            raise IndexError(f"Index {index} out of range (0-{len(self.attempts) - 1})")
        self.current_index = index


class DateAuditResult(BaseModel):
    """Result of auditing dates in a resume for coherence and staleness."""

    entries: list[DateEntry] = Field(
        default_factory=list,
        description="Parsed date entries with start, end, label, section",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable warnings about dates",
    )
    ordering_issues: list[str] = Field(
        default_factory=list,
        description="Entries that are out of chronological order",
    )
    staleness_issues: list[str] = Field(
        default_factory=list,
        description="Entries that suggest the CV may be outdated",
    )
    employment_gaps: list[str] = Field(
        default_factory=list,
        description="Likely employment gaps detected between experience entries",
    )
    overlap_issues: list[str] = Field(
        default_factory=list,
        description="Experience entries with overlapping date ranges",
    )
    is_stale: bool = Field(
        default=False,
        description="True if the CV appears to be significantly outdated",
    )
    is_ordered: bool = Field(
        default=True,
        description="True if entries are in correct chronological order",
    )
    latest_date: str = Field(default="", description="Most recent date found in the resume")
    earliest_date: str = Field(default="", description="Earliest date found in the resume")

    model_config = {"extra": "forbid"}


class ATSCompatibilityResult(BaseModel):
    """Result of checking a resume for ATS (Applicant Tracking System) compatibility."""

    score: float = Field(description="Overall ATS compatibility score 0.0-1.0")
    checks: list[dict[str, object]] = Field(
        default_factory=list,
        description="List of individual check results with name, passed, details",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable warnings about ATS compatibility issues",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable suggestions to improve ATS compatibility",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_compatible(self) -> bool:
        return self.score >= 0.6

    model_config = {"extra": "forbid"}


class ResumeParsingReport(BaseModel):
    source: str
    ocr_mode: str = "auto"
    text_length: int = 0
    parsed_name: str = ""
    parsed_email: str = ""
    parsed_phone: str = ""
    parsed_skills: list[str] = Field(default_factory=list)
    parsed_summary_preview: str = ""
    warnings: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class MatchReport(BaseModel):
    embedding_model: str = ""
    device: str = ""
    load_time_ms: int = 0
    job_count: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class LLMReport(BaseModel):
    model: str = ""
    endpoint: str = ""
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    temperature: float | None = None
    calls: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class TailoringReport(BaseModel):
    job_title: str = ""
    company: str = ""
    tone: str = ""
    tone_confidence: float = 0.0
    pre_match_score: float | None = None
    attempts: int = 0
    ats_before: float = 0.0
    ats_after: float = 0.0
    hallucination_actions: list[str] = Field(default_factory=list)
    changes_summary: str = ""

    model_config = {"extra": "forbid"}


class IOReport(BaseModel):
    files_written: list[str] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    batch_summary_path: str | None = None

    model_config = {"extra": "forbid"}


class VerboseReport(BaseModel):
    command: str
    args: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    duration_ms: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    resume: ResumeParsingReport | None = None
    ats: ATSCompatibilityResult | None = None
    match: MatchReport | None = None
    llm: LLMReport | None = None
    tailoring: TailoringReport | None = None
    batch_tailoring: list[TailoringReport] = Field(default_factory=list)
    io: IOReport | None = None
    errors: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class LLMEndpointCheck(BaseModel):
    """Reachability + model availability for the configured LLM endpoint."""

    api_base: str
    reachable: bool
    model_configured: str
    http_status: int | None = None
    model_available: bool = False  # configured id present in the endpoint's /models list
    models_seen: list[str] = Field(default_factory=list)
    error: str | None = None

    model_config = {"extra": "forbid"}


class VLLMProcessCheck(BaseModel):
    """Status of a local vLLM server process, if one is running."""

    running: bool = False
    pid: int | None = None
    command: str | None = None
    binary_path: str | None = None
    compatible: bool = False
    needs_restart_reason: str | None = None

    model_config = {"extra": "forbid"}


class EmbeddingsCheck(BaseModel):
    """Whether the semantic-matching embedding model is already downloaded."""

    model_name: str
    cached: bool
    cache_path: str | None = None

    model_config = {"extra": "forbid"}


class SelfHostCheck(BaseModel):
    """Optional self-hosting prerequisites (only relevant when running scripts/serve-vllm.sh)."""

    vllm_installed: bool
    hf_token_present: bool

    model_config = {"extra": "forbid"}


class BrowserCheck(BaseModel):
    """Playwright + Chromium availability for browser-based boards."""

    playwright_installed: bool
    chromium_executable: str | None = None
    channel: str | None = None  # configured [browser] channel (e.g. "chrome"); None = bundled
    host_chrome: str | None = None  # resolved host Chrome path when channel="chrome" (else None)
    error: str | None = None

    model_config = {"extra": "forbid"}


class SystemBinariesCheck(BaseModel):
    """Optional system binaries the tool can use."""

    pdftotext_available: bool
    xvfb_available: bool
    pdftotext_path: str | None = None
    xvfb_path: str | None = None

    model_config = {"extra": "forbid"}


class ConfigCheck(BaseModel):
    """config.toml presence, parseability, and security hints."""

    config_file_found: bool
    config_file_path: str | None = None
    config_file_parseable: bool = True
    plaintext_credentials: bool = False
    resume_path_set: bool = False
    resume_path_exists: bool = False
    # Résumé identity/age/parse — surfaced by `doctor` as plain INFO so a human catches a stale or
    # wrong CV (a filename that reads "Resume" not the current one, a 2-yr-old file) that no
    # threshold can. resume_sanity_note carries a soft secondary ⚠ (thin/old), never a fail.
    resume_filename: str = ""
    resume_age_days: int | None = None
    resume_parsed_skills: int | None = None
    resume_sanity_note: str = ""
    output_dir_writable: bool = False
    error: str | None = None

    model_config = {"extra": "forbid"}


class SessionHealth(BaseModel):
    """Best-effort health of an authenticated board session."""

    board: JobBoard
    healthy: bool
    details: str

    model_config = {"extra": "forbid"}


class PDFRenderingCheck(BaseModel):
    """PDF rendering toolchain health (typst package + compile smoke test)."""

    ok: bool
    message: str

    model_config = {"extra": "forbid"}


class CapabilityReadiness(BaseModel):
    """Derived readiness verdict for a user-facing capability."""

    ready: bool = False
    details: str = ""

    model_config = {"extra": "forbid"}


class DoctorReadiness(BaseModel):
    """Capability-level first-use readiness summary."""

    ai_generation: CapabilityReadiness = Field(default_factory=CapabilityReadiness)
    matching: CapabilityReadiness = Field(default_factory=CapabilityReadiness)
    browser_workflows: CapabilityReadiness = Field(default_factory=CapabilityReadiness)
    pdf_output: CapabilityReadiness = Field(default_factory=CapabilityReadiness)

    model_config = {"extra": "forbid"}


class DoctorReport(BaseModel):
    """Aggregate AI-backend health check rendered by `job-applicator doctor`."""

    llm: LLMEndpointCheck
    embeddings: EmbeddingsCheck
    self_host: SelfHostCheck
    browser: BrowserCheck
    system: SystemBinariesCheck
    config: ConfigCheck
    vllm_process: VLLMProcessCheck = Field(default_factory=VLLMProcessCheck)
    pdf_rendering: PDFRenderingCheck
    readiness: DoctorReadiness = Field(default_factory=DoctorReadiness)

    @property
    def ok(self) -> bool:
        """Blocking signal: the endpoint answered /models with HTTP 200. Connection
        failures and auth failures (401/403) are both not-ok but rendered differently.
        Model-presence and the embeddings cache are advisory only (cloud/Ollama name
        models differently; a fresh box downloads the embedder on first use).

        Browser/system/config checks are advisory: a headless server may use only the
        match/tailor pipeline and intentionally skip browser features.

        A plain property, not a computed_field, so the model still round-trips through
        model_dump()/model_validate() under extra='forbid'."""
        return self.llm.reachable and self.llm.http_status == 200

    model_config = {"extra": "forbid"}
