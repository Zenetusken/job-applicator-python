"""Shared Pydantic models — typed data contracts between layers."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, computed_field

from job_applicator.utils.text import contains_word


class JobBoard(StrEnum):
    """Supported job boards."""

    LINKEDIN = "linkedin"
    INDEED = "indeed"


class ApplicationStatus(StrEnum):
    """Outcome of an application attempt."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FAILED = "failed"
    SKIPPED = "skipped"
    ALREADY_APPLIED = "already_applied"


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
                if contains_word(text_lower, kw):
                    return level
    return None


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

    model_config = {"extra": "forbid"}


class StyleGuide(BaseModel):
    """Writing style patterns extracted from example resumes/cover letters."""

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


class ApplicationResult(BaseModel):
    """Outcome of an application attempt."""

    job: JobListing
    status: ApplicationStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    cover_letter: str | None = None
    error_message: str | None = None
    notes: str = ""

    model_config = {"extra": "forbid"}


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
