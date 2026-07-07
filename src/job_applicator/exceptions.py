"""Typed exception hierarchy for job applicator."""

from __future__ import annotations


class JobApplicatorError(Exception):
    """Base exception for all job applicator errors."""

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class ConfigError(JobApplicatorError):
    """Configuration loading or validation error."""


class CookieError(JobApplicatorError):
    """Cookie import/read error — unsupported browser, missing optional dependency,
    or an unreadable on-disk browser cookie store."""


class BrowserError(JobApplicatorError):
    """Browser automation error."""


class NavigationError(BrowserError):
    """Page navigation failed."""


class ElementNotFoundError(BrowserError):
    """Required element not found on page."""


class ScraperError(JobApplicatorError):
    """Job scraping error."""


class LoginRequiredError(ScraperError):
    """Scraper requires authentication."""


class RateLimitError(ScraperError):
    """Rate limit exceeded."""


class ApplicatorError(JobApplicatorError):
    """Job application submission error."""


class FormFillingError(ApplicatorError):
    """Error filling application form."""


class DocumentError(JobApplicatorError):
    """Document loading or parsing error."""


class TailorIntegrityError(DocumentError):
    """A non-interactive tailoring run produced a draft that failed integrity checks."""


class ResumeNotFoundError(DocumentError):
    """Resume file not found."""


class CoverLetterError(DocumentError):
    """Cover letter generation error."""


class CoverLetterGroundingError(CoverLetterError):
    """Cover letter grounding verification failed or found unsupported claims."""


class PDFRenderError(DocumentError):
    """Raised when PDF rendering fails."""


class LLMError(JobApplicatorError):
    """LLM API call failed.

    LLM calls back several features (cover letters, resume tailoring, style
    analysis), so this is a direct ``JobApplicatorError`` rather than a
    ``CoverLetterError`` subclass — an LLM failure during tailoring is not a
    cover-letter error and must not be caught by ``except CoverLetterError``.
    """


class GroundingUnavailableError(JobApplicatorError):
    """The semantic grounding verifier could not run (endpoint down, circuit open, bad parse).

    The fail-safe contract (spec §3 #4): a verifier failure must NEVER be masked as a clean,
    verified document. Callers catch this, fall back to the deterministic English floor, and
    surface "semantic check skipped" — never report the document as honesty-verified.
    """


class SelectorHealthError(JobApplicatorError):
    """Live selector health preflight found required selector drift."""
