"""Typed exception hierarchy for job applicator."""


class JobApplicatorError(Exception):
    """Base exception for all job applicator errors."""

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class ConfigError(JobApplicatorError):
    """Configuration loading or validation error."""


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


class ResumeNotFoundError(DocumentError):
    """Resume file not found."""


class CoverLetterError(DocumentError):
    """Cover letter generation error."""


class LLMError(CoverLetterError):
    """LLM API call failed."""
