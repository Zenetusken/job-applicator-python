"""Unit tests for exceptions."""

from __future__ import annotations

from job_applicator.exceptions import (
    ApplicatorError,
    BrowserError,
    ConfigError,
    CoverLetterError,
    DocumentError,
    ElementNotFoundError,
    JobApplicatorError,
    LLMError,
    LoginRequiredError,
    NavigationError,
    RateLimitError,
    ResumeNotFoundError,
    ScraperError,
)


def test_exception_hierarchy() -> None:
    assert issubclass(ConfigError, JobApplicatorError)
    assert issubclass(BrowserError, JobApplicatorError)
    assert issubclass(NavigationError, BrowserError)
    assert issubclass(ElementNotFoundError, BrowserError)
    assert issubclass(ScraperError, JobApplicatorError)
    assert issubclass(LoginRequiredError, ScraperError)
    assert issubclass(RateLimitError, ScraperError)
    assert issubclass(ApplicatorError, JobApplicatorError)
    assert issubclass(DocumentError, JobApplicatorError)
    assert issubclass(ResumeNotFoundError, DocumentError)
    assert issubclass(CoverLetterError, DocumentError)
    # LLMError backs several features (cover letters, tailoring, style analysis),
    # so it is a direct JobApplicatorError, NOT a CoverLetterError subclass.
    assert issubclass(LLMError, JobApplicatorError)
    assert not issubclass(LLMError, CoverLetterError)


def test_exception_context() -> None:
    exc = JobApplicatorError("test error", context={"key": "value"})
    assert str(exc) == "test error"
    assert exc.context == {"key": "value"}


def test_exception_default_context() -> None:
    exc = JobApplicatorError("no context")
    assert exc.context == {}
