"""Unit tests for LLM utilities."""

from __future__ import annotations

import pytest

from job_applicator.utils.llm import strip_thinking_process


def test_strip_thinking_process_lives_in_utils_with_back_compat_reexport() -> None:
    """L-3: the helper now lives in utils.llm; cover_letter re-exports the same object."""
    from job_applicator.documents import cover_letter
    from job_applicator.utils import llm

    assert llm.strip_thinking_process is strip_thinking_process
    assert cover_letter.strip_thinking_process is strip_thinking_process


def test_strip_thinking_process_with_thinking() -> None:
    """Test stripping thinking process from LLM output."""
    text_with_thinking = """Thinking Process:

1.  **Analyze the Request:**
    *   **Role:** Senior Python Developer.
    *   **Company:** TechCorp Solutions.

2.  **Drafting:**
    *   Paragraph 1: Opening statement.

Dear Hiring Team,

I am writing to express my interest in the position.

Sincerely,
John Doe"""

    result = strip_thinking_process(text_with_thinking)
    assert "Thinking Process" not in result
    assert "Dear Hiring Team" in result
    assert "John Doe" in result


def test_strip_thinking_process_clean_text() -> None:
    """Test that clean text passes through unchanged."""
    clean_text = """Dear Hiring Manager,

I am writing to apply for the Python Developer position.

Best regards,
Jane Smith"""

    result = strip_thinking_process(clean_text)
    assert result == clean_text


def test_strip_thinking_process_multiple_dear() -> None:
    """Test handling multiple 'Dear' occurrences (thinking + actual)."""
    text = """Thinking about the letter...

    Draft 1: Dear Team, ...

    Draft 2: Dear Hiring Manager, ...

    Final version:

    Dear Hiring Manager,

    I am excited to apply for this role.

    Sincerely,
    Applicant"""

    result = strip_thinking_process(text)
    # Should keep the last "Dear" section
    assert "Final version" not in result or "Dear Hiring Manager" in result
    assert "Sincerely" in result


def test_strip_thinking_process_empty() -> None:
    """Test handling empty input."""
    result = strip_thinking_process("")
    assert result == ""


def test_strip_thinking_process_am_writing() -> None:
    """Test stripping with 'I am writing' as letter start."""
    text = """Let me think about this...

    1. Analyze requirements
    2. Draft response

    I am writing to express my interest in the position.

    Thank you for your consideration."""

    result = strip_thinking_process(text)
    assert "I am writing" in result
    assert "Thank you" in result


async def test_circuit_breaker_passes_success() -> None:
    from job_applicator.utils.llm import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=2)

    async def ok() -> str:
        return "ok"

    result = await breaker.call(ok)
    assert result == "ok"


async def test_circuit_breaker_opens_after_failures() -> None:
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=60.0)

    async def fail() -> str:
        raise LLMError("boom")

    with pytest.raises(LLMError, match="boom"):
        await breaker.call(fail)
    with pytest.raises(LLMError, match="boom"):
        await breaker.call(fail)
    with pytest.raises(LLMError, match="circuit breaker"):
        await breaker.call(fail)


async def test_circuit_breaker_success_resets_failures() -> None:
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=60.0)

    async def fail() -> str:
        raise LLMError("boom")

    async def ok() -> str:
        return "ok"

    with pytest.raises(LLMError):
        await breaker.call(fail)
    assert await breaker.call(ok) == "ok"
    with pytest.raises(LLMError):
        await breaker.call(fail)
    # A single failure after success should NOT open the breaker.
    assert await breaker.call(ok) == "ok"


async def test_validated_output_retries_on_validation_failure() -> None:
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import ValidatedOutput

    calls = 0

    async def producer() -> str:
        nonlocal calls
        calls += 1
        return "bad" if calls == 1 else "good"

    def validator(value: str) -> None:
        if value != "good":
            raise LLMError("not good")

    result = await ValidatedOutput(max_retries=1).call(producer, validator)
    assert result == "good"
    assert calls == 2


async def test_validated_output_gives_up_after_retries() -> None:
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import ValidatedOutput

    async def producer() -> str:
        return "bad"

    def validator(value: str) -> None:
        raise LLMError("always bad")

    with pytest.raises(LLMError, match="always bad"):
        await ValidatedOutput(max_retries=1).call(producer, validator)


def test_strip_thinking_process_none_returns_empty() -> None:
    """H3: litellm ``message.content`` is Optional[str]; None must not crash."""
    assert strip_thinking_process(None) == ""


def test_circuit_open_error_is_llmerror_subclass() -> None:
    """M4: CircuitOpenError is an LLMError so existing handlers still catch it."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitOpenError

    assert issubclass(CircuitOpenError, LLMError)


async def test_breaker_open_raises_circuit_open_error() -> None:
    """M4: an open breaker raises the distinct CircuitOpenError, not a bare LLMError."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitBreaker, CircuitOpenError

    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=60.0)

    async def fail() -> str:
        raise LLMError("boom")

    with pytest.raises(LLMError, match="boom"):
        await breaker.call(fail)
    with pytest.raises(CircuitOpenError, match="circuit breaker"):
        await breaker.call(fail)


async def test_async_retry_does_not_retry_excluded() -> None:
    """M4: a CircuitOpenError must fail fast — never retried against the open breaker."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitOpenError
    from job_applicator.utils.retry import async_retry

    calls = 0

    @async_retry(
        max_attempts=3, base_delay=0.0, exceptions=(LLMError,), exclude=(CircuitOpenError,)
    )
    async def open_fast() -> str:
        nonlocal calls
        calls += 1
        raise CircuitOpenError("open")

    with pytest.raises(CircuitOpenError):
        await open_fast()
    assert calls == 1


async def test_async_retry_retries_transport_error() -> None:
    """A transport LLMError IS retried up to max_attempts (initial + retries)."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.retry import async_retry

    calls = 0

    @async_retry(max_attempts=2, base_delay=0.0, exceptions=(LLMError,))
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        raise LLMError("transport")

    with pytest.raises(LLMError, match="transport"):
        await flaky()
    assert calls == 2


def test_cover_letter_generator_breaker_is_injectable() -> None:
    """M1: breaker is injectable for tests; defaults to the process-shared instance."""
    from job_applicator.config import LLMConfig
    from job_applicator.documents import cover_letter
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.utils.llm import CircuitBreaker

    injected = CircuitBreaker(name="test")
    assert CoverLetterGenerator(LLMConfig(), breaker=injected)._breaker is injected
    # Default is the shared module-level breaker (must span jobs in a batch run).
    assert CoverLetterGenerator(LLMConfig())._breaker is cover_letter._CIRCUIT_BREAKER
