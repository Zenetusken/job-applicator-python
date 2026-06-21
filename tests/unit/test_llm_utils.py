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

    async def producer(prev: object) -> str:
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

    async def producer(prev: object) -> str:
        return "bad"

    def validator(value: str) -> None:
        raise LLMError("always bad")

    with pytest.raises(LLMError, match="always bad"):
        await ValidatedOutput(max_retries=1).call(producer, validator)


async def test_validated_output_feeds_prior_error_to_retry() -> None:
    """Cycle 2a: the retry receives the prior validation error so it can re-prompt."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import ValidatedOutput

    seen: list[object] = []

    async def producer(prev: object) -> str:
        seen.append(prev)
        return "bad" if len(seen) == 1 else "good"

    def validator(value: str) -> None:
        if value != "good":
            raise LLMError("placeholder text")

    result = await ValidatedOutput(max_retries=1).call(producer, validator)
    assert result == "good"
    assert seen[0] is None  # first attempt: no prior error
    assert isinstance(seen[1], LLMError)  # retry: fed the validation error


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


def test_cover_letter_generator_runtime_is_injectable() -> None:
    """Cycle 1: breaker is injected via an LLMRuntime context object; the default is
    built from config — no module-global remains."""
    from job_applicator.config import LLMConfig
    from job_applicator.documents import cover_letter
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.utils.llm import CircuitBreaker, LLMRuntime

    runtime = LLMRuntime(breaker=CircuitBreaker(name="test"))
    assert CoverLetterGenerator(LLMConfig(), runtime=runtime)._breaker is runtime.breaker
    # No shared module-global; the default generator builds its own from config.
    assert not hasattr(cover_letter, "_CIRCUIT_BREAKER")
    gen = CoverLetterGenerator(LLMConfig())
    assert gen._breaker is gen._runtime.breaker


def test_circuit_breaker_from_config_uses_thresholds() -> None:
    """Cycle 1b (item 3): breaker thresholds come from LLMResilienceConfig."""
    from job_applicator.config import LLMResilienceConfig
    from job_applicator.utils.llm import CircuitBreaker

    resilience = LLMResilienceConfig(
        failure_threshold=5,
        window_seconds=10.0,
        recovery_timeout_seconds=7.0,
    )
    breaker = CircuitBreaker.from_config(resilience, name="x")
    assert breaker.failure_threshold == 5
    assert breaker.window_seconds == 10.0
    assert breaker.recovery_timeout_seconds == 7.0
    assert breaker.name == "x"


def test_llm_runtime_from_config_carries_policy() -> None:
    """Cycle 1b: LLMRuntime carries the breaker AND validation_max_retries from config."""
    from job_applicator.config import LLMResilienceConfig
    from job_applicator.utils.llm import LLMRuntime

    runtime = LLMRuntime.from_config(
        LLMResilienceConfig(failure_threshold=4, validation_max_retries=3), name="x"
    )
    assert runtime.breaker.failure_threshold == 4
    assert runtime.breaker.name == "x"
    assert runtime.validation_max_retries == 3


async def test_breaker_half_open_admits_one_probe_and_closes_on_success() -> None:
    """Cycle 1 (item 2): after cooldown the breaker admits a probe; success closes it."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.0)

    async def fail() -> str:
        raise LLMError("boom")

    async def ok() -> str:
        return "ok"

    with pytest.raises(LLMError, match="boom"):
        await breaker.call(fail)  # trips OPEN; recovery 0 → HALF_OPEN immediately
    assert await breaker.call(ok) == "ok"  # the half-open probe succeeds → CLOSED
    assert await breaker.call(ok) == "ok"  # closed: normal calls flow


async def test_breaker_half_open_probe_failure_reopens() -> None:
    """Cycle 1 (item 2): a failed half-open probe re-opens the breaker (fail fast)."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitBreaker, CircuitOpenError

    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.0)

    async def fail() -> str:
        raise LLMError("boom")

    with pytest.raises(LLMError, match="boom"):
        await breaker.call(fail)  # OPEN; recovery 0 → HALF_OPEN next
    breaker.recovery_timeout_seconds = 60.0  # make the re-open observable as OPEN
    with pytest.raises(LLMError, match="boom"):
        await breaker.call(fail)  # the half-open probe; fails → re-OPEN for 60s
    with pytest.raises(CircuitOpenError):
        await breaker.call(fail)  # OPEN → rejected fast


def test_breaker_stray_success_during_open_does_not_reopen() -> None:
    """Cycle 1 (concurrency): a success from a call admitted while CLOSED must NOT
    reopen a breaker that OPENed while that call was in flight — only a half-open
    probe closes. Guards the batch Semaphore(3) interleaving."""
    from job_applicator.utils.llm import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=60.0)
    assert breaker._admit() is True  # admitted while CLOSED (in flight, not a probe)
    breaker._record_failure()  # concurrent traffic trips the breaker OPEN
    assert breaker._open_until is not None
    breaker._record_success()  # the stray in-flight call now returns successfully
    assert breaker._open_until is not None  # must STAY open for the cooldown


async def test_breaker_half_open_admits_exactly_one_concurrent_probe() -> None:
    """Cycle 1b (item 2, concurrency): after cooldown, N concurrent callers yield
    EXACTLY ONE probe; the rest fail fast. The check-and-set is synchronous, so a
    sequential test passes even on a buggy version — this gather test is the guard."""
    import asyncio

    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitBreaker, CircuitOpenError

    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.0)

    async def fail() -> str:
        raise LLMError("boom")

    with pytest.raises(LLMError, match="boom"):
        await breaker.call(fail)  # OPEN; recovery 0 → HALF_OPEN

    started = 0

    async def probe() -> str:
        nonlocal started
        started += 1
        await asyncio.sleep(0.02)  # hold the probe in flight while peers arrive
        return "ok"

    results = await asyncio.gather(*(breaker.call(probe) for _ in range(5)), return_exceptions=True)
    assert started == 1  # exactly one probe ran
    assert sum(r == "ok" for r in results) == 1
    assert sum(isinstance(r, CircuitOpenError) for r in results) == 4
