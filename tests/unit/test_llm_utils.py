"""Unit tests for LLM utilities."""

from __future__ import annotations

from pathlib import Path

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


def test_litellm_model_adds_openai_prefix_when_api_base_set() -> None:
    """F5: OpenAI-compatible endpoints (local vLLM, …) get the ``openai/`` prefix."""
    from job_applicator.config import LLMConfig
    from job_applicator.utils.llm import litellm_model

    cfg = LLMConfig(api_base="http://localhost:8000/v1", model="cyankiwi/Qwen3.5-4B-AWQ-4bit")
    assert litellm_model(cfg) == "openai/cyankiwi/Qwen3.5-4B-AWQ-4bit"


def test_litellm_model_bare_when_no_api_base() -> None:
    """F5: with no api_base, litellm routes by provider — pass the bare id unchanged."""
    from job_applicator.config import LLMConfig
    from job_applicator.utils.llm import litellm_model

    cfg = LLMConfig(api_base="", model="gpt-4o-mini")
    assert litellm_model(cfg) == "gpt-4o-mini"


def test_openai_prefix_rule_lives_only_in_litellm_model() -> None:
    """F5 (anti-drift): the ``openai/`` prefix is constructed in exactly one place —
    the ``litellm_model()`` helper. Guards against a new completion caller
    copy-pasting the inline prefix logic and re-introducing the 6-way duplication."""
    src = Path(__file__).resolve().parents[2] / "src" / "job_applicator"
    hits = [
        f"{py.relative_to(src)}:{i}"
        for py in src.rglob("*.py")
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1)
        if 'f"openai/{' in line
    ]
    outside_helper = [h for h in hits if not h.startswith("utils/llm.py:")]
    assert not outside_helper, f"inline 'openai/' prefix must use litellm_model(): {outside_helper}"
    assert hits, "expected litellm_model() itself to still build the prefix"


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


def test_quiet_litellm_suppresses_banner_and_logs_idempotently() -> None:
    """quiet_litellm silences litellm's framework noise (banner + INFO logs) so it can't
    pollute the CLI's stdout/stderr on a successful call, and is safe to call repeatedly.
    Saves/restores the process-global litellm state so the test stays hermetic."""
    import logging

    import litellm

    from job_applicator.utils.llm import quiet_litellm

    lit_logger = logging.getLogger("LiteLLM")
    low_logger = logging.getLogger("litellm")  # distinct, case-sensitive logger
    saved = (litellm.suppress_debug_info, lit_logger.level, low_logger.level)
    try:
        quiet_litellm()
        assert litellm.suppress_debug_info is True
        assert lit_logger.level == logging.WARNING
        assert low_logger.level == logging.WARNING
        quiet_litellm()  # idempotent — must not raise
        assert litellm.suppress_debug_info is True
    finally:
        litellm.suppress_debug_info = saved[0]
        lit_logger.setLevel(saved[1])
        low_logger.setLevel(saved[2])


async def test_llm_runtime_run_returns_success() -> None:
    """LLMRuntime.run passes None as the previous error and returns the result."""
    from job_applicator.utils.llm import LLMRuntime

    runtime = LLMRuntime.defaults(name="test")

    async def producer(prev: object) -> str:
        assert prev is None
        return "ok"

    assert await runtime.run(producer) == "ok"


async def test_llm_runtime_run_validator_retry_succeeds() -> None:
    """A validation failure is retried, feeding the prior error back to the closure."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import LLMRuntime

    runtime = LLMRuntime.defaults(name="test")
    runtime.validation_max_retries = 2
    seen: list[object] = []

    async def producer(prev: object) -> str:
        seen.append(prev)
        return "good" if len(seen) > 1 else "bad"

    def validator(value: str) -> None:
        if value != "good":
            raise LLMError("not good")

    assert await runtime.run(producer, validator=validator) == "good"
    assert seen[0] is None
    assert isinstance(seen[1], LLMError)


async def test_llm_runtime_run_validator_retry_exhausted() -> None:
    """After validation_max_retries failures, the last validation error is raised."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import LLMRuntime

    runtime = LLMRuntime.defaults(name="test")
    runtime.validation_max_retries = 1

    async def producer(prev: object) -> str:
        return "bad"

    def validator(value: str) -> None:
        raise LLMError("always bad")

    with pytest.raises(LLMError, match="always bad"):
        await runtime.run(producer, validator=validator)


async def test_llm_runtime_run_circuit_breaker_open() -> None:
    """An open circuit breaker rejects the call before the closure runs."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitOpenError, LLMRuntime

    runtime = LLMRuntime.defaults(name="test")
    runtime.breaker.failure_threshold = 1
    runtime.breaker.recovery_timeout_seconds = 60.0

    async def fail(_prev: object) -> str:
        raise LLMError("boom")

    with pytest.raises(LLMError, match="boom"):
        await runtime.run(fail)
    with pytest.raises(CircuitOpenError, match="circuit breaker"):
        await runtime.run(fail)


async def test_llm_runtime_run_validator_still_uses_breaker() -> None:
    """Validator-enabled attempts still route through the circuit breaker.

    A transport error on the validator path is recorded as a breaker failure,
    and the next call is rejected with CircuitOpenError.
    """
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitOpenError, LLMRuntime

    runtime = LLMRuntime.defaults(name="test")
    runtime.validation_max_retries = 2
    runtime.breaker.failure_threshold = 1
    runtime.breaker.recovery_timeout_seconds = 60.0

    async def fail(_prev: object) -> str:
        raise LLMError("transport boom")

    def validator(value: str) -> None:
        pass

    with pytest.raises(LLMError, match="transport boom"):
        await runtime.run(fail, validator=validator)
    with pytest.raises(CircuitOpenError, match="circuit breaker"):
        await runtime.run(fail, validator=validator)


async def test_llm_runtime_run_validator_breaker_open_raises_circuit_open_error() -> None:
    """A validator-enabled call is rejected fast when the breaker is already open."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import CircuitOpenError, LLMRuntime

    runtime = LLMRuntime.defaults(name="test")
    runtime.breaker.failure_threshold = 1
    runtime.breaker.recovery_timeout_seconds = 60.0

    async def fail(_prev: object) -> str:
        raise LLMError("boom")

    # Open the breaker on the non-validator path.
    with pytest.raises(LLMError, match="boom"):
        await runtime.run(fail)

    async def producer(_prev: object) -> str:
        return "ok"

    def validator(value: str) -> None:
        pass

    with pytest.raises(CircuitOpenError, match="circuit breaker"):
        await runtime.run(producer, validator=validator)


async def test_llm_runtime_run_validator_records_transport_failure() -> None:
    """Transport errors during validator retry are recorded by the breaker."""
    from job_applicator.exceptions import LLMError
    from job_applicator.utils.llm import LLMRuntime

    runtime = LLMRuntime.defaults(name="test")
    runtime.validation_max_retries = 1
    runtime.breaker.failure_threshold = 2
    runtime.breaker.window_seconds = 60.0

    calls = 0

    async def flaky(_prev: object) -> str:
        nonlocal calls
        calls += 1
        raise LLMError("transport")

    def validator(value: str) -> None:
        pass

    with pytest.raises(LLMError, match="transport"):
        await runtime.run(flaky, validator=validator)
    assert calls == 1
    assert len(runtime.breaker._failures) == 1
