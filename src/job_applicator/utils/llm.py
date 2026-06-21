"""Shared helpers for post-processing raw LLM output and hardening LLM calls."""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

from job_applicator.exceptions import LLMError
from job_applicator.utils.logging import get_logger

if TYPE_CHECKING:
    from job_applicator.config import LLMResilienceConfig

logger = get_logger("utils.llm")

T = TypeVar("T")

# Single source of truth for the "bring up a local endpoint" pointer — also rendered
# by `job-applicator doctor`, so the hint and the diagnostic never drift.
SERVE_SCRIPT = "scripts/serve-vllm.sh"

# Narrow fallback markers for a *connection* failure, used only when the exception
# isn't one of litellm's typed errors. Timeout wording is deliberately excluded: a
# timeout means the endpoint is reachable but slow, not down — classifying it as
# "unreachable" (and telling the user to start an already-running server) was a bug.
_CONNECTION_MARKERS = (
    "connection refused",
    "connection error",
    "failed to establish",
    "max retries exceeded",
    "errno 111",
    "name or service not known",
    "nodename nor servname",
)


def _connection_and_timeout_types() -> tuple[tuple[type, ...], tuple[type, ...]]:
    """litellm's own connection/timeout exception types, as (connection, timeout).

    They are siblings — ``litellm.Timeout`` descends from OpenAI's
    ``APIConnectionError``, not litellm's, so ``isinstance`` order doesn't matter.
    Returns empty tuples if litellm is unavailable (it is a core dependency)."""
    try:
        from litellm.exceptions import APIConnectionError, Timeout
    except ImportError:
        return (), ()
    return (APIConnectionError,), (Timeout,)


def llm_call_error(exc: Exception, api_base: str) -> LLMError:
    """Wrap a failed LLM call in an ``LLMError`` with an actionable hint.

    Classifies by litellm's typed exceptions first (robust), with a narrow
    connection-only string fallback. A timeout is reported as reachable-but-slow
    (never "start a server"); a genuine connection failure says how to bring one up.
    """
    conn_types, timeout_types = _connection_and_timeout_types()
    lowered = str(exc).lower()
    if timeout_types and isinstance(exc, timeout_types):
        return LLMError(
            f"The LLM endpoint at {api_base} timed out — reachable but slow or "
            f"overloaded. Check it with `job-applicator doctor`; a smaller model "
            f"responds faster. (cause: {exc})"
        )
    if (conn_types and isinstance(exc, conn_types)) or any(
        m in lowered for m in _CONNECTION_MARKERS
    ):
        return LLMError(
            f"Can't reach the LLM endpoint at {api_base}. Start one ({SERVE_SCRIPT}) "
            f"or point your llm.api_base at a running provider, then verify with "
            f"`job-applicator doctor`. (cause: {exc})"
        )
    return LLMError(f"LLM call failed: {exc}")


def strip_thinking_process(text: str | None) -> str:
    """Remove thinking process blocks from LLM output.

    Some models (like Qwen) output their reasoning before the final answer.
    This function strips that out, leaving only the clean response.

    ``text`` is typed ``str | None`` because litellm's ``message.content`` is
    optional (None on an empty/all-filtered completion); a falsy value yields ""
    so callers never hit a ``TypeError`` here.
    """
    if not text:
        return ""
    # Strategy: Find where the actual content starts
    # Letters start with "Dear", "Hello", etc.
    # Resumes start with a name (ALL CAPS) or contact info.

    # First, check if there's a thinking block
    if "Thinking Process:" in text or re.match(r"^\s*\d+\.\s+\*{2}", text):
        # Look for "Final Polish:", "Final version:", or similar markers
        final_marker_pattern = r"(?:Final\s+(?:Polish|version|draft|letter|resume|output)[:\s]*\n)"
        final_match = re.search(final_marker_pattern, text, re.IGNORECASE)

        if final_match:
            text = text[final_match.end() :]
        else:
            # Try letter openings first
            letter_pattern = r"(?:^|\n)\s*(Dear\s|Hello\s|To\s)"
            letter_match = re.search(letter_pattern, text, re.IGNORECASE)

            if letter_match:
                text = text[letter_match.start() :]
            else:
                # Look for resume-style content after thinking block
                # Pattern: name line (ALL CAPS, 2-4 words) followed by
                # contact info or section headers
                resume_pattern = (
                    r"(?:^|\n)"
                    r"(?:Here\s+(?:is|are)\s+.*?(?:resume|tailored).*?\n)?"
                    r"\s*[A-Z][A-Z\s]{5,40}\n"
                    r"(?:.*@.*\n)?"  # optional email on next line
                )
                resume_match = re.search(resume_pattern, text, re.MULTILINE)

                if resume_match:
                    text = text[resume_match.start() :]
                    # Strip leading "Here is..." intro if present
                    text = re.sub(
                        r"^Here\s+(?:is|are)\s+.*?\n",
                        "",
                        text,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                else:
                    # Last resort: find first line that looks like content
                    # (not a numbered step, not a markdown bold)
                    lines = text.split("\n")
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        # Skip thinking process lines
                        if not stripped:
                            continue
                        if re.match(r"^\d+\.\s+\*{2}", stripped):
                            continue
                        if stripped.startswith("**") and stripped.endswith("**"):
                            continue
                        if stripped.startswith("*"):
                            continue
                        if stripped.startswith("Thinking"):
                            continue
                        # Looks like actual content
                        text = "\n".join(lines[i:])
                        break

    # Clean up
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing thinking process (model may output content then think)
    # Look for patterns that indicate thinking resumed after content
    trailing_thinking_patterns = [
        r"\n\s*\*Wait,.*",  # *Wait, I need to check...*
        r"\n\s*\*Revised\s+(?:Skills|Experience|Education).*",
        r"\n\s*\*Final\s+check.*",
        r"\n\s*\*Wait,.*one more.*",
        r"\n\s*\*Drafting.*",
        r"\n\s*\*Correction.*",
        r"\n\s*\*Actually.*",
        r"\n\s*Wait,\s+I\s+need.*",
        r"\n\s*If I omit.*",
        r"\n\s*However,\s+.*invent.*",
        r"\n\s*I will check if.*",
        r"\n\s*Source text:.*",
        r"\n\s*There is no Education.*",
        r"\n\s*\*Wait, regarding.*",
        r"\n\s*\*Revised Skills:\*",
        r"\n\s*\*Final check on.*",
        r"\n\s*\(I need to.*",
        r"\n\s*\(Wait,.*",
        r"\n\s*Wait, I should check.*",
        r"\n\s*I will rewrite these.*",
        r"\n\s*Revised Bullet.*",
        r"\n\s*\*Revised Bullet.*",
    ]
    for pattern in trailing_thinking_patterns:
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            text = text[: match.start()]

    # Last resort: find last bullet point and truncate thinking after it
    # Only apply if there are bullets AND trailing text looks like thinking
    last_bullet = -1
    for i, line in enumerate(text.split("\n")):
        if line.strip().startswith(("•", "·")):
            last_bullet = i
    if last_bullet > 0:
        lines = text.split("\n")
        after_bullets = "\n".join(lines[last_bullet + 1 :])
        # Only truncate if what follows looks like thinking
        if re.search(
            r"\*Wait|I need to|I will check|Revised|Final check",
            after_bullets,
        ):
            text = "\n".join(lines[: last_bullet + 1])

    text = text.strip()

    return text


class CircuitOpenError(LLMError):
    """Raised when a call is rejected because the circuit breaker is open.

    A distinct subclass so retry layers can fail FAST on it (excluding it from
    their retryable set) — retrying a circuit-open rejection only re-hits the
    same open breaker and burns backoff before surfacing the same error.
    """


class _BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """In-memory circuit breaker for LLM calls, with a half-open probe.

    CLOSED → OPEN after ``failure_threshold`` failures within ``window_seconds``;
    stays OPEN for ``recovery_timeout_seconds``, then admits exactly ONE probe
    (HALF_OPEN). The probe succeeding closes the breaker; its failure re-opens it
    for another cooldown. Callers arriving while OPEN — or while a probe is in
    flight — get ``CircuitOpenError`` (fail fast), avoiding a thundering herd onto
    a still-down endpoint. In-memory + per-process; guards a down endpoint, not
    distributed workers.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        window_seconds: float = 60.0,
        recovery_timeout_seconds: float = 30.0,
        name: str = "llm",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.name = name
        self._failures: list[float] = []
        self._open_until: float | None = None
        self._probe_in_flight = False

    @classmethod
    def from_config(cls, resilience: LLMResilienceConfig, name: str = "llm") -> CircuitBreaker:
        """Build a breaker from the centralized resilience policy."""
        return cls(
            failure_threshold=resilience.failure_threshold,
            window_seconds=resilience.window_seconds,
            recovery_timeout_seconds=resilience.recovery_timeout_seconds,
            name=name,
        )

    def _now(self) -> float:
        return time.monotonic()

    @property
    def state(self) -> _BreakerState:
        if self._open_until is None:
            return _BreakerState.CLOSED
        if self._now() < self._open_until:
            return _BreakerState.OPEN
        return _BreakerState.HALF_OPEN

    def _admit(self) -> bool:
        """Decide whether to run this call; reserve the single half-open probe.

        Synchronous (atomic within the asyncio loop): CLOSED admits; OPEN rejects;
        HALF_OPEN admits exactly one probe and rejects the rest until it resolves.
        """
        state = self.state
        if state is _BreakerState.CLOSED:
            return True
        if state is _BreakerState.OPEN:
            return False
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def _record_failure(self) -> None:
        now = self._now()
        if self._probe_in_flight:
            # The half-open probe failed → re-open for another cooldown.
            self._probe_in_flight = False
            self._open_until = now + self.recovery_timeout_seconds
            logger.warning("Circuit breaker '%s' probe failed; re-opened", self.name)
            return
        cutoff = now - self.window_seconds
        self._failures = [t for t in self._failures if t > cutoff]
        self._failures.append(now)
        if len(self._failures) >= self.failure_threshold:
            self._open_until = now + self.recovery_timeout_seconds
            logger.warning(
                "Circuit breaker '%s' opened after %d failures in %.0fs",
                self.name,
                len(self._failures),
                self.window_seconds,
            )

    def _record_success(self) -> None:
        # Only a half-open PROBE success closes the breaker. A stray success from a
        # call admitted while CLOSED that is still in flight when the breaker OPENs
        # must NOT reopen it (that success predates the open) — clear failures only.
        if self._probe_in_flight:
            self._probe_in_flight = False
            self._open_until = None
            logger.info("Circuit breaker '%s' probe succeeded; closed", self.name)
        self._failures.clear()

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Invoke ``func`` unless the circuit is open (or a probe is in flight)."""
        if not self._admit():
            remaining = max(int((self._open_until or 0) - self._now()), 0)
            raise CircuitOpenError(
                f"LLM circuit breaker '{self.name}' is open — too many recent failures. "
                f"Wait {remaining}s or check the endpoint with `job-applicator doctor`."
            )
        try:
            result = await func()
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result


@dataclass
class LLMRuntime:
    """Per-command runtime carrying the shared LLM resilience policy.

    Created once per CLI command and shared across every LLM consumer in that
    command (cover-letter generation, résumé tailoring, …) so the circuit breaker
    spans, e.g., all jobs in a batch run — a passed context object rather than a
    module-global, per the 'no global mutable state' rule.
    """

    breaker: CircuitBreaker
    validation_max_retries: int = 1

    @classmethod
    def from_config(cls, resilience: LLMResilienceConfig, name: str = "llm") -> LLMRuntime:
        return cls(
            breaker=CircuitBreaker.from_config(resilience, name=name),
            validation_max_retries=resilience.validation_max_retries,
        )

    @classmethod
    def defaults(cls, name: str = "llm") -> LLMRuntime:
        """A runtime with built-in defaults — for standalone/library use without config."""
        return cls(breaker=CircuitBreaker(name=name))


class ValidatedOutput:
    """Retry an LLM call when its output fails a validator.

    This catches *content* problems (empty text, placeholders, malformed JSON)
    rather than transport errors. Transport errors should be handled by a retry
    decorator or circuit breaker.
    """

    def __init__(self, max_retries: int = 1) -> None:
        self.max_retries = max_retries

    async def call(
        self,
        func: Callable[[LLMError | None], Awaitable[T]],
        validator: Callable[[T], None],
    ) -> T:
        """Call ``func`` and validate; retry on validation failure, feeding the prior
        error back so the closure can re-prompt with the rejection.

        ``func`` receives the previous validation error (None on the first attempt).
        ``await func(...)`` stays OUTSIDE the try — only the validator is wrapped — so
        transport/circuit errors propagate un-fed-back (content vs transport stay split).
        """
        last_error: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            result = await func(last_error)
            try:
                validator(result)
                return result
            except LLMError as exc:
                last_error = exc
                logger.warning("Output validation failed (attempt %d): %s", attempt + 1, exc)
        raise last_error or LLMError("LLM output validation failed after retries")
