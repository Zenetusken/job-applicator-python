"""Shared helpers for post-processing raw LLM output."""

from __future__ import annotations

import re

from job_applicator.exceptions import LLMError

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


def strip_thinking_process(text: str) -> str:
    """Remove thinking process blocks from LLM output.

    Some models (like Qwen) output their reasoning before the final answer.
    This function strips that out, leaving only the clean response.
    """
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
