"""Shared source-only policy text for unattended résumé tailoring."""

from __future__ import annotations

STRICT_NONINTERACTIVE_INSTRUCTIONS = (
    "For non-interactive output, prioritize accuracy over embellishment. Use only facts, "
    "metrics, tools, duties, dates, employers, and outcomes explicitly present in the "
    "original résumé. Do not add new responsibilities, optional sections, aspirations, "
    "deployment claims, performance claims, collaboration claims, or outcomes. It is acceptable "
    "to make fewer changes if that is what keeps every claim source-backed. Preserve the "
    "résumé's existing name, email, phone number, and location exactly when present. In translated "
    "output, keep source-owned job titles, course names, skill/tool names, certifications, "
    "employers, and schools verbatim; do not turn a skills or coursework list into one long "
    "translated prose claim."
)

STRICT_GROUNDING_FEEDBACK = (
    "Remove every unsupported or weakly supported claim. Use only facts, metrics, tools, duties, "
    "dates, employers, and outcomes explicitly present in the original résumé. Prefer shorter "
    "source-backed bullets over embellished claims. Do not add new responsibilities, optional "
    "sections, aspirations, deployment claims, performance claims, collaboration claims, or "
    "outcomes. Preserve the résumé's existing name, email, phone number, and location exactly "
    "when present. In translated output, keep source-owned job titles, course names, skill/tool "
    "names, certifications, employers, and schools verbatim; do not turn a skills or coursework "
    "list into one long translated prose claim."
)


def source_only_instructions(user_instructions: str = "") -> str:
    """Prefix caller instructions with the shared unattended source-only policy."""
    if user_instructions:
        return f"{STRICT_NONINTERACTIVE_INSTRUCTIONS}\n\n{user_instructions}"
    return STRICT_NONINTERACTIVE_INSTRUCTIONS
