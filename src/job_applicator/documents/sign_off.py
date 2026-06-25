"""Cover-letter sign-off extraction and validation.

A generated cover letter must close with a recognized sign-off word followed by
a signature that matches the applicant's name. This module provides the
measurement and the guard so the generator can reject and retry bad sign-offs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from job_applicator.exceptions import LLMError

if TYPE_CHECKING:
    from job_applicator.models import UserProfile

# Recognized sign-off phrases. The list is intentionally conservative for
# professional cover letters. Longer phrases are listed first so the regex
# alternation matches the most specific variant.
_SIGN_OFFS: tuple[str, ...] = (
    "yours sincerely",
    "sincerely yours",
    "yours faithfully",
    "yours truly",
    "best regards",
    "warm regards",
    "kind regards",
    "with thanks",
    "thank you",
    "thanks",
    "sincerely",
    "faithfully",
    "respectfully",
    "cordially",
    "regards",
    "truly",
    "best wishes",
    "best",
)

# Pre-compute a regex that matches a sign-off line, with optional trailing
# punctuation/whitespace, so extraction stays cheap and locale-agnostic.
_SIGN_OFF_RE = re.compile(
    r"^(?P<word>" + "|".join(re.escape(w) for w in _SIGN_OFFS) + r")[,\s]*$",
    re.IGNORECASE,
)

# Same sign-off vocabulary for a single-line closing such as
# "Sincerely, John Doe".
_SINGLE_LINE_SIGN_OFF_RE = re.compile(
    r"^(?P<word>" + "|".join(re.escape(w) for w in _SIGN_OFFS) + r")[,:\s]+(?P<sig>.+?)$",
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens; keeps internal apostrophes and hyphens."""
    return re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", text.lower())


def extract_sign_off(text: str) -> tuple[str, str] | None:
    """Extract the closing word and signature from the end of a letter.

    Supports both the canonical two-line closing (``Sincerely,\\nJohn Doe``) and
    a single-line closing (``Sincerely, John Doe``). Returns ``None`` when no
    recognized sign-off is found.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    # Single-line closing on the final non-empty line.
    single_match = _SINGLE_LINE_SIGN_OFF_RE.match(lines[-1])
    if single_match:
        return single_match.group("word").lower(), single_match.group("sig").strip()

    if len(lines) < 2:
        return None

    closing_line, signature_line = lines[-2], lines[-1]
    match = _SIGN_OFF_RE.match(closing_line)
    if not match:
        return None

    return match.group("word").lower(), signature_line


def validate_sign_off(text: str, user: UserProfile) -> None:
    """Raise ``LLMError`` if the letter lacks a valid sign-off for this user.

    Validation is intentionally forgiving about the *closing word* (any common
    professional sign-off is accepted) but strict about the *signature*: when
    both first and last name are known, the signature must contain the full
    name as whole tokens. If only one part is known, that part must appear as a
    whole token. Substring matches (e.g. ``Sam`` inside ``Samantha``) are not
    accepted.
    """
    extracted = extract_sign_off(text)
    if extracted is None:
        raise LLMError("Cover letter missing a proper sign-off (e.g. 'Sincerely, <name>')")

    _closing, signature = extracted
    signature_tokens = _tokenize(signature)

    full_name = f"{user.first_name} {user.last_name}".strip()
    identifiers: list[str] = []
    if full_name:
        identifiers.append(full_name)
    elif user.first_name:
        identifiers.append(user.first_name)
    elif user.last_name:
        identifiers.append(user.last_name)

    if not identifiers:
        # Unknown applicant name — nothing to validate against.
        return

    for ident in identifiers:
        ident_tokens = _tokenize(ident)
        if ident_tokens and all(token in signature_tokens for token in ident_tokens):
            return

    display_name = full_name or user.first_name or user.last_name
    raise LLMError(
        f"Cover letter signed as '{signature}', expected a signature matching "
        f"the applicant's name ({display_name})."
    )
