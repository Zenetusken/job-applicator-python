"""Property-based fuzz tests for the Typst escaping layer.

These tests are fast and do not require the optional ``typst`` package; they only
exercise the Jinja2 escape filter and the helper that detects unescaped metacharacters.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

if pytest.importorskip("job_applicator.documents.pdf_renderer", reason="pdf module available"):
    from job_applicator.documents.pdf_renderer import (
        _has_unescaped_typst_metachar,
        _typst_escape,
    )


# Strategy: arbitrary Unicode text plus a bias toward Typst metacharacters.
_TYPST_METACHARS = r"\#_*$\"`{}[]<>@/\n\r"


@st.composite
def _dangerous_text(draw: st.DrawFn) -> str:
    """Generate strings likely to contain Typst metacharacters, including escapes."""
    chars = draw(
        st.lists(
            st.one_of(
                st.characters(min_codepoint=32, max_codepoint=126),
                st.sampled_from(list(_TYPST_METACHARS)),
                st.just("\\\\"),  # already-escaped backslash pairs
            ),
            min_size=0,
            max_size=200,
        )
    )
    return "".join(chars)


@given(value=_dangerous_text())
@settings(max_examples=500)
def test_typst_escape_never_leaves_unescaped_metachar(value: str) -> None:
    """Escaping a value must remove every unescaped Typst metacharacter."""
    escaped = _typst_escape(value)
    assert not _has_unescaped_typst_metachar(escaped)


@given(value=_dangerous_text())
@settings(max_examples=500)
def test_typst_escape_is_idempotent(value: str) -> None:
    """Escaping an already-escaped string must be a no-op."""
    once = _typst_escape(value)
    twice = _typst_escape(once)
    assert once == twice


@given(value=_dangerous_text())
@settings(max_examples=500)
def test_typst_escape_preserves_plain_ascii(value: str) -> None:
    """Strings without Typst metacharacters are returned unchanged (up to str())."""
    # Strip anything that counts as a metachar or newline.
    safe = re.sub(r"[\\#_*$\"`{}\[\]<>@/\n\r]", "", value)
    escaped = _typst_escape(safe)
    assert escaped == safe
