from __future__ import annotations

import pytest

from job_applicator.documents.pdf_renderer import _typst_escape, typst_template_env


@pytest.mark.unit
def test_typst_escape_metacharacters() -> None:
    """Every Typst metacharacter, label/reference marker, and comment sequence is escaped.

    Newlines are replaced with spaces.
    """
    raw = '#_ *$"\\`{}[]\r\n<>@///*'
    escaped = _typst_escape(raw)
    assert escaped == r"\#\_ \*\$\"\\\`\{\}\[\]  \<\>\@\/\/\/\*"


@pytest.mark.unit
def test_typst_escape_plain_text_unchanged() -> None:
    assert _typst_escape("Hello, world!") == "Hello, world!"


@pytest.mark.unit
def test_typst_escape_is_idempotent() -> None:
    """Escaping an already-escaped string must not double-escape it."""
    once = _typst_escape("#_ *")
    twice = _typst_escape(once)
    assert once == twice


@pytest.mark.unit
def test_typst_escape_labels_and_references() -> None:
    """Angle brackets and at-signs used for Typst labels/references are escaped."""
    assert _typst_escape("see <intro> and email me@example.com") == (
        r"see \<intro\> and email me\@example.com"
    )


@pytest.mark.unit
def test_typst_escape_comments() -> None:
    """Comment sequences // and /* ... */ are escaped so they cannot start a comment."""
    assert _typst_escape("// not a comment") == r"\/\/ not a comment"
    assert _typst_escape("/* block */") == r"\/\* block \*/"


@pytest.mark.unit
def test_typst_template_env_has_escape_filter(tmp_path) -> None:
    """The typst_escape filter is available and finalize does not double-escape it."""
    template = tmp_path / "test.typ"
    template.write_text("{{ value | typst_escape }}")
    env = typst_template_env(tmp_path)
    result = env.get_template("test.typ").render(value="#_ *")
    assert result == r"\#\_ \*"


@pytest.mark.unit
def test_typst_template_env_finalize_escapes_by_default(tmp_path) -> None:
    """Values rendered without an explicit filter are escaped by finalize."""
    template = tmp_path / "test.typ"
    template.write_text("{{ value }}")
    env = typst_template_env(tmp_path)
    result = env.get_template("test.typ").render(value="#_ *")
    assert result == r"\#\_ \*"
