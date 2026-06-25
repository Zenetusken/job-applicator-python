from __future__ import annotations

import pytest

from job_applicator.documents.pdf_renderer import _typst_escape, typst_template_env


@pytest.mark.unit
def test_typst_escape_metacharacters() -> None:
    """Every Typst metacharacter is escaped and newlines become spaces."""
    raw = '#_ *$"\\`{}[]\r\n'
    escaped = _typst_escape(raw)
    assert escaped == r"\#\_ \*\$\"\\\`\{\}\[\]  "


@pytest.mark.unit
def test_typst_escape_plain_text_unchanged() -> None:
    assert _typst_escape("Hello, world!") == "Hello, world!"


@pytest.mark.unit
def test_typst_template_env_has_escape_filter(tmp_path) -> None:
    template = tmp_path / "test.typ"
    template.write_text("{{ value | typst_escape }}")
    env = typst_template_env(tmp_path)
    result = env.get_template("test.typ").render(value="#_ *")
    assert result == r"\#\_ \*"
