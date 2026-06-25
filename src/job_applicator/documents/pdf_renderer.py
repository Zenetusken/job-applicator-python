from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, PackageLoader

# Characters that must be escaped when they appear unescaped in Typst source.
# Backslash and slash are handled separately because they participate in escape
# sequences and comments.
_SIMPLE_METACHARS = frozenset('#_*$"`{}[]<>\n\r@')


def _has_unescaped_typst_metachar(text: str) -> bool:
    """Return True if *text* contains any Typst metacharacter that is not already escaped."""
    backslash_count = 0
    for i, ch in enumerate(text):
        if ch == "\\":
            backslash_count += 1
            continue
        escaped = backslash_count % 2 == 1
        backslash_count = 0
        if escaped:
            continue
        if ch in _SIMPLE_METACHARS:
            return True
        if ch == "/" and i + 1 < len(text) and text[i + 1] in "/*":
            return True
    # A trailing backslash with no following character is unescaped.
    return backslash_count % 2 == 1


def _typst_escape(value: object) -> str:
    """Escape a value for safe interpolation into a Typst template.

    The full escaped set is:

    * ``\\`` (backslash) — escaped first so the other escapes are not doubled.
    * ``# _ * $ " ` { } [ ] < > @`` — Typst markup/label/reference metacharacters.
    * ``//`` and ``/*`` — the leading slash is escaped so these cannot start a
      Typst comment.
    * ``\n`` and ``\r`` — replaced with spaces because Typst treats raw newlines
      as line breaks in many contexts where the caller expects a single paragraph.

    The function is idempotent: passing an already-escaped string back in returns
    it unchanged. This lets the Jinja2 ``finalize`` callback apply escaping by
    default without double-escaping values that were explicitly passed through
    the ``typst_escape`` filter.
    """
    text = str(value)
    if not _has_unescaped_typst_metachar(text):
        return text
    # Escape backslash first so we don't double-escape later substitutions.
    text = text.replace("\\", "\\\\")
    # Escape comment-starting sequences deterministically so they cannot be
    # interpreted as Typst comments.
    text = re.sub(r"//|/\*", lambda m: r"\/\/" if m.group() == "//" else r"\/" + "*", text)
    replacements = {
        "#": "\\#",
        "_": "\\_",
        "*": "\\*",
        "$": "\\$",
        '"': '\\"',
        "`": "\\`",
        "{": "\\{",
        "}": "\\}",
        "[": "\\[",
        "]": "\\]",
        "<": "\\<",
        ">": "\\>",
        "@": "\\@",
        "\n": " ",
        "\r": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _create_jinja_env(template_dir: Path | str | None = None) -> Environment:
    """Create a Jinja2 environment for Typst templates.

    By default the loader reads package templates from ``job_applicator/templates``.
    A custom directory can be supplied instead for testing or user overrides.
    """
    if template_dir is not None:
        loader: FileSystemLoader | PackageLoader = FileSystemLoader(str(template_dir))
    else:
        loader = PackageLoader("job_applicator", "templates")
    # Typst source files are not HTML; escaping is handled by the typst_escape filter
    # and by the finalize callback installed by typst_template_env().
    return Environment(loader=loader, autoescape=False)  # noqa: S701


def typst_template_env(template_dir: Path | str | None = None) -> Environment:
    """Return a Jinja2 environment configured for Typst rendering."""
    env = _create_jinja_env(template_dir)
    env.filters["typst_escape"] = _typst_escape
    env.finalize = lambda x: _typst_escape(x) if x is not None else ""
    return env
