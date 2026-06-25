from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, PackageLoader


def _typst_escape(value: object) -> str:
    """Escape a value for safe interpolation into a Typst template.

    Backslash is escaped first to avoid double-escaping the other escapes.
    Newlines are replaced with spaces because Typst treats raw newlines as
    line breaks in many contexts where the caller expects a single paragraph.
    """
    text = str(value)
    # Escape backslash first so we don't double-escape later substitutions.
    text = text.replace("\\", "\\\\")
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
    # Typst source files are not HTML; escaping is handled by the typst_escape filter.
    return Environment(loader=loader, autoescape=False)  # noqa: S701


def typst_template_env(template_dir: Path | str | None = None) -> Environment:
    """Return a Jinja2 environment configured for Typst rendering."""
    env = _create_jinja_env(template_dir)
    env.filters["typst_escape"] = _typst_escape
    return env
