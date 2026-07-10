"""PDF rendering for tailored résumés and cover letters via Typst."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, PackageLoader, TemplateNotFound

from job_applicator.config import AppSettings
from job_applicator.documents.cover_letter import CoverLetterOutput
from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedResume,
    FormattedSourceLine,
    FormattedSourceSection,
)
from job_applicator.documents.job_category import detect_job_category
from job_applicator.documents.resume_document import ResumeDocument, canonical_resume_text
from job_applicator.documents.sign_off import extract_sign_off
from job_applicator.exceptions import PDFRenderError, TailorIntegrityError
from job_applicator.models import CoverLetterResult, JobListing, TailoredResume
from job_applicator.utils.language import detect_language
from job_applicator.utils.llm import LLMRuntime
from job_applicator.utils.path import safe_filename_slug

_FR_MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


def _localized_date(language_code: str) -> str:
    """Today's date as a letterhead string in the letter's language ('fr' -> '29 juin 2026')."""
    now = datetime.now()
    if language_code == "fr":
        return f"{now.day} {_FR_MONTHS[now.month - 1]} {now.year}"
    return now.strftime("%B %d, %Y")


# Characters that must be escaped when they appear unescaped in Typst source.
# Backslash and slash are handled separately because they participate in escape
# sequences and comments.
_SIMPLE_METACHARS = frozenset('#_*$"`{}[]<>\n\r@')

# Built-in Typst templates shipped with the package. Used for actionable
# "template not found" error messages.
_BUILTIN_TEMPLATES: tuple[str, ...] = (
    "cv/modern.typ",
    "cv/classic.typ",
    "cv/minimal.typ",
    "cover_letter/modern.typ",
    "cover_letter/classic.typ",
    "cover_letter/minimal.typ",
)

_BULLET_LINE_RE = re.compile(r"^\s*[•*+-]\s+(?P<text>.+?)\s*$")


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


class PDFRenderer:
    """Render tailored résumés and cover letters to PDF via Typst."""

    def __init__(
        self,
        settings: AppSettings,
        template_dir: Path | None = None,
        output_dir: Path | None = None,
        runtime: LLMRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.output_dir = output_dir or Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._env = typst_template_env(template_dir)
        # Kept in the signature for artifact-pipeline compatibility. Rendering no longer calls
        # an LLM, so the shared runtime is intentionally unused.
        _ = runtime

    @classmethod
    def shutdown(cls) -> None:
        """Compatibility no-op for older callers that explicitly shut down the renderer."""
        return None

    def _validate_cover_letter_sign_off(self, formatted: FormattedCoverLetter) -> None:
        """Ensure the formatted cover letter ends with a recognized sign-off."""
        closing_text = "\n".join(filter(None, [formatted.closing, formatted.signature]))
        if extract_sign_off(closing_text) is None:
            raise PDFRenderError(
                f"Cover letter closing '{formatted.closing}' is not a recognized sign-off"
            )

    @staticmethod
    def _source_line(line: str) -> FormattedSourceLine:
        if not line.strip():
            return FormattedSourceLine(text="", is_blank=True)
        bullet = _BULLET_LINE_RE.match(line)
        if bullet is not None:
            return FormattedSourceLine(text=bullet.group("text"), is_bullet=True)
        return FormattedSourceLine(text=line)

    def _format_resume(
        self,
        tailored: TailoredResume,
        category: str,
    ) -> FormattedResume:
        try:
            document = ResumeDocument.parse(tailored.tailored_text)
        except TailorIntegrityError as exc:
            raise PDFRenderError(f"Cannot render unstructured résumé text: {exc}") from exc
        preamble = [self._source_line(line) for line in document.preamble_lines]
        name = next((line.text for line in preamble if not line.is_blank), "")
        if not name:
            raise PDFRenderError("Cannot render résumé without a source preamble/name line")
        sections = [
            FormattedSourceSection(
                heading=section.heading,
                lines=[self._source_line(line) for line in section.body_lines],
            )
            for section in document.sections
        ]
        return FormattedResume(
            name=name,
            experience=[],
            job_category=category,
            source_preamble=preamble,
            source_sections=sections,
        )

    def _format_cover_letter(
        self,
        result: CoverLetterResult,
        job: JobListing | None,
        category: str,
    ) -> FormattedCoverLetter:
        text = canonical_resume_text(result.cover_letter_text)
        sign_off = extract_sign_off(text)
        if sign_off is None:
            raise PDFRenderError("Cover letter source text has no recognized sign-off")

        lines = text.splitlines()
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        if not nonempty:
            raise PDFRenderError("Cover letter source text is empty")
        last_index = nonempty[-1]
        one_line_sign_off = extract_sign_off(lines[last_index].strip()) is not None
        body_end = last_index if one_line_sign_off else nonempty[-2]
        body_text = "\n".join(lines[:body_end]).strip()
        blocks = [block.strip() for block in re.split(r"\n\s*\n", body_text) if block.strip()]
        if not blocks:
            raise PDFRenderError("Cover letter source text has no greeting or body")
        greeting_lines = blocks[0].splitlines()
        greeting = greeting_lines[0].strip()
        paragraphs = [
            re.sub(r"\s+", " ", block).strip()
            for block in (["\n".join(greeting_lines[1:])] if len(greeting_lines) > 1 else [])
            + blocks[1:]
            if block.strip()
        ]
        if not paragraphs:
            raise PDFRenderError("Cover letter source text has no body paragraphs")
        formatted = FormattedCoverLetter(
            recipient_company=job.company if job is not None else result.job_company,
            date=_localized_date(detect_language(text)),
            greeting=greeting,
            paragraphs=paragraphs,
            closing=sign_off[0].capitalize(),
            signature=sign_off[1],
            job_category=category,
        )
        self._validate_cover_letter_sign_off(formatted)
        return formatted

    @staticmethod
    def _normalize_cover_letter_input(
        cover_letter: CoverLetterResult | CoverLetterOutput,
    ) -> CoverLetterResult:
        """Normalize ``CoverLetterOutput`` to ``CoverLetterResult``."""
        if isinstance(cover_letter, CoverLetterOutput):
            return CoverLetterResult(
                cover_letter_text=cover_letter.cover_letter,
                job_title="",
                job_company="",
            )
        return cover_letter

    async def render_resume(
        self,
        tailored: TailoredResume,
        job: JobListing | None = None,
        template: str = "modern",
        category: str | None = None,
        output_path: Path | None = None,
    ) -> Path:
        if category is None:
            category = detect_job_category(job)
        formatted = self._format_resume(tailored, category)
        target = (
            output_path if output_path is not None else self._resume_output_path(tailored, template)
        )
        return await self._render_and_compile(
            template_name=f"cv/{template}.typ",
            context={"resume": formatted},
            output_path=target,
        )

    async def render_cover_letter(
        self,
        cover_letter: CoverLetterResult | CoverLetterOutput,
        job: JobListing | None = None,
        template: str = "modern",
        category: str | None = None,
        output_path: Path | None = None,
    ) -> Path:
        result = self._normalize_cover_letter_input(cover_letter)
        if category is None:
            category = detect_job_category(job)
        formatted = self._format_cover_letter(result, job, category)
        target = (
            output_path
            if output_path is not None
            else self._cover_letter_output_path(result, template)
        )
        return await self._render_and_compile(
            template_name=f"cover_letter/{template}.typ",
            context={
                "cover_letter": formatted,
                "resume": {"name": formatted.signature, "email": ""},
            },
            output_path=target,
        )

    async def _render_and_compile(
        self,
        template_name: str,
        context: dict[str, Any],
        output_path: Path,
    ) -> Path:
        source_path = self.output_dir / f"_tmp_{uuid.uuid4().hex}.typ"
        try:
            template = self._env.get_template(template_name)
        except TemplateNotFound as exc:
            raise PDFRenderError(
                f"Template not found: {template_name}. "
                f"Built-in templates: {', '.join(_BUILTIN_TEMPLATES)}"
            ) from exc
        rendered = template.render(**context)
        try:
            source_path.write_text(rendered, encoding="utf-8")
        except Exception as exc:
            source_path.unlink(missing_ok=True)
            raise PDFRenderError(f"Failed to write Typst source: {exc}") from exc
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            source_path.unlink(missing_ok=True)
            raise PDFRenderError(f"Failed to create output directory: {exc}") from exc
        try:
            await self._compile_pdf(source_path, output_path)
        except Exception as exc:
            raise PDFRenderError(
                f"PDF compilation failed: {exc}", {"source": str(source_path)}
            ) from exc
        else:
            source_path.unlink(missing_ok=True)
        return output_path

    async def _compile_pdf(self, source_path: Path, output_path: Path) -> None:
        _compile_typst(source_path, output_path)

    def _resume_output_path(self, tailored: TailoredResume, template: str) -> Path:
        now = datetime.now()
        company = safe_filename_slug(tailored.job_company)
        title = safe_filename_slug(tailored.job_title)
        base = (
            f"tailored_{company}_{title}_{now.strftime('%Y%m%d_%H%M%S')}"
            f"_{now.microsecond:06d}_{template}"
        )
        return self.output_dir / f"{base}.pdf"

    def _cover_letter_output_path(self, result: CoverLetterResult, template: str) -> Path:
        now = datetime.now()
        company = safe_filename_slug(result.job_company)
        title = safe_filename_slug(result.job_title)
        base = (
            f"cover_letter_{company}_{title}_{now.strftime('%Y%m%d_%H%M%S')}"
            f"_{now.microsecond:06d}_{template}"
        )
        return self.output_dir / f"{base}.pdf"


def _compile_typst(source_path: Path, output_path: Path) -> None:
    """Compile a Typst source file to PDF.

    ``typst`` is imported inside this function so the module can be loaded even
    when the optional ``[pdf]`` extra is not installed. ``PDFRenderError`` is a
    normal ``JobApplicatorError`` and pickles cleanly across the spawn process
    boundary, so typed exceptions are preserved.
    """
    try:
        import typst
    except ImportError as exc:
        raise PDFRenderError(
            "typst package not installed; run: pip install 'job-applicator[pdf]'"
        ) from exc
    try:
        typst.compile(str(source_path), output=str(output_path), format="pdf")
    except Exception as exc:
        raise PDFRenderError(str(exc)) from exc
