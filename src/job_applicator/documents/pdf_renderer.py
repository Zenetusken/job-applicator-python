"""PDF rendering for tailored résumés and cover letters via Typst."""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast

from jinja2 import Environment, FileSystemLoader, PackageLoader

from job_applicator.config import AppSettings
from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedResume,
)
from job_applicator.documents.job_category import detect_job_category
from job_applicator.exceptions import LLMError, PDFRenderError
from job_applicator.models import CoverLetterResult, JobListing, TailoredResume
from job_applicator.utils.llm import quiet_litellm

# Characters that must be escaped when they appear unescaped in Typst source.
# Backslash and slash are handled separately because they participate in escape
# sequences and comments.
_SIMPLE_METACHARS = frozenset('#_*$"`{}[]<>\n\r@')


RESUME_SYSTEM_PROMPT = (
    "You are a résumé formatter. Given a tailored plain-text résumé and optional job "
    "details, emit a structured JSON object matching the FormattedResume schema exactly. "
    "Do not invent contact information; omit fields you cannot verify."
)

COVER_LETTER_SYSTEM_PROMPT = (
    "You are a cover-letter formatter. Given a cover letter text, split it into greeting, "
    "body paragraphs, closing, and signature. Emit a structured JSON object matching the "
    "FormattedCoverLetter schema exactly."
)


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


def _build_resume_format_prompt(
    tailored: TailoredResume, job: JobListing | None, category: str
) -> str:
    job_text = (
        f"Title: {job.title}\nCompany: {job.company}\nDescription: {job.description}"
        if job
        else "No job provided."
    )
    return (
        f"Job category: {category}\n\n"
        f"{job_text}\n\n"
        f"Tailored résumé text:\n{tailored.tailored_text}\n\n"
        "Return a FormattedResume JSON object."
    )


def _build_cover_letter_format_prompt(
    result: CoverLetterResult, job: JobListing | None, category: str
) -> str:
    job_text = f"Title: {job.title}\nCompany: {job.company}" if job else "No job provided."
    return (
        f"Job category: {category}\n\n"
        f"{job_text}\n\n"
        f"Cover letter text:\n{result.cover_letter_text}\n\n"
        "Return a FormattedCoverLetter JSON object."
    )


class PDFRenderer:
    """Render tailored résumés and cover letters to PDF via Typst."""

    _executor: ClassVar[ProcessPoolExecutor | None] = None

    def __init__(
        self,
        settings: AppSettings,
        template_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.output_dir = output_dir or Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._env = typst_template_env(template_dir)
        self._client: Any | None = None

    @classmethod
    def _get_executor(cls) -> ProcessPoolExecutor:
        if cls._executor is None or getattr(cls._executor, "_processes", None) is None:
            cls._executor = ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"))
        return cls._executor

    @classmethod
    def shutdown(cls) -> None:
        """Shut down the shared process pool, if any."""
        if cls._executor is not None:
            cls._executor.shutdown(wait=True)
            cls._executor = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                quiet_litellm()
                import instructor
                from litellm import acompletion

                self._client = instructor.from_litellm(acompletion)
            except ImportError as exc:
                raise LLMError("instructor or litellm not installed") from exc
        return self._client

    async def _format_resume_with_instructor(
        self,
        tailored: TailoredResume,
        job: JobListing | None,
        category: str,
    ) -> FormattedResume:
        config = self.settings.llm
        model = f"openai/{config.model}" if config.api_base else config.model
        prompt = _build_resume_format_prompt(tailored, job, category)
        client = self._get_client()
        try:
            response = await client.create(
                model=model,
                api_base=config.api_base,
                api_key=config.api_key,
                messages=[
                    {"role": "system", "content": RESUME_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_model=FormattedResume,
                max_retries=1,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as exc:
            raise PDFRenderError(f"Failed to format resume for PDF: {exc}") from exc
        return cast(FormattedResume, response)

    async def _format_cover_letter_with_instructor(
        self,
        result: CoverLetterResult,
        job: JobListing | None,
        category: str,
    ) -> FormattedCoverLetter:
        config = self.settings.llm
        model = f"openai/{config.model}" if config.api_base else config.model
        prompt = _build_cover_letter_format_prompt(result, job, category)
        client = self._get_client()
        try:
            response = await client.create(
                model=model,
                api_base=config.api_base,
                api_key=config.api_key,
                messages=[
                    {"role": "system", "content": COVER_LETTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_model=FormattedCoverLetter,
                max_retries=1,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as exc:
            raise PDFRenderError(f"Failed to format cover letter for PDF: {exc}") from exc
        return cast(FormattedCoverLetter, response)

    async def render_resume(
        self,
        tailored: TailoredResume,
        job: JobListing | None = None,
        template: str = "modern",
        category: str | None = None,
    ) -> Path:
        if category is None:
            category = detect_job_category(job)
        formatted = await self._format_resume_with_instructor(tailored, job, category)
        return await self._render_and_compile(
            template_name=f"cv/{template}.typ",
            context={"resume": formatted},
            output_path=self._resume_output_path(tailored, template),
        )

    async def render_cover_letter(
        self,
        result: CoverLetterResult,
        job: JobListing | None = None,
        template: str = "modern",
        category: str | None = None,
    ) -> Path:
        if category is None:
            category = detect_job_category(job)
        formatted = await self._format_cover_letter_with_instructor(result, job, category)
        return await self._render_and_compile(
            template_name=f"cover_letter/{template}.typ",
            context={
                "cover_letter": formatted,
                "resume": {"name": formatted.signature, "email": ""},
            },
            output_path=self._cover_letter_output_path(result, template),
        )

    async def _render_and_compile(
        self,
        template_name: str,
        context: dict[str, Any],
        output_path: Path,
    ) -> Path:
        source_path = self.output_dir / f"_tmp_{uuid.uuid4().hex}.typ"
        rendered = self._env.get_template(template_name).render(**context)
        source_path.write_text(rendered, encoding="utf-8")
        try:
            executor = self._get_executor()
            await asyncio.get_running_loop().run_in_executor(
                executor, _compile_typst, source_path, output_path
            )
        except Exception as exc:
            raise PDFRenderError(
                f"PDF compilation failed: {exc}", {"source": str(source_path)}
            ) from exc
        else:
            source_path.unlink(missing_ok=True)
        return output_path

    def _resume_output_path(self, tailored: TailoredResume, template: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = f"tailored_{_safe(tailored.job_company)}_{_safe(tailored.job_title)}_{ts}_{template}"
        return self.output_dir / f"{base}.pdf"

    def _cover_letter_output_path(self, result: CoverLetterResult, template: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = f"cover_letter_{_safe(result.job_company)}_{_safe(result.job_title)}_{ts}_{template}"
        return self.output_dir / f"{base}.pdf"


atexit.register(PDFRenderer.shutdown)


def _compile_typst(source_path: Path, output_path: Path) -> None:
    """Compile a Typst source file to PDF.

    ``typst`` is imported inside this function so the module can be loaded even
    when the optional ``[pdf]`` extra is not installed.
    """
    import typst

    typst.compile(str(source_path), output=str(output_path), format="pdf")


def _safe(text: str) -> str:
    """Create a filesystem-safe slug from ``text``."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:30]
