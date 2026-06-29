"""PDF rendering for tailored résumés and cover letters via Typst."""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import re
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast

from jinja2 import Environment, FileSystemLoader, PackageLoader, TemplateNotFound

from job_applicator.config import AppSettings
from job_applicator.documents.cover_letter import CoverLetterOutput
from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedResume,
)
from job_applicator.documents.job_category import detect_job_category
from job_applicator.documents.sign_off import extract_sign_off
from job_applicator.exceptions import LLMError, PDFRenderError
from job_applicator.models import CoverLetterResult, JobListing, TailoredResume
from job_applicator.utils.llm import LLMRuntime, litellm_model, quiet_litellm
from job_applicator.utils.path import safe_filename_slug

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
    _executor_lock: ClassVar[threading.Lock] = threading.Lock()

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
        self._client_lock = asyncio.Lock()
        self._llm_runtime = LLMRuntime.from_config(settings.llm_resilience, name="pdf_renderer")

    @classmethod
    def _get_executor(cls) -> ProcessPoolExecutor:
        with cls._executor_lock:
            if cls._executor is None or getattr(cls._executor, "_processes", None) is None:
                cls._executor = ProcessPoolExecutor(
                    max_workers=2, mp_context=mp.get_context("spawn")
                )
            return cls._executor

    @classmethod
    def shutdown(cls) -> None:
        """Shut down the shared process pool, if any."""
        with cls._executor_lock:
            if cls._executor is not None:
                cls._executor.shutdown(wait=True)
                cls._executor = None

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                quiet_litellm()
                import instructor
                from litellm import acompletion

                self._client = instructor.from_litellm(acompletion)
            except ImportError as exc:
                raise LLMError("instructor or litellm not installed") from exc
        return self._client

    def _validate_resume_skills(self, formatted: FormattedResume) -> None:
        """Ensure emphasized skills are actually present in the skills list.

        This is a minimal post-format guard against LLM hallucination. It does
        not re-run the full resume-tailor validator because formatting is a
        structural, not semantic, transformation.
        """
        if not formatted.emphasized_skills:
            return
        available: set[str] = set()
        for group in formatted.skills or []:
            for skill in group.skills:
                available.add(skill.lower().strip())
        missing = [
            skill for skill in formatted.emphasized_skills if skill.lower().strip() not in available
        ]
        if missing:
            raise PDFRenderError(
                f"Emphasized skills not found in skills list: {', '.join(missing)}"
            )

    def _validate_cover_letter_sign_off(self, formatted: FormattedCoverLetter) -> None:
        """Ensure the formatted cover letter ends with a recognized sign-off."""
        closing_text = "\n".join(filter(None, [formatted.closing, formatted.signature]))
        if extract_sign_off(closing_text) is None:
            raise PDFRenderError(
                f"Cover letter closing '{formatted.closing}' is not a recognized sign-off"
            )

    async def _format_resume_with_instructor(
        self,
        tailored: TailoredResume,
        job: JobListing | None,
        category: str,
    ) -> FormattedResume:
        config = self.settings.llm
        model = litellm_model(config)
        prompt = _build_resume_format_prompt(tailored, job, category)
        client = await self._get_client()

        async def _call(_prev: LLMError | None) -> FormattedResume:
            return cast(
                FormattedResume,
                await client.create(
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
                ),
            )

        try:
            response = await self._llm_runtime.run(_call)
        except Exception as exc:
            raise PDFRenderError(f"Failed to format resume for PDF: {exc}") from exc
        self._validate_resume_skills(response)
        return response

    async def _format_cover_letter_with_instructor(
        self,
        result: CoverLetterResult,
        job: JobListing | None,
        category: str,
    ) -> FormattedCoverLetter:
        # The cover-letter step honours the [cover_letter] model override (the résumé formatter
        # above stays on [llm]); the prose model and its PDF formatter use the same model.
        config = self.settings.cover_letter_llm()
        model = litellm_model(config)
        prompt = _build_cover_letter_format_prompt(result, job, category)
        client = await self._get_client()

        async def _call(_prev: LLMError | None) -> FormattedCoverLetter:
            return cast(
                FormattedCoverLetter,
                await client.create(
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
                ),
            )

        try:
            response = await self._llm_runtime.run(_call)
        except Exception as exc:
            raise PDFRenderError(f"Failed to format cover letter for PDF: {exc}") from exc
        # The LLM cannot know today's date (it hallucinates one, e.g. "2023-10-10"); set the
        # letterhead date deterministically so it is always accurate.
        response.date = datetime.now().strftime("%B %d, %Y")
        # The templates append the comma after the closing word, so strip any the LLM included
        # ("Sincerely," -> "Sincerely") to avoid a doubled comma ("Sincerely,,") in the render.
        response.closing = response.closing.rstrip(" ,")
        self._validate_cover_letter_sign_off(response)
        return response

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
        formatted = await self._format_resume_with_instructor(tailored, job, category)
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
        formatted = await self._format_cover_letter_with_instructor(result, job, category)
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


atexit.register(PDFRenderer.shutdown)


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
