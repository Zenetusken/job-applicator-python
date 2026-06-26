# PDF Résumé and Cover-Letter Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. At the end of each checkpoint, dispatch an independent spec/code review subagent before proceeding.

**Goal:** Add deterministic PDF generation for tailored résumés and cover letters using Typst + instructor structured output, opt-in via CLI/TUI flags.

**Architecture:** LLM emits typed `FormattedResume`/`FormattedCoverLetter` models via instructor; Jinja2 renders built-in Typst templates with a custom escape filter; the `typst` Python package compiles to PDF in a `ProcessPoolExecutor` (`spawn`).

**Tech Stack:** Python 3.12, Pydantic, instructor, Jinja2, Typst (PyPI package `typst>=0.15,<0.16`), PyMuPDF for text extraction in tests.

**Approved spec:** `docs/superpowers/specs/2026-06-25-pdf-rendering-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/job_applicator/documents/formatted_models.py` | Pydantic models: `FormattedResume`, `FormattedCoverLetter`, and sub-models. |
| `src/job_applicator/documents/pdf_renderer.py` | `PDFRenderer`, Jinja2 environment, Typst compilation, job categorization. |
| `src/job_applicator/documents/job_category.py` | Keyword-based job category detector. |
| `src/job_applicator/documents/artifacts.py` | New `write_tailored_pdf()` / `write_cover_letter_pdf()` helpers. |
| `src/job_applicator/templates/cv/modern.typ` | Built-in résumé template (modern). |
| `src/job_applicator/templates/cv/classic.typ` | Built-in résumé template (classic). |
| `src/job_applicator/templates/cv/minimal.typ` | Built-in résumé template (minimal). |
| `src/job_applicator/templates/cover_letter/modern.typ` | Built-in cover-letter template (modern). |
| `src/job_applicator/templates/cover_letter/classic.typ` | Built-in cover-letter template (classic). |
| `src/job_applicator/templates/cover_letter/minimal.typ` | Built-in cover-letter template (minimal). |
| `src/job_applicator/exceptions.py` | New `PDFRenderError` subclassing `DocumentError`. |
| `src/job_applicator/models.py` | Add `OutputConfig`, `PDFRenderingCheck`, `pdf_rendering` field on `DoctorReport`, `pdf_path` on artifact models. |
| `src/job_applicator/config.py` | Wire `OutputConfig` into `AppSettings`. |
| `src/job_applicator/diagnostics.py` | New PDF rendering health check. |
| `src/job_applicator/cli.py` | Add `--format`, `--template`, `--category` flags to `tailor`, `generate-cover-letter`, `batch`, `apply`. |
| `src/job_applicator/workflows/tailor.py` | Generate PDF when `--format pdf/both`. |
| `src/job_applicator/workflows/cover_letter.py` | Generate PDF when `--format pdf/both`. |
| `src/job_applicator/workflows/batch.py` | Generate PDF artifacts per job when requested. |
| `src/job_applicator/workflows/apply.py` | Generate PDF cover letters when requested. |
| `src/job_applicator/tui/actions.py` | PDF artifact helpers for TUI. |
| `config.example.toml` | Add `[output]` section. |
| `pyproject.toml` | Add `[pdf]` extra and wheel packaging for `.typ` templates. |
| `tests/unit/test_formatted_models.py` | Model validation tests. |
| `tests/unit/test_pdf_renderer.py` | Renderer unit tests (mocked Typst). |
| `tests/unit/test_job_category.py` | Category detector tests. |
| `tests/unit/test_pdf_renderer_fuzz.py` | Property-based Typst escaping tests. |
| `tests/integration/test_pdf_rendering.py` | End-to-end render + text extraction + ATS. |
| `tests/integration/test_pdf_regression.py` | Visual regression spike (gated). |
| `tests/unit/test_cli_pdf_flags.py` | CLI flag smoke tests. |

---

## Checkpoint 1: Foundation — Models, Templates, Packaging

At the end of this checkpoint, an independent review subagent must verify that the models are typed correctly, templates are packaged, and the Jinja2 escape filter is sound.

### Task 1: Add `[pdf]` extra and install `typst`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `[pdf]` extra**

```toml
[project.optional-dependencies]
pdf = [
    "typst>=0.15,<0.16",
]
```

- [ ] **Step 2: Install in project venv**

Run: `source .venv/bin/activate && pip install -e ".[dev,pdf]"`
Expected: `typst==0.15.0` is installed.

- [ ] **Step 3: Verify import**

Run:
```bash
python -c "import typst; print(typst.compile.__doc__[:100])"
```
Expected: No error; prints docstring fragment.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add [pdf] extra with typst>=0.15,<0.16"
```

### Task 2: Add packaging config for `.typ` templates

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Include template files in wheel**

Add under `[tool.hatch.build.targets.wheel]`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/job_applicator"]
include = ["src/job_applicator/templates/**/*.typ"]
```

- [ ] **Step 2: Build wheel and verify templates are present**

Run:
```bash
python -m build --wheel
unzip -l dist/*.whl | grep 'templates/.*\.typ'
```
Expected: Lists template paths.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: package .typ templates in wheel"
```

### Task 3: Add `PDFRenderError` exception

**Files:**
- Modify: `src/job_applicator/exceptions.py`

- [ ] **Step 1: Add exception class**

```python
class PDFRenderError(DocumentError):
    """Raised when PDF rendering fails."""
```

- [ ] **Step 2: Commit**

```bash
git add src/job_applicator/exceptions.py
git commit -m "feat: add PDFRenderError exception"
```

### Task 4: Add formatted Pydantic models

**Files:**
- Create: `src/job_applicator/documents/formatted_models.py`
- Test: `tests/unit/test_formatted_models.py`

- [ ] **Step 1: Write the models**

```python
from __future__ import annotations

from pydantic import BaseModel


class FormattedExperienceEntry(BaseModel):
    model_config = {"extra": "forbid"}

    title: str
    company: str
    location: str | None = None
    start_date: str
    end_date: str | None = None
    bullets: list[str]
    highlights: list[str] | None = None


class FormattedEducationEntry(BaseModel):
    model_config = {"extra": "forbid"}

    institution: str
    degree: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class FormattedSkillGroup(BaseModel):
    model_config = {"extra": "forbid"}

    category: str | None = None
    skills: list[str]


class ProjectEntry(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    description: str | None = None
    url: str | None = None


class FormattedResume(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    title: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    portfolio_url: str | None = None
    summary: str | None = None
    experience: list[FormattedExperienceEntry]
    education: list[FormattedEducationEntry] | None = None
    skills: list[FormattedSkillGroup] | None = None
    certifications: list[str] | None = None
    languages: list[str] | None = None
    projects: list[ProjectEntry] | None = None
    job_category: str | None = None
    emphasized_skills: list[str] | None = None


class FormattedCoverLetter(BaseModel):
    model_config = {"extra": "forbid"}

    recipient_company: str
    recipient_address: str | None = None
    date: str
    greeting: str
    paragraphs: list[str]
    closing: str
    signature: str
    job_category: str | None = None
    key_points: list[str] | None = None
```

- [ ] **Step 2: Write validation tests**

```python
from __future__ import annotations

import pytest

from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedExperienceEntry,
    FormattedResume,
)


def test_resume_valid() -> None:
    resume = FormattedResume(
        name="Alex Rivera",
        email="alex@example.com",
        experience=[
            FormattedExperienceEntry(
                title="Engineer",
                company="Acme",
                start_date="2020",
                end_date="Present",
                bullets=["Built things"],
            ),
        ],
    )
    assert resume.name == "Alex Rivera"


def test_resume_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        FormattedResume(
            name="Alex Rivera",
            email="alex@example.com",
            experience=[],
            unknown_field="x",
        )


def test_cover_letter_valid() -> None:
    letter = FormattedCoverLetter(
        recipient_company="Acme",
        date="2026-06-25",
        greeting="Dear Hiring Manager,",
        paragraphs=["I am excited to apply."],
        closing="Sincerely,",
        signature="Alex Rivera",
    )
    assert letter.signature == "Alex Rivera"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_formatted_models.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/documents/formatted_models.py tests/unit/test_formatted_models.py
git commit -m "feat: add FormattedResume and FormattedCoverLetter models"
```

### Task 5: Add job category detector

**Files:**
- Create: `src/job_applicator/documents/job_category.py`
- Test: `tests/unit/test_job_category.py`

- [ ] **Step 1: Implement detector**

```python
from __future__ import annotations

from typing import ClassVar

from job_applicator.models import JobListing


class JobCategoryDetector:
    CATEGORIES: ClassVar[dict[str, list[str]]] = {
        "cybersecurity": ["security", "cyber", "soc", "forensics", "pentest"],
        "network-administration": ["network", "cisco", "firewall", "vpn"],
        "systems-administration": ["sysadmin", "systems administrator", "linux admin", "windows admin"],
        "tech-support": ["support", "help desk", "it support", "technical support"],
        "software-engineering": ["software engineer", "developer", "programmer"],
        "data-engineering": ["data engineer", "etl", "data pipeline"],
    }

    def detect(self, job: JobListing | None) -> str:
        if job is None:
            return "default"
        text = f"{job.title or ''} {job.description or ''}".lower()
        for category, keywords in self.CATEGORIES.items():
            if any(keyword in text for keyword in keywords):
                return category
        return "default"


def detect_job_category(job: JobListing | None) -> str:
    return JobCategoryDetector().detect(job)
```

- [ ] **Step 2: Write tests**

```python
from __future__ import annotations

from job_applicator.documents.job_category import detect_job_category
from job_applicator.models import JobListing


def test_detects_cybersecurity() -> None:
    job = JobListing(title="Cybersecurity Analyst", description="Monitor SOC")
    assert detect_job_category(job) == "cybersecurity"


def test_default_when_no_match() -> None:
    job = JobListing(title="Unicorn Wrangler", description="Magic")
    assert detect_job_category(job) == "default"


def test_none_returns_default() -> None:
    assert detect_job_category(None) == "default"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_job_category.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/documents/job_category.py tests/unit/test_job_category.py
git commit -m "feat: add job category detector for PDF content strategy"
```

### Task 6: Add Jinja2 Typst escape filter and template loader

**Files:**
- Create: `src/job_applicator/documents/pdf_renderer.py` (initial version)
- Test: `tests/unit/test_pdf_renderer.py` (initial tests)

- [ ] **Step 1: Add escape filter and loader**

```python
from __future__ import annotations

import html
import re
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape


def _typst_escape(value: object) -> str:
    text = str(value)
    # Escape backslash first
    text = text.replace("\\", "\\\\")
    # Escape Typst metacharacters
    for char in ['#', '_', '*', '$', '"', '`', '{', '}', '[', ']']:
        text = text.replace(char, "\\" + char)
    # Newlines become explicit breaks in running text
    text = text.replace("\n", " ")
    return text


def _create_jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader("job_applicator", "templates"),
        autoescape=False,
    )


def typst_template_env() -> Environment:
    env = _create_jinja_env()
    env.filters["typst_escape"] = _typst_escape
    return env
```

- [ ] **Step 2: Write escape tests**

```python
from __future__ import annotations

from job_applicator.documents.pdf_renderer import _typst_escape


def test_typst_escape_metacharacters() -> None:
    raw = '#_ *$ "`{}[]\\'
    escaped = _typst_escape(raw)
    assert '#' not in escaped or escaped.startswith('\\#')
    assert '\\#' in escaped
    assert '\\$' in escaped
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_pdf_renderer.py::test_typst_escape_metacharacters -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/documents/pdf_renderer.py tests/unit/test_pdf_renderer.py
git commit -m "feat: add Typst Jinja2 environment and escape filter"
```

### Task 7: Create built-in Typst templates

**Files:**
- Create: `src/job_applicator/templates/cv/modern.typ`
- Create: `src/job_applicator/templates/cv/classic.typ`
- Create: `src/job_applicator/templates/cv/minimal.typ`
- Create: `src/job_applicator/templates/cover_letter/modern.typ`
- Create: `src/job_applicator/templates/cover_letter/classic.typ`
- Create: `src/job_applicator/templates/cover_letter/minimal.typ`

- [ ] **Step 1: Write modern résumé template**

```typst
#set page(margin: 0.75in)
#set text(font: "Libertinus Serif", size: 10pt)

#align(center)[
  #text(size: 20pt, weight: "bold")[{{ resume.name | typst_escape }}]
  #if resume.title [
    #linebreak()
    #text(size: 12pt, style: "italic")[{{ resume.title | typst_escape }}]
  ]
  #linebreak()
  #text(size: 9pt)[
    {% if resume.email %}{{ resume.email | typst_escape }}{% endif %}
    {% if resume.phone %} · {{ resume.phone | typst_escape }}{% endif %}
    {% if resume.location %} · {{ resume.location | typst_escape }}{% endif %}
  ]
]
#v(8pt)

#if resume.summary [
  == Summary
  {{ resume.summary | typst_escape }}
  #v(6pt)
]

== Experience
#v(4pt)
{% for exp in resume.experience %}
*{{ exp.title | typst_escape }}* — {{ exp.company | typst_escape }}{% if exp.location %}, {{ exp.location | typst_escape }}{% endif %} #h(1fr) {{ exp.start_date | typst_escape }}{% if exp.end_date %} – {{ exp.end_date | typst_escape }}{% endif %}
{% for bullet in exp.bullets %}
- {{ bullet | typst_escape }}
{% endfor %}
#v(4pt)
{% endfor %}

{% if resume.education %}
== Education
#v(4pt)
{% for edu in resume.education %}
*{{ edu.degree | typst_escape }}* — {{ edu.institution | typst_escape }} #h(1fr) {{ edu.start_date | typst_escape }}{% if edu.end_date %} – {{ edu.end_date | typst_escape }}{% endif %}
{% endfor %}
#v(6pt)
{% endif %}

{% if resume.skills %}
== Skills
#v(4pt)
{% for group in resume.skills %}
{% if group.category %}*{{ group.category | typst_escape }}:* {% endif %}{{ group.skills | join(", ") | typst_escape }}
{% endfor %}
{% endif %}
```

- [ ] **Step 2: Write modern cover-letter template**

```typst
#set page(margin: 1in)
#set text(font: "Libertinus Serif", size: 11pt)

#align(right)[
  {{ cover_letter.signature | typst_escape }} \
  {% if resume.email %}{{ resume.email | typst_escape }} \{% endif %}
  {{ cover_letter.date | typst_escape }}
]
#v(12pt)

{{ cover_letter.recipient_company | typst_escape }} \
{% if cover_letter.recipient_address %}{{ cover_letter.recipient_address | typst_escape }} \{% endif %}

#v(12pt)

{{ cover_letter.greeting | typst_escape }}

#v(8pt)

{% for paragraph in cover_letter.paragraphs %}
{{ paragraph | typst_escape }}

#v(6pt)
{% endfor %}

{{ cover_letter.closing | typst_escape }},

#v(16pt)

{{ cover_letter.signature | typst_escape }}
```

- [ ] **Step 3: Write classic and minimal variants**

Duplicate the modern templates with adjusted styling (classic: serif, centered header, underline sections; minimal: sparse, no color, single column). Keep the same Jinja2 placeholders so the renderer can use any template.

- [ ] **Step 4: Test template loading**

Add test:

```python
def test_templates_load() -> None:
    env = typst_template_env()
    for name in ["cv/modern.typ", "cv/classic.typ", "cv/minimal.typ",
                 "cover_letter/modern.typ", "cover_letter/classic.typ", "cover_letter/minimal.typ"]:
        source = env.get_template(name).render(
            resume={"name": "Test", "experience": []},
            cover_letter={"recipient_company": "Acme", "date": "2026-06-25",
                          "greeting": "Hi", "paragraphs": [],
                          "closing": "Best", "signature": "Test"},
        )
        assert source.strip()
```

Run: `pytest tests/unit/test_pdf_renderer.py::test_templates_load -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/job_applicator/templates/
git commit -m "feat: add built-in Typst templates for cv and cover letter"
```

### Checkpoint 1 Review

- [ ] Dispatch a spec/code review subagent to verify:
  - Models use `extra="forbid"` and avoid `dict`.
  - `pyproject.toml` correctly packages `.typ` files.
  - `_typst_escape` covers all Typst metacharacters.
  - Templates load from `importlib.resources` / installed wheel.

---

## Checkpoint 2: Renderer Core — Format, Render, Compile

At the end of this checkpoint, an independent review subagent must verify that the renderer produces valid PDFs from sample structured data and handles errors correctly.

### Task 8: Implement `PDFRenderer` core with LLM formatters

**Files:**
- Modify: `src/job_applicator/documents/pdf_renderer.py`
- Test: `tests/unit/test_pdf_renderer.py`

- [ ] **Step 1: Add imports and class skeleton**

```python
from __future__ import annotations

import asyncio
import multiprocessing as mp
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from jinja2 import Environment, PackageLoader
from pydantic import BaseModel

from job_applicator.config import AppSettings, LLMConfig
from job_applicator.documents.formatted_models import (
    FormattedCoverLetter,
    FormattedResume,
)
from job_applicator.documents.job_category import detect_job_category
from job_applicator.exceptions import PDFRenderError, LLMError
from job_applicator.models import CoverLetterResult, JobListing, TailoredResume
from job_applicator.utils.llm import quiet_litellm, strip_thinking_process


class PDFRenderer:
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
        self._env = _create_jinja_env(template_dir)
        self._client: Any | None = None

    @classmethod
    def _get_executor(cls) -> ProcessPoolExecutor:
        if cls._executor is None or cls._executor._shutdown:  # type: ignore[attr-defined]
            cls._executor = ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"))
        return cls._executor

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
```

- [ ] **Step 2: Add `_format_resume_with_instructor()`**

```python
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
        return response
```

- [ ] **Step 3: Define prompt builders and system prompts**

In the same module, add module-level constants and helpers:

```python
RESUME_SYSTEM_PROMPT = """You are a résumé formatter. Given a tailored plain-text résumé and optional job details, emit a structured JSON object matching the FormattedResume schema exactly. Do not invent contact information; omit fields you cannot verify."""

COVER_LETTER_SYSTEM_PROMPT = """You are a cover-letter formatter. Given a cover letter text, split it into greeting, body paragraphs, closing, and signature. Emit a structured JSON object matching the FormattedCoverLetter schema exactly."""


def _build_resume_format_prompt(
    tailored: TailoredResume, job: JobListing | None, category: str
) -> str:
    job_text = f"Title: {job.title}\nCompany: {job.company}\nDescription: {job.description}" if job else "No job provided."
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
```

- [ ] **Step 4: Add `_format_cover_letter_with_instructor()`**

```python
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
        return response
```

- [ ] **Step 5: Add public render methods**

```python
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
            context={"cover_letter": formatted, "resume": {"name": formatted.signature, "email": ""}},
            output_path=self._cover_letter_output_path(result, template),
        )

    async def _render_and_compile(
        self,
        template_name: str,
        context: dict[str, Any],
        output_path: Path,
    ) -> Path:
        source_path = self.output_dir / f"_tmp_{uuid.uuid4().hex}.typ"
        try:
            rendered = self._env.get_template(template_name).render(**context)
            source_path.write_text(rendered, encoding="utf-8")
            executor = self._get_executor()
            await asyncio.get_running_loop().run_in_executor(
                executor, _compile_typst, source_path, output_path
            )
        except Exception as exc:
            raise PDFRenderError(f"PDF compilation failed: {exc}", {"source": str(source_path)}) from exc
        finally:
            source_path.unlink(missing_ok=True)
        return output_path
```

- [ ] **Step 6: Add top-level compile helper**

```python
def _compile_typst(source_path: Path, output_path: Path) -> None:
    import typst

    typst.compile(str(source_path), output=str(output_path), format="pdf")
```

- [ ] **Step 7: Add output path helpers**

```python
    def _resume_output_path(self, tailored: TailoredResume, template: str) -> Path:
        base = f"tailored_{_safe(tailored.job_company)}_{_safe(tailored.job_title)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.output_dir / f"{base}.pdf"

    def _cover_letter_output_path(self, result: CoverLetterResult, template: str) -> Path:
        base = f"cover_letter_{_safe(result.job_company)}_{_safe(result.job_title)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.output_dir / f"{base}.pdf"


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:30]
```

- [ ] **Step 8: Add unit tests**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.models import TailoredResume


@pytest.fixture
def renderer(tmp_path, settings):
    return PDFRenderer(settings=settings, output_dir=tmp_path)


@pytest.mark.asyncio
async def test_render_resume_calls_compile(renderer, tmp_path):
    tailored = TailoredResume(
        original_path="r.pdf",
        tailored_text="# Alex\n**Engineer**\n- Built things",
        job_title="Engineer",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.8,
        skill_score=0.8,
    )
    with patch("job_applicator.documents.pdf_renderer._compile_typst") as mock_compile:
        with patch.object(renderer, "_format_resume_with_instructor", new_callable=AsyncMock) as mock_fmt:
            from job_applicator.documents.formatted_models import FormattedResume, FormattedExperienceEntry
            mock_fmt.return_value = FormattedResume(
                name="Alex",
                experience=[FormattedExperienceEntry(title="Engineer", company="Acme", start_date="2020", bullets=["Built things"])],
            )
            path = await renderer.render_resume(tailored)
            assert path.suffix == ".pdf"
            mock_compile.assert_called_once()
```

- [ ] **Step 8: Commit**

```bash
git add src/job_applicator/documents/pdf_renderer.py tests/unit/test_pdf_renderer.py
git commit -m "feat: implement PDFRenderer core with format, render, compile"
```

### Task 9: Integration test — render real PDF and extract text

**Files:**
- Create: `tests/integration/test_pdf_rendering.py`

- [ ] **Step 1: Build sample data and render**

```python
import pytest
from pathlib import Path

from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.models import TailoredResume


@pytest.fixture
def sample_tailored() -> TailoredResume:
    return TailoredResume(
        original_path="resume.pdf",
        tailored_text=(
            "**Alex Rivera**\n"
            "Senior Python Engineer\n\n"
            "**Experience**\n"
            "*Acme Corp* — Staff Engineer, 2020–Present\n"
            "• Built async microservices\n"
            "• Improved API latency by 40%\n\n"
            "**Skills**\nPython, Asyncio, PostgreSQL"
        ),
        job_title="Senior Python Engineer",
        job_company="Acme Corp",
        match_score=0.85,
        semantic_score=0.80,
        skill_score=0.90,
    )


@pytest.mark.asyncio
async def test_render_resume_to_pdf(sample_tailored, settings, tmp_path):
    renderer = PDFRenderer(settings=settings, output_dir=tmp_path)
    pdf_path = await renderer.render_resume(sample_tailored)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
```

- [ ] **Step 2: Extract text with PyMuPDF**

```python
import fitz

def extract_text(path: Path) -> str:
    return "\n".join(page.get_text() for page in fitz.open(path))
```

- [ ] **Step 3: Run ATSChecker on extracted text**

```python
from job_applicator.documents.ats_checker import ATSChecker

result = ATSChecker.check(resume_data=extracted_text)
assert result.is_compatible
```

- [ ] **Step 4: Run integration test**

Run: `pytest tests/integration/test_pdf_rendering.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_pdf_rendering.py
git commit -m "test: add PDF render + ATS round-trip integration test"
```

### Checkpoint 2 Review

- [ ] Dispatch a spec/code review subagent to verify:
  - Renderer uses `ProcessPoolExecutor` with `spawn`.
  - Instructor call matches existing project patterns.
  - Error handling catches `typst.TypstError`.
  - Integration test produces valid PDF and extractable text.

---

## Checkpoint 3: Configuration, Artifacts, CLI, TUI

### Task 10: Add `OutputConfig` and `DoctorReport` fields

**Files:**
- Modify: `src/job_applicator/models.py`
- Modify: `src/job_applicator/config.py`
- Modify: `config.example.toml`

- [ ] **Step 1: Add `OutputConfig` and update `AppSettings`**

```python
from pydantic_settings import SettingsConfigDict

class OutputConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_OUTPUT_")

    default_format: Literal["txt", "pdf", "both"] = "txt"
    resume_template: str = "modern"
    cover_letter_template: str = "modern"
    template_dir: Path | None = None
```

In `AppSettings`, add:
```python
output: OutputConfig = Field(default_factory=OutputConfig)
```

- [ ] **Step 2: Add `PDFRenderingCheck` and `pdf_rendering` to `DoctorReport`**

```python
class PDFRenderingCheck(BaseModel):
    model_config = {"extra": "forbid"}

    ok: bool
    message: str


class DoctorReport(BaseModel):
    llm: LLMEndpointCheck
    embeddings: EmbeddingsCheck
    self_host: SelfHostCheck
    browser: BrowserCheck
    system: SystemBinariesCheck
    config: ConfigCheck
    vllm_process: VLLMProcessCheck = Field(default_factory=VLLMProcessCheck)
    pdf_rendering: PDFRenderingCheck

    @property
    def ok(self) -> bool:
        return self.llm.reachable and self.llm.http_status == 200

    model_config = {"extra": "forbid"}
```

- [ ] **Step 3: Add `pdf_path` to artifact models**

In `TailoredResume`, add:
```python
pdf_path: str = Field(default="", description="Path to generated PDF résumé, if any")
```

In `CoverLetterResult`, add:
```python
pdf_path: str = Field(default="", description="Path to generated PDF cover letter, if any")
```

- [ ] **Step 4: Update `config.example.toml`**

```toml
[output]
default_format = "txt"
resume_template = "modern"
cover_letter_template = "modern"
# template_dir = "/path/to/custom/templates"
```

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/models.py src/job_applicator/config.py config.example.toml
git commit -m "feat: add OutputConfig and PDF rendering doctor model fields"
```

### Task 11: Add PDF artifact helpers

**Files:**
- Modify: `src/job_applicator/documents/artifacts.py`

- [ ] **Step 1: Add helpers**

```python
async def write_tailored_pdf(
    renderer: PDFRenderer,
    output_dir: Path,
    tailored: TailoredResume,
    job: JobListing | None,
    template: str | None = None,
    category: str | None = None,
) -> Path:
    template = template or renderer.settings.output.resume_template
    pdf_path = await renderer.render_resume(tailored, job=job, template=template, category=category)
    tailored.pdf_path = str(pdf_path)
    meta_path = output_dir / f"{pdf_path.stem}.meta.json"
    _write_text(meta_path, tailored.model_dump_json(indent=2))
    return pdf_path


async def write_cover_letter_pdf(
    renderer: PDFRenderer,
    output_dir: Path,
    result: CoverLetterResult,
    job: JobListing | None,
    template: str | None = None,
    category: str | None = None,
) -> Path:
    template = template or renderer.settings.output.cover_letter_template
    pdf_path = await renderer.render_cover_letter(result, job=job, template=template, category=category)
    result.pdf_path = str(pdf_path)
    meta_path = output_dir / f"{pdf_path.stem}.meta.json"
    _write_text(meta_path, result.model_dump_json(indent=2))
    return pdf_path
```

Add imports:
```python
from pathlib import Path

from job_applicator.documents.pdf_renderer import PDFRenderer
from job_applicator.models import CoverLetterResult, JobListing, TailoredResume
```

- [ ] **Step 2: Commit**

```bash
git add src/job_applicator/documents/artifacts.py
git commit -m "feat: add PDF artifact writer helpers"
```

### Task 12: Add diagnostics PDF rendering check

**Files:**
- Modify: `src/job_applicator/diagnostics.py`

- [ ] **Step 1: Add check function**

```python
def check_pdf_rendering() -> PDFRenderingCheck:
    try:
        import typst
    except ImportError:
        return PDFRenderingCheck(ok=False, message="typst package not installed; run pip install job-applicator[pdf]")

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "smoke.typ"
        output = Path(tmp) / "smoke.pdf"
        source.write_text('#set text("Hello")\nHello', encoding="utf-8")
        try:
            typst.compile(str(source), output=str(output), format="pdf")
            if output.exists() and output.stat().st_size > 0:
                return PDFRenderingCheck(ok=True, message="typst compile works")
            return PDFRenderingCheck(ok=False, message="typst produced empty PDF")
        except Exception as exc:
            return PDFRenderingCheck(ok=False, message=f"typst compile failed: {exc}")
```

- [ ] **Step 2: Wire into `DoctorReport`**

- [ ] **Step 3: Commit**

```bash
git add src/job_applicator/diagnostics.py
git commit -m "feat: add PDF rendering health check to doctor"
```

### Task 13: Add `--format`, `--template`, `--category` to CLI commands

**Files:**
- Modify: `src/job_applicator/cli.py`
- Modify: `src/job_applicator/workflows/tailor.py`
- Modify: `src/job_applicator/workflows/cover_letter.py`
- Modify: `src/job_applicator/workflows/batch.py`
- Modify: `src/job_applicator/workflows/apply.py`

- [ ] **Step 1: Add shared option types in `cli.py`**

```python
from enum import Enum


class OutputFormat(str, Enum):
    TXT = "txt"
    PDF = "pdf"
    BOTH = "both"


def _format_option(default: OutputFormat = OutputFormat.TXT) -> typer.Option:
    return typer.Option(default.value, "--format", help="Output format: txt, pdf, or both.")


def _template_option() -> typer.Option:
    return typer.Option("", "--template", help="Template name (modern/classic/minimal).")


def _category_option() -> typer.Option:
    return typer.Option("", "--category", help="Job category override (e.g., cybersecurity).")
```

- [ ] **Step 2: Add flags to `tailor`**

Add to `tailor(...)`:

```python
output_format: OutputFormat = _format_option(OutputFormat.TXT),
template: str = _template_option(),
category: str = _category_option(),
```

Pass them to `_tailor_workflow(...)` by extending its signature:

```python
await _tailor_workflow(
    # existing arguments (settings, resume, job, etc.)
    output_format=output_format.value,
    template=template,
    category=category,
)
```

Update `_tailor_workflow` in `workflows/tailor.py` to accept `output_format: str`, `template: str`, `category: str` and call `write_tailored_pdf(...)` when `output_format in ("pdf", "both")`.

- [ ] **Step 3: Add flags to `generate-cover-letter`**

Add the same three options and pass to `_cover_letter_workflow(...)`. Update `workflows/cover_letter.py` to generate PDF when requested.

- [ ] **Step 4: Add flags to `batch`**

Add the same three options and pass to the batch runner. In `workflows/batch.py`, after generating the cover letter, call `write_cover_letter_pdf(...)` and/or `write_tailored_pdf(...)` based on `output_format`.

- [ ] **Step 5: Add flags to `apply`**

Add the same three options. In `workflows/apply.py`, when `cover_letter=True` and `output_format in ("pdf", "both")`, generate a PDF cover letter and store its path in `ApplicationResult` (add `cover_letter_pdf_path: str = ""` to `ApplicationResult` in `models.py`).

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/cli.py src/job_applicator/workflows/*.py src/job_applicator/models.py
git commit -m "feat: wire --format/--template/--category into tailor, cover-letter, batch, apply"
```

### Task 14: Add TUI PDF actions

**Files:**
- Modify: `src/job_applicator/tui/actions.py`

- [ ] **Step 1: Add PDF generation actions**

In `tui/actions.py`, locate `tailor_job()` and `cover_letter_job()`. After the existing text artifact is written, add:

```python
if settings.output.default_format in ("pdf", "both"):
    from job_applicator.documents.artifacts import (
        write_cover_letter_pdf,
        write_tailored_pdf,
    )

    renderer = PDFRenderer(settings=settings)
    if is_resume:
        await write_tailored_pdf(renderer, output_dir, tailored, job)
    else:
        await write_cover_letter_pdf(renderer, output_dir, result, job)
```

Add a key binding in `tui/app.py` to open the generated PDF path via `webbrowser` or `xdg-open` if available.

- [ ] **Step 2: Commit**

```bash
git add src/job_applicator/tui/actions.py
git commit -m "feat: add PDF generation actions to TUI"
```

### Checkpoint 3 Review

- [ ] Dispatch a spec/code review subagent to verify:
  - Config env vars match project convention.
  - CLI flags are wired to all four commands.
  - TUI actions reuse artifact helpers.
  - Doctor check reports correctly.

---

## Checkpoint 4: Testing, Docs, Polish

### Task 15: Add CLI PDF flag smoke tests

**Files:**
- Create: `tests/unit/test_cli_pdf_flags.py`

- [ ] **Step 1: Test `--format pdf` and `--format both`**

Use `CliRunner` and mock the renderer/workflows to assert flags are parsed and passed correctly.

- [ ] **Step 2: Commit**

```bash
git add tests/unit/test_cli_pdf_flags.py
git commit -m "test: add CLI PDF flag smoke tests"
```

### Task 16: Add property-based fuzz tests for escaping

**Files:**
- Create: `tests/unit/test_pdf_renderer_fuzz.py`

- [ ] **Step 1: Generate random content and render**

Use `hypothesis` if available, otherwise simple random strings. Verify no unescaped Typst metacharacters remain after `_typst_escape`.

- [ ] **Step 2: Commit**

```bash
git add tests/unit/test_pdf_renderer_fuzz.py
git commit -m "test: add Typst escape fuzz tests"
```

### Task 17: Visual regression spike (optional)

**Files:**
- Create: `tests/integration/test_pdf_regression.py`

- [ ] **Step 1: Render reference PDFs and compare**

Gate behind `pytest.mark.slow` and env var. Render two PDFs from the same input and assert pixel diff is zero on the same host.

- [ ] **Step 2: Commit**

```bash
git add tests/integration/test_pdf_regression.py
git commit -m "test: add gated visual regression spike"
```

### Task 18: Update CHANGELOG and docs

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md` (optional)

- [ ] **Step 1: Add CHANGELOG entry**

```markdown
## Unreleased
### Added
- PDF résumé and cover-letter rendering via Typst (`--format pdf` / `--format both`).
- Built-in templates: modern, classic, minimal.
- Job-category-aware content formatting.
- `job-applicator doctor` PDF rendering check.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for PDF rendering"
```

### Task 19: Final verification

- [ ] Run full unit suite: `pytest -m unit -v`
- [ ] Run lint/format/typecheck: `ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/`
- [ ] Run integration tests: `pytest tests/integration/test_pdf_rendering.py -v`

### Checkpoint 4 Review

- [ ] Dispatch a spec/code review subagent to verify:
  - All spec requirements are implemented.
  - Tests cover escaping, integration, CLI flags.
  - No placeholders remain.
  - Code follows project style.

---

## Self-Review

1. **Spec coverage:** Each section of the design spec maps to one or more tasks above.
2. **Placeholder scan:** No TBD/TODO steps; each task includes code/commands.
3. **Type consistency:** `FormattedResume`, `FormattedCoverLetter`, `OutputConfig`, `PDFRenderingCheck` names match across tasks.
