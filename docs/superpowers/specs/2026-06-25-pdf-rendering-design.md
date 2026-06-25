# PDF Résumé and Cover-Letter Rendering Design

**Date:** 2026-06-25  
**Status:** Draft — pending implementation plan  
**Decision:** Adopt Typst + instructor structured output, opt-in via CLI flags, job-category-aware content prompts.

## 1. Goal

Add deterministic, high-quality PDF generation for tailored résumés and cover letters. The LLM produces structured content; a template engine renders it to PDF. This avoids asking the LLM to generate markup, which historically leads to hallucinated LaTeX/HTML, broken layouts, and injection risks.

## 2. Non-Goals

- Replace the existing plain-text artifacts as the default output format.
- Build a WYSIWYG template editor.
- Support Microsoft Word or LaTeX rendering in the first version.

## 3. Context from Codebase Exploration

- Existing output is **plain text with markdown styling** (`**Skills**`, `• bullets`, `*dates*`).
- `ResumeData.experience` and `ResumeData.education` exist in the Pydantic schema but are **not populated by the parser**.
- `CoverLetterOutput` already has `extra="forbid"` and emits `cover_letter` + `key_points`.
- `TailoredResume` and `CoverLetterResult` are the current artifact models.
- `documents/artifacts.py` centralizes writing `.txt` + `.meta.json` for the TUI and should be the PDF integration point.
- `jinja2>=3.1` is already a core dependency; **Typst is not**.
- Playwright is present but the chosen design intentionally avoids HTML→PDF because typography is weaker and the toolchain has more moving parts.
- The project enforces mypy strict mode, ruff 100-char lines, double quotes, `from __future__ import annotations`, and Pydantic `extra="forbid"`.

## 4. Design Decisions

| Topic | Decision |
|-------|----------|
| Artifacts | Résumés **and** cover letters in the first version. |
| Content source | New structured Pydantic models produced via **instructor** tool-mode structured output. |
| Template engine | **Jinja2** placeholders inside **Typst** source files. |
| Renderer backend | **Python `typst` package** from PyPI (in-process, no subprocess fallback). |
| User workflow | Opt-in via `--format pdf` / `--format both` on commands; default remains plain text. |
| Built-in templates | `modern`, `classic`, `minimal`. |
| Job category tailoring | Same visual templates; the LLM content prompt adapts to categories such as Tech Support, Cybersecurity, Systems Administration, Network Administration, etc. |
| Apply workflow | PDF cover letters are generated during `apply --format pdf` (both dry-run and `--submit`). |
| Testing | ATS round-trip, Jinja2 escaping, CLI smoke tests, visual regression, and property-based tests. |

## 5. Architecture

```
┌─────────────────────┐     ┌─────────────────────────────┐     ┌─────────────────┐
│ ResumeData /        │     │  instructor structured      │     │  FormattedResume│
│ TailoredResume      │────▶│  output (tool mode)         │────▶│  or             │
│ CoverLetterResult   │     │  + job-category prompt      │     │  FormattedCover │
└─────────────────────┘     └─────────────────────────────┘     └────────┬────────┘
                                                                         │
                                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Jinja2 renderer (custom typst_escape filter) renders a built-in Typst template      │
│  templates/cv/{modern,classic,minimal}.typ                                           │
│  templates/cover_letter/{modern,classic,minimal}.typ                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                                         │
                                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  typst Python package compiles .typ → PDF                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                                         │
                                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Output: output/cv_<company>_<title>_<ts>.pdf                                        │
│  Output: output/cover_letter_<company>_<title>_<ts>.pdf                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                                         │
                                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ATSChecker validates extracted PDF text (optional, enabled in tests/CI)             │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 6. Data Models

New models live in `src/job_applicator/documents/formatted_models.py`.

### 6.1 `FormattedResume`

```python
class FormattedExperienceEntry(BaseModel):
    model_config = {"extra": "forbid"}
    title: str
    company: str
    location: str | None = None
    start_date: str
    end_date: str | None
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
    category: str | None = None  # e.g. "Languages", "Cloud", "Security"
    skills: list[str]

class FormattedResume(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    title: str | None = None
    email: str
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
    projects: list[dict[str, str]] | None = None
    job_category: str | None = None
    emphasized_skills: list[str] | None = None
```

### 6.2 `FormattedCoverLetter`

```python
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

## 6.3 Bridging Plain Text to Structured Content

Because the current `ResumeLoader` populates `raw_text` but not `ResumeData.experience` / `education`, the formatter must extract structure from the existing tailored artifact. Two implementation options were considered:

1. **Parse the markdown-styled `TailoredResume.tailored_text`.** Fragile because the LLM may change header/bullet conventions between runs.
2. **Ask the LLM to re-format the tailored text into `FormattedResume`.** Robust and deterministic; uses instructor structured output and keeps the visual template separate from content extraction.

The design chooses option 2. The formatter prompt includes the full `tailored_text` and the original `ResumeData` fields, and the LLM emits a `FormattedResume` that the template consumes directly. This is a one-time structured-output call per PDF render.

## 7. Renderer Module

New module: `src/job_applicator/documents/pdf_renderer.py`.

### 7.1 `PDFRenderer`

```python
class PDFRenderer:
    def __init__(
        self,
        settings: AppSettings,
        template_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None: ...

    async def render_resume(
        self,
        tailored: TailoredResume,
        job: JobListing | None = None,
        template: str = "modern",
        category: str | None = None,
    ) -> Path: ...

    async def render_cover_letter(
        self,
        result: CoverLetterResult | CoverLetterOutput,
        job: JobListing | None = None,
        template: str = "modern",
        category: str | None = None,
    ) -> Path: ...
```

### 7.2 Internal pipeline

1. **Categorize** the job (if not provided) from `JobListing.title` + `description` via a lightweight keyword-based heuristic. The heuristic maps title/description keywords to one of the built-in categories; if no keyword matches, use `default`.
2. **Format** with instructor: call the LLM with a system prompt that includes the job category and asks for `FormattedResume` / `FormattedCoverLetter`.
3. **Validate** the structured output with existing guards (sign-off validation for cover letters, skill hallucination guards for résumés).
4. **Render** Jinja2 template with a custom `typst_escape` filter; no HTML autoescaping is used.
5. **Compile** with `typst` Python package in `asyncio.to_thread`.
6. **Return** the PDF path and optionally run `ATSChecker` on the extracted text.

### 7.3 Typst compilation backend

Use the `typst` PyPI package only. The exact call will be determined during implementation; the expected shape is:

```python
import typst

typst.compile(str(source_path), output=str(pdf_path), format="pdf")
```

If the API differs, the renderer will adapt in a single private method so the rest of the code is insulated.

## 8. Template System

### 8.1 Layout

```
src/job_applicator/templates/
├── cv/
│   ├── modern.typ
│   ├── classic.typ
│   └── minimal.typ
└── cover_letter/
    ├── modern.typ
    ├── classic.typ
    └── minimal.typ
```

Templates are loaded with `importlib.resources` so they work from installed wheels.

### 8.2 Jinja2 escaping

A dedicated Jinja2 environment is used for Typst rendering:
- `autoescape=False` because HTML escaping is wrong for Typst source.
- A custom filter `typst_escape` that escapes Typst special characters (`#`, `_`, `*`, `$`, `"`, `\`, backticks, braces, brackets).
- All variable interpolations use the typst filter: `{{ resume.name | typst_escape }}`.
- No raw markup from the LLM is ever injected into the template.

### 8.3 Job-category content strategy

The prompt that generates `FormattedResume` / `FormattedCoverLetter` is parameterized by job category. Example categories:

- `tech-support`
- `cybersecurity`
- `systems-administration`
- `network-administration`
- `software-engineering`
- `data-engineering`
- `default`

Each category prompt section instructs the LLM to:
- Emphasize relevant sections.
- Use industry-appropriate action verbs.
- Highlight certifications or tools common in that category.
- De-emphasize less relevant experience.

The visual template does not change; only the structured content does.

## 9. CLI Integration

### 9.1 New / updated flags

| Command | New flags |
|---------|-----------|
| `tailor` | `--format {txt\|pdf\|both}` (default: `txt`), `--template {modern\|classic\|minimal}`, `--category <name>` |
| `generate-cover-letter` | `--format {txt\|pdf\|both}`, `--template ...`, `--category ...` |
| `batch` | `--format {txt\|pdf\|both}`, `--template ...`, `--category ...` |
| `apply` | `--format {txt\|pdf\|both}`, `--template ...`, `--category ...` |

### 9.2 Default behavior

- Default remains plain text to preserve existing behavior and batch performance.
- `--format pdf` writes the PDF plus the existing `.meta.json` sidecar.
- `--format both` writes `.txt` + `.pdf` + `.meta.json`.
- `--template` defaults to `AppSettings.output.resume_template` / `cover_letter_template` or `"modern"`.
- `--category` defaults to auto-detection from the job listing.
- `TailoredResume` and `CoverLetterResult` gain an optional `pdf_path` field so sidecars can reference generated PDFs.

### 9.3 Artifact paths

- Résumé PDF: `output/cv_<safe_company>_<safe_title>_<YYYYMMDD_HHMMSS>.pdf`
- Cover letter PDF: `output/cover_letter_<safe_company>_<safe_title>_<YYYYMMDD_HHMMSS>.pdf`
- `.meta.json` sidecars are updated to include `pdf_path` when applicable.

## 10. TUI Integration

- `documents/artifacts.py` gains `write_tailored_pdf()` and `write_cover_letter_pdf()` helpers.
- `tui/actions.py` calls these helpers when the user requests PDF output.
- Add a PDF preview/open action in the TUI detail view.

## 11. Diagnostics

Add a rendering health check to `diagnostics.py`:

- Verify the `typst` Python package is importable.
- Run a minimal compile (e.g., compile a one-line Typst file to PDF in a temp dir).
- Report success/failure in `job-applicator doctor`.

## 12. Configuration

Add to `AppSettings` / `config.py`:

```python
class OutputConfig(BaseModel):
    default_format: Literal["txt", "pdf", "both"] = "txt"
    resume_template: str = "modern"
    cover_letter_template: str = "modern"
    template_dir: Path | None = None  # optional user templates
```

Corresponding env vars: `JOB_APPLICATOR_OUTPUT_DEFAULT_FORMAT`, etc.

## 13. Error Handling

- All renderer errors subclass `JobApplicatorError` (e.g., `PDFRenderError`).
- Missing `typst` package → clear message with install instructions (`pip install job-applicator[pdf]`).
- Template not found → `PDFRenderError` with list of built-in names.
- Typst compilation failure → capture stderr and raise `PDFRenderError`.
- Invalid structured output from LLM → retry via existing `LLMResilienceConfig` and circuit breaker; after exhaustion, raise `PDFRenderError`.

## 14. Testing Strategy

### 14.1 Unit tests

- `tests/unit/test_pdf_renderer.py`
  - Context building from `TailoredResume` and `CoverLetterResult`.
  - Jinja2 `typst_escape` filter covers all Typst special characters.
  - Template discovery: built-in names, optional user `template_dir`, missing template error.
  - Mock the Typst compile call to assert correct source/output paths.
- `tests/unit/test_formatted_models.py`
  - Validate `FormattedResume` / `FormattedCoverLetter` with valid and invalid payloads.
  - Confirm `extra="forbid"` rejects unknown fields.

### 14.2 Integration tests

- `tests/integration/test_pdf_rendering.py`
  - Render a sample résumé and cover letter to real PDFs using the `typst` package.
  - Extract text from each PDF with PyMuPDF.
  - Run `ATSChecker` on extracted text and assert compatibility.

### 14.3 Visual regression tests

- `tests/integration/test_pdf_regression.py`
  - Generate PDFs for a fixed set of inputs.
  - Compare rasterized page images (not raw PDF bytes, which embed creation timestamps) against checked-in references.
  - Marked `slow` so they run only in CI or on demand.

### 14.4 Property-based tests

- `tests/unit/test_pdf_renderer_fuzz.py`
  - Generate random but valid `FormattedResume` / `FormattedCoverLetter` data.
  - Verify rendering never crashes and produced source contains no unescaped Typst metacharacters.

### 14.5 CLI tests

- Update `test_tailor_workflow.py`, `test_workflow.py`, and add `test_cli_pdf_flags.py`.
  - Verify `--format pdf`, `--format both`, `--template classic`, `--category cybersecurity` produce expected artifacts.

## 15. Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
pdf = [
    "typst>=0.14",
]
```

`jinja2` is already a core dependency, so no new required dependencies are introduced for users who do not use PDF rendering.

## 16. Rollout / Migration

- No migration needed; plain-text artifacts remain default.
- Existing `output/*.meta.json` files without `pdf_path` are still valid.
- `CHANGELOG.md` will note the new `[pdf]` extra and commands.

## 17. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `typst` Python package API changes or is hard to install. | Pin a minimum version; isolate the compile call behind one private method. |
| LLM emits malformed structured content. | Use instructor tool mode + existing retry/circuit breaker + Pydantic validation. |
| Template escaping misses a Typst metacharacter. | Comprehensive unit tests + property-based fuzz tests. |
| Batch performance degrades with PDF generation. | PDF is opt-in; rendering runs in `asyncio.to_thread`; consider content-hash caching later if needed. |
| Cover-letter sign-off validation breaks due to layout. | Keep sign-off text block intact in the template; validate the structured `FormattedCoverLetter.signature` before rendering. |

## 18. Open Questions for Implementation Plan

1. Exact API of the `typst` Python package for PDF compilation.
2. Whether to cache formatted structured output by content hash.
3. Final list of built-in job categories and their prompt sections.
