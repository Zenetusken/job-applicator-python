# Verbose CLI Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent. Steps use checkbox syntax for tracking.

**Goal:** Add a global `--verbose` / `-V` flag and `--log-file` path that produce a structured observability report for every CLI command.

**Architecture:** A `VerboseContext` dataclass attached via Typer context carries a `VerboseReporter` instance. Each command records events through reporter helpers and renders the report in a `try/finally` block. Report models live in `models.py`, rendering logic in `utils/verbose.py`.

**Tech Stack:** Typer, Rich, Pydantic, Python 3.12.

---

### Task 1: Add verbose report Pydantic models

**Covers:** [S4]

**Files:**
- Modify: `src/job_applicator/models.py`
- Test: `tests/unit/test_verbose.py`

- [ ] **Step 1: Write failing tests**

```python
from job_applicator.models import (
    ATSReport,
    IOReport,
    LLMReport,
    MatchReport,
    ResumeParsingReport,
    TailoringReport,
    VerboseReport,
)


def test_resume_parsing_report_defaults() -> None:
    r = ResumeParsingReport(source="resume.pdf")
    assert r.source == "resume.pdf"


def test_verbose_report_serializes() -> None:
    v = VerboseReport(command="ats-check", args={"resume": "r.pdf"})
    data = v.model_dump()
    assert data["command"] == "ats-check"
    assert data["args"]["resume"] == "r.pdf"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py -v`
Expected: FAIL - models not defined.

- [ ] **Step 3: Add models to `models.py`**

Append after existing models. Add `Any` to imports at the top:

```python
from typing import Any
```

Then append:

```python
class ResumeParsingReport(BaseModel):
    source: str
    ocr_mode: str = "auto"
    text_length: int = 0
    parsed_name: str = ""
    parsed_email: str = ""
    parsed_phone: str = ""
    parsed_skills: list[str] = Field(default_factory=list)
    parsed_summary_preview: str = ""
    warnings: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ATSReport(BaseModel):
    score: float = 0.0
    is_compatible: bool = False
    checks: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class MatchReport(BaseModel):
    embedding_model: str = ""
    device: str = ""
    load_time_ms: int = 0
    job_count: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class LLMReport(BaseModel):
    model: str = ""
    endpoint: str = ""
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    temperature: float | None = None
    calls: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class TailoringReport(BaseModel):
    tone: str = ""
    tone_confidence: float = 0.0
    pre_match_score: float | None = None
    attempts: int = 0
    ats_before: float = 0.0
    ats_after: float = 0.0
    hallucination_actions: list[str] = Field(default_factory=list)
    changes_summary: str = ""

    model_config = {"extra": "forbid"}


class IOReport(BaseModel):
    files_written: list[str] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    batch_summary_path: str | None = None

    model_config = {"extra": "forbid"}


class VerboseReport(BaseModel):
    command: str
    args: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    duration_ms: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    resume: ResumeParsingReport | None = None
    ats: ATSReport | None = None
    match: MatchReport | None = None
    llm: LLMReport | None = None
    tailoring: TailoringReport | None = None
    io: IOReport | None = None
    errors: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
```

Add `Any` to the `typing` import at the top if not present.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/job_applicator/models.py tests/unit/test_verbose.py
git commit -m "feat(verbose): add VerboseReport and sub-report Pydantic models"
```

---

### Task 2: Add VerboseReporter utility

**Covers:** [S5, S6, S7, S8]

**Files:**
- Create: `src/job_applicator/utils/verbose.py`
- Test: `tests/unit/test_verbose.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from job_applicator.utils.verbose import VerboseReporter


def test_reporter_collects_resume_info() -> None:
    reporter = VerboseReporter(command="ats-check", args={"resume": "r.pdf"}, config={})
    reporter.record_resume(
        source="r.pdf",
        ocr_mode="auto",
        text_length=1234,
        parsed_name="John",
        parsed_email="j@example.com",
        parsed_phone="555-1234",
        parsed_skills=["Python"],
        parsed_summary_preview="Summary...",
    )
    report = reporter.report
    assert report.resume is not None
    assert report.resume.parsed_name == "John"


def test_reporter_writes_log_file(tmp_path: Path) -> None:
    reporter = VerboseReporter(command="ats-check", args={}, config={})
    reporter.record_ats(score=1.0, is_compatible=True, checks=[], warnings=[], suggestions=[])
    log_path = tmp_path / "out.json"
    reporter.render(console=None, log_file=str(log_path))
    assert log_path.exists()


def test_reporter_collects_errors() -> None:
    reporter = VerboseReporter(command="ats-check", args={}, config={})
    reporter.record_error("something failed")
    assert reporter.report.errors == ["something failed"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py -v`
Expected: FAIL - VerboseReporter not defined.

- [ ] **Step 3: Implement VerboseReporter**

```python
"""Structured observability reporter for CLI commands."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from job_applicator.models import (
    ATSReport,
    IOReport,
    LLMReport,
    MatchReport,
    ResumeParsingReport,
    TailoringReport,
    VerboseReport,
)


class VerboseReporter:
    """Collect and render structured observability reports."""

    def __init__(self, command: str, args: dict[str, Any], config: dict[str, Any]) -> None:
        self._started_at = datetime.now()
        self.report = VerboseReport(
            command=command,
            args=args,
            config=config,
        )

    def record_resume(
        self,
        *,
        source: str,
        ocr_mode: str = "auto",
        text_length: int = 0,
        parsed_name: str = "",
        parsed_email: str = "",
        parsed_phone: str = "",
        parsed_skills: list[str] | None = None,
        parsed_summary_preview: str = "",
        warnings: list[str] | None = None,
    ) -> None:
        self.report.resume = ResumeParsingReport(
            source=source,
            ocr_mode=ocr_mode,
            text_length=text_length,
            parsed_name=parsed_name,
            parsed_email=parsed_email,
            parsed_phone=parsed_phone,
            parsed_skills=parsed_skills or [],
            parsed_summary_preview=parsed_summary_preview,
            warnings=warnings or [],
        )

    def record_ats(
        self,
        *,
        score: float,
        is_compatible: bool,
        checks: list[dict[str, Any]],
        warnings: list[str],
        suggestions: list[str],
    ) -> None:
        self.report.ats = ATSReport(
            score=score,
            is_compatible=is_compatible,
            checks=checks,
            warnings=warnings,
            suggestions=suggestions,
        )

    def record_match(
        self,
        *,
        embedding_model: str,
        device: str,
        load_time_ms: int,
        results: list[dict[str, Any]],
    ) -> None:
        self.report.match = MatchReport(
            embedding_model=embedding_model,
            device=device,
            load_time_ms=load_time_ms,
            job_count=len(results),
            results=results,
        )

    def record_llm_call(
        self,
        *,
        model: str,
        endpoint: str,
        prompt_tokens: int | None = None,
        response_tokens: int | None = None,
        temperature: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.report.llm is None:
            self.report.llm = LLMReport(model=model, endpoint=endpoint)
        self.report.llm.calls.append(
            {
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "temperature": temperature,
                "details": details or {},
            }
        )

    def record_tailoring(
        self,
        *,
        tone: str = "",
        tone_confidence: float = 0.0,
        pre_match_score: float | None = None,
        attempts: int = 0,
        ats_before: float = 0.0,
        ats_after: float = 0.0,
        hallucination_actions: list[str] | None = None,
        changes_summary: str = "",
    ) -> None:
        self.report.tailoring = TailoringReport(
            tone=tone,
            tone_confidence=tone_confidence,
            pre_match_score=pre_match_score,
            attempts=attempts,
            ats_before=ats_before,
            ats_after=ats_after,
            hallucination_actions=hallucination_actions or [],
            changes_summary=changes_summary,
        )

    def record_io(
        self,
        *,
        files_written: list[str] | None = None,
        files_read: list[str] | None = None,
        batch_summary_path: str | None = None,
    ) -> None:
        if self.report.io is None:
            self.report.io = IOReport()
        if files_written:
            self.report.io.files_written.extend(files_written)
        if files_read:
            self.report.io.files_read.extend(files_read)
        if batch_summary_path:
            self.report.io.batch_summary_path = batch_summary_path

    def record_error(self, message: str) -> None:
        self.report.errors.append(message)

    def _finalize(self) -> None:
        self.report.duration_ms = int(
            (datetime.now() - self._started_at).total_seconds() * 1000
        )

    def render(self, console: Console | None, log_file: str | None = None) -> None:
        self._finalize()
        if console is not None:
            self._render_terminal(console)
        if log_file:
            Path(log_file).write_text(
                self.report.model_dump_json(indent=2), encoding="utf-8"
            )

    def _render_terminal(self, console: Console) -> None:
        table = Table(title="Observability Report")
        table.add_column("Section", style="cyan")
        table.add_column("Value")

        table.add_row("Command", self.report.command)
        table.add_row("Duration", f"{self.report.duration_ms} ms")

        if self.report.resume:
            r = self.report.resume
            table.add_row(
                "Resume",
                f"{r.source} | text={r.text_length} | skills={len(r.parsed_skills)}",
            )

        if self.report.ats:
            a = self.report.ats
            status = "PASS" if a.is_compatible else "FAIL"
            table.add_row("ATS", f"{status} ({a.score:.0%})")

        if self.report.match:
            m = self.report.match
            table.add_row("Match", f"{m.job_count} jobs | model={m.embedding_model}")

        if self.report.tailoring:
            t = self.report.tailoring
            table.add_row("Tailoring", f"{t.attempts} attempt(s) | tone={t.tone}")

        if self.report.io:
            io = self.report.io
            table.add_row("I/O", f"written={len(io.files_written)} read={len(io.files_read)}")

        if self.report.errors:
            table.add_row("Errors", "; ".join(self.report.errors))

        console.print(Panel(table, title="Verbose Report", expand=False))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/job_applicator/utils/verbose.py tests/unit/test_verbose.py
git commit -m "feat(verbose): add VerboseReporter utility"
```

---

### Task 3: Wire global flags and create context

**Covers:** [S3, S7]

**Files:**
- Modify: `src/job_applicator/cli.py`
- Test: `tests/unit/test_verbose.py`

- [ ] **Step 1: Write failing test**

```python
from typer.testing import CliRunner

from job_applicator.cli import app

runner = CliRunner()


def test_verbose_flag_is_accepted() -> None:
    result = runner.invoke(app, ["--verbose", "ats-check", "--help"])
    assert result.exit_code == 0


def test_log_file_requires_verbose() -> None:
    result = runner.invoke(app, ["--log-file", "out.json", "ats-check", "--help"])
    assert result.exit_code != 0
    assert "verbose" in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py::test_verbose_flag_is_accepted tests/unit/test_verbose.py::test_log_file_requires_verbose -v`
Expected: FAIL.

- [ ] **Step 3: Modify `cli.py` main callback**

Add imports:

```python
from dataclasses import dataclass
from typing import Any

from job_applicator.utils.verbose import VerboseReporter


@dataclass
class VerboseContext:
    verbose: bool
    log_file: str | None = None
```

Change `main()` signature and body:

```python
@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Emit structured observability report.",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write verbose report to file (requires --verbose).",
    ),
) -> None:
    """Automated job application tool with AI-powered cover letters."""
    if log_file and not verbose:
        raise typer.BadParameter("--log-file requires --verbose")
    ctx.obj = VerboseContext(verbose=verbose, log_file=log_file)
```

Move `version_callback` logic into the callback. Remove the standalone `@app.callback()` without args if it exists.

- [ ] **Step 4: Add helper to get reporter from context**

Add near top of `cli.py`:

```python
def _get_reporter(
    ctx: typer.Context,
    command: str,
    args: dict[str, Any],
    config: dict[str, Any],
) -> VerboseReporter | None:
    vctx = ctx.obj
    if not isinstance(vctx, VerboseContext) or not vctx.verbose:
        return None
    return VerboseReporter(command=command, args=args, config=config)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py::test_verbose_flag_is_accepted tests/unit/test_verbose.py::test_log_file_requires_verbose -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/cli.py tests/unit/test_verbose.py
git commit -m "feat(verbose): add global --verbose and --log-file flags"
```

---

### Task 4: Instrument `ats-check` command

**Covers:** [S4, S7]

**Files:**
- Modify: `src/job_applicator/cli.py` (`ats_check` function)
- Test: `tests/unit/test_verbose.py`

- [ ] **Step 1: Write failing integration test**

```python
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from job_applicator.cli import app
from job_applicator.models import ResumeData

runner = CliRunner()


def test_ats_check_verbose_output(tmp_path: Path) -> None:
    resume = tmp_path / "resume.pdf"
    resume.write_text("dummy")
    with patch("job_applicator.cli.ResumeLoader") as mock_loader:
        mock_loader.return_value.load.return_value = ResumeData(
            raw_text="John Doe\\njohn@example.com\\n555-1234\\nSkills: Python",
            name="John Doe",
            email="john@example.com",
            phone="555-1234",
            skills=["Python"],
        )
        result = runner.invoke(app, ["--verbose", "ats-check", "--resume", str(resume)])
    assert result.exit_code == 0
    assert "Verbose Report" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py::test_ats_check_verbose_output -v`
Expected: FAIL - no report rendered.

- [ ] **Step 3: Instrument `ats_check`**

Inside `ats_check()`, after loading settings and resolving OCR mode:

```python
reporter = _get_reporter(
    ctx=typer.Context.get_current_context(),
    command="ats-check",
    args={"resume": settings.resume_path, "ocr_mode": effective_ocr_mode},
    config=_sanitize_config(settings),
)
```

After `resume_data = loader.load(...)`:

```python
if reporter:
    reporter.record_resume(
        source=str(settings.resume_path),
        ocr_mode=effective_ocr_mode,
        text_length=len(resume_data.raw_text),
        parsed_name=resume_data.name,
        parsed_email=resume_data.email,
        parsed_phone=resume_data.phone,
        parsed_skills=resume_data.skills,
        parsed_summary_preview=resume_data.summary[:100],
    )
```

After `result = checker.check(resume_data)`:

```python
if reporter:
    reporter.record_ats(
        score=result.score,
        is_compatible=result.is_compatible,
        checks=result.checks,
        warnings=result.warnings,
        suggestions=result.suggestions,
    )
```

Wrap the existing function body in `try/finally` and render at the end:

```python
try:
    # existing body
finally:
    if reporter:
        log_file = None
        vctx = ctx.obj
        if isinstance(vctx, VerboseContext):
            log_file = vctx.log_file
        reporter.render(console, log_file=log_file)
```

Note: `ats_check` must accept `ctx: typer.Context` as first parameter.

- [ ] **Step 4: Add `_sanitize_config` helper**

```python
def _sanitize_config(settings: AppSettings) -> dict[str, Any]:
    data = settings.model_dump()
    _redact_secrets(data)
    return data


def _redact_secrets(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and any(
                s in key.lower() for s in ("password", "secret", "key", "token")
            ):
                obj[key] = "[REDACTED]"
            else:
                _redact_secrets(value)
    elif isinstance(obj, list):
        for item in obj:
            _redact_secrets(item)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/cli.py tests/unit/test_verbose.py
git commit -m "feat(verbose): instrument ats-check command"
```

---

### Task 5: Instrument `match` and `batch` commands

**Covers:** [S4, S7]

**Files:**
- Modify: `src/job_applicator/cli.py` (`match`, `batch`)

- [ ] **Step 1: Instrument `match`**

Ensure `match` signature starts with `ctx: typer.Context`. Create reporter after settings load:

```python
reporter = _get_reporter(
    ctx=ctx,
    command="match",
    args={"resume": settings.resume_path, "jobs_file": jobs_file, "top_k": top_k},
    config=_sanitize_config(settings),
)
```

Record resume after load, ATS after preflight, match after ranking:

```python
if reporter and matches:
    reporter.record_match(
        embedding_model=settings.embedding.model_name,
        device=settings.embedding.device,
        load_time_ms=0,
        results=[
            {
                "rank": i + 1,
                "title": m.job.title,
                "company": m.job.company,
                "score": round(m.score, 4),
                "semantic_score": round(m.semantic_score, 4),
                "skill_score": round(m.skill_score, 4),
                "matched_skills": m.matched_skills,
                "missing_skills": m.missing_skills,
            }
            for i, m in enumerate(matches)
        ],
    )
```

Wrap body in `try/finally` and render using `ctx.obj` `VerboseContext.log_file`. In exception handlers call `reporter.record_error(str(exc))` before re-raising.

- [ ] **Step 2: Instrument `batch`**

Ensure `batch` signature starts with `ctx: typer.Context`. Create reporter after settings load:

```python
reporter = _get_reporter(
    ctx=ctx,
    command="batch",
    args={
        "resume": settings.resume_path,
        "jobs_file": jobs_file,
        "query": query,
        "top_k": top_k,
        "cover_letter": cover_letter,
    },
    config=_sanitize_config(settings),
)
```

Add a `written_paths: list[str] = []` mutable list near the reporter. In `_process_one()`, append saved file paths to this list. After all jobs finish, record IO:

```python
if reporter:
    reporter.record_io(
        files_written=written_paths,
        batch_summary_path=str(output_dir / "batch_summary.json"),
    )
```

Wrap body in `try/finally` and render using `ctx.obj` `VerboseContext.log_file`. In exception handlers call `reporter.record_error(str(exc))` before re-raising.

- [ ] **Step 3: Run integration tests**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py tests/unit/test_batch.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat(verbose): instrument match and batch commands"
```

---

### Task 6: Instrument `tailor` and `generate-cover-letter` commands

**Covers:** [S4, S7]

**Files:**
- Modify: `src/job_applicator/cli.py` (`tailor`, `generate_cover_letter`)

- [ ] **Step 1: Instrument `tailor`**

Ensure `tailor` signature starts with `ctx: typer.Context`. Create reporter after settings load:

```python
reporter = _get_reporter(
    ctx=ctx,
    command="tailor",
    args={
        "resume": settings.resume_path,
        "job": job_description,
        "min_score": min_score,
        "interactive": interactive,
    },
    config=_sanitize_config(settings),
)
```

Record resume after load, ATS preflight score, and pre-tailor match score. Record LLM call after invoking the tailor:

```python
if reporter:
    reporter.record_llm_call(
        model=settings.llm.model,
        endpoint=settings.llm.api_base,
        temperature=settings.llm.temperature,
        details={"job_title": job.title if job else "", "interactive": interactive},
    )
```

Record tailoring after LLM succeeds:

```python
if reporter and tailored:
    reporter.record_tailoring(
        tone=tailored.style_guide.tone_profile if tailored.style_guide else "",
        tone_confidence=0.0,
        pre_match_score=pre_match_score,
        attempts=1,
        ats_before=ats_before,
        ats_after=ats_after,
        hallucination_actions=[],
        changes_summary=tailored.changes_summary or "",
    )
```

Record IO after saving the file and wrap body in `try/finally`. Render using `ctx.obj` `VerboseContext.log_file`. In exception handlers call `reporter.record_error(str(exc))` before re-raising.

- [ ] **Step 2: Instrument `generate-cover-letter`**

Ensure `generate_cover_letter` signature starts with `ctx: typer.Context`. Create reporter, record resume, style guide name, LLM model, and output file path. Record LLM call:

```python
if reporter:
    reporter.record_llm_call(
        model=settings.llm.model,
        endpoint=settings.llm.api_base,
        temperature=settings.llm.temperature,
        details={"style_guide": style_guide_path or "default"},
    )
```

Wrap body in `try/finally` and render using `ctx.obj` `VerboseContext.log_file`. In exception handlers call `reporter.record_error(str(exc))` before re-raising.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_verbose.py tests/unit/test_workflow.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat(verbose): instrument tailor and generate-cover-letter commands"
```

---

### Task 7: Instrument remaining commands

**Covers:** [S3, S7]

**Files:**
- Modify: `src/job_applicator/cli.py` (`search`, `apply`, `config_init`)

- [ ] **Step 1: Instrument `search`**

Ensure `search` signature starts with `ctx: typer.Context`. Create reporter after settings load:

```python
reporter = _get_reporter(
    ctx=ctx,
    command="search",
    args={"query": query, "board": board, "location": location, "limit": limit},
    config=_sanitize_config(settings),
)
```

After fetching results:

```python
if reporter:
    reporter.record_io(files_written=[str(saved_path)] if saved_path else [])
```

Wrap body in `try/finally` and render using `ctx.obj` `VerboseContext.log_file`. In exception handlers call `reporter.record_error(str(exc))` before re-raising.

- [ ] **Step 2: Instrument `apply`**

Ensure `apply` signature starts with `ctx: typer.Context`. Create reporter after settings load:

```python
reporter = _get_reporter(
    ctx=ctx,
    command="apply",
    args={"resume": settings.resume_path, "jobs_file": jobs_file, "limit": limit},
    config=_sanitize_config(settings),
)
```

Record resume after load and application statuses after the run:

```python
if reporter and results:
    reporter.record_io(files_written=[str(r.output_path) for r in results if r.output_path])
```

Wrap body in `try/finally` and render using `ctx.obj` `VerboseContext.log_file`. In exception handlers call `reporter.record_error(str(exc))` before re-raising.

- [ ] **Step 3: Instrument `config_init`**

Ensure `config_init` signature starts with `ctx: typer.Context`. Create reporter:

```python
reporter = _get_reporter(
    ctx=ctx,
    command="config-init",
    args={"output": output_path},
    config={},
)
```

After writing the file:

```python
if reporter:
    reporter.record_io(files_written=[str(output_path)])
```

Wrap body in `try/finally` and render using `ctx.obj` `VerboseContext.log_file`. In exception handlers call `reporter.record_error(str(exc))` before re-raising.

- [ ] **Step 4: Add tests for remaining commands**

Add to `tests/unit/test_verbose.py`:

```python
def test_search_verbose_flag() -> None:
    result = runner.invoke(app, ["--verbose", "search", "--help"])
    assert result.exit_code == 0

def test_config_init_verbose() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["--verbose", "config-init", "--output", "config.toml"])
        assert result.exit_code == 0
        assert "Verbose Report" in result.output
```

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/cli.py tests/unit/test_verbose.py
git commit -m "feat(verbose): instrument search, apply, and config-init commands"
```

---

### Task 8: Final verification

**Covers:** [S9]

- [ ] **Step 1: Run lint/format/typecheck**

```bash
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/mypy src/job_applicator/ --ignore-missing-imports
```

- [ ] **Step 2: Run full unit tests**

```bash
.venv/bin/python -m pytest tests/unit/ -q
```

Expected: 315+ tests pass.

- [ ] **Step 3: Run live smoke test**

Use a local PDF path on your machine (replace the example path):

```bash
.venv/bin/job-applicator --verbose ats-check --resume "/path/to/your/resume.pdf"
```

Expected: normal ATS output plus "Verbose Report" panel.

Also test `--log-file`:

```bash
.venv/bin/job-applicator --verbose --log-file /tmp/report.json ats-check --resume "/path/to/your/resume.pdf"
```

Expected: terminal report plus JSON file at `/tmp/report.json`.

- [ ] **Step 4: Commit verification**

```bash
git add -A
git commit -m "chore(verbose): final verification"
```

---

## Self-Review Checklist

- [S1] problem: Task 1 (context)
- [S2] solution overview: Task 3
- [S3] flags: Task 3
- [S4] data model: Task 1
- [S5] sanitization: Task 4
- [S6] terminal output: Task 2
- [S7] log file: Tasks 2, 4
- [S8] architecture: Tasks 2-3, 4-7
- [S9] validation: Tasks 1, 2, 4, 7, 8
