# Verbose CLI Flag Design Spec

## [S1] Problem

The job-applicator CLI currently has no unified way to observe what happens during a command run. Users must rely on INFO-level logs and scattered console output. This makes debugging parsing, matching, LLM tailoring, and ATS issues difficult.

## [S2] Solution Overview

Add a global `--verbose` / `-V` (uppercase) flag and an optional `--log-file` path to the CLI. When `--verbose` is enabled, every command emits a structured observability report in the terminal and optionally writes a machine-readable JSON trace to `--log-file`.

## [S3] Scope

- Global `--verbose` / `-V` flag on `job-applicator` callback.
- Optional `--log-file` global flag (only valid when `--verbose` is enabled; error if used alone).
- All commands affected: search, apply, generate-cover-letter, match, batch, tailor, ats-check, config-init.
- Report includes command metadata, resume parsing, ATS, matching, LLM/tailoring, and file I/O details.
- `--verbose` does **not** change application log level; it is an additional structured report layer.

## [S4] Data Model

All models added to `src/job_applicator/models.py`.

### [S4.1] ResumeParsingReport

- `source`: str
- `ocr_mode`: str
- `text_length`: int
- `parsed_name`: str
- `parsed_email`: str
- `parsed_phone`: str
- `parsed_skills`: list[str]
- `parsed_summary_preview`: str
- `warnings`: list[str]

### [S4.2] ATSReport

- `score`: float
- `is_compatible`: bool
- `checks`: list[dict[str, Any]]
- `warnings`: list[str]
- `suggestions`: list[str]

### [S4.3] MatchReport

- `embedding_model`: str
- `device`: str
- `load_time_ms`: int
- `job_count`: int
- `results`: list[dict[str, Any]]

### [S4.4] LLMReport

- `model`: str
- `endpoint`: str
- `prompt_tokens`: int | None
- `response_tokens`: int | None
- `temperature`: float | None
- `calls`: list[dict[str, Any]]

### [S4.5] TailoringReport

- `tone`: str
- `tone_confidence`: float
- `pre_match_score`: float | None
- `attempts`: int
- `ats_before`: float
- `ats_after`: float
- `hallucination_actions`: list[str]
- `changes_summary`: str

### [S4.6] IOReport

- `files_written`: list[str]
- `files_read`: list[str]
- `batch_summary_path`: str | None

Scope: user-visible output/input files only. Temporary files (OCR temp PNGs, pdftotext temp files) are excluded.

### [S4.7] VerboseReport

- `command`: str
- `args`: dict[str, Any]
- `started_at`: datetime
- `duration_ms`: int
- `config`: dict (sanitized)
- `resume`: ResumeParsingReport | None
- `ats`: ATSReport | None
- `match`: MatchReport | None
- `llm`: LLMReport | None
- `tailoring`: TailoringReport | None
- `io`: IOReport | None
- `errors`: list[str]

## [S5] Sanitization

Config is serialized to dict via Pydantic `model_dump()`. Any field whose name contains `password`, `secret`, `key`, or `token` is replaced with `"[REDACTED]"`.

## [S6] Terminal Output

Use Rich panels/tables. Non-verbose output remains unchanged. Verbose output appends a single "Observability Report" panel after the normal command output.

## [S7] Log File

JSON serialization of `VerboseReport`. `--log-file` accepts a path. If omitted, no file is written. Passing `--log-file` without `--verbose` raises `typer.BadParameter`.

## [S8] Architecture

1. `VerboseContext` dataclass attached to Typer context via `app.callback()`.
2. `VerboseReporter` class in `src/job_applicator/utils/verbose.py` builds `VerboseReport` incrementally.
3. Each command instruments key points by calling `reporter.record_*()` helpers.
4. Each command wraps its body in `try/finally` and calls `reporter.render()` at the end.
5. `render()` prints the Rich panel and writes `--log-file` if requested.
6. Unhandled exceptions skip rendering; errors should be captured via `record_error()` before re-raising when possible.
7. **Exit hook**: A Typer callback or `atexit` handler is **not** used. Rendering is the command's responsibility in `finally`. If an exception propagates past `finally` (e.g., `SystemExit` raised inside `finally`), the report is lost. This is acceptable because the alternative (rendering in an exit hook) risks printing after the terminal is torn down or after Typer has already emitted its error message.

## [S9] Validation

- Unit tests for `VerboseReporter` and `VerboseReport` serialization in `tests/unit/test_verbose.py`.
- Integration test that runs `ats-check` with `--verbose` and asserts report sections exist.
- Lint/format/typecheck clean.
