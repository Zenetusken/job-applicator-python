# AGENTS.md

## Project

AI-powered job application tool. Scrapes job boards, matches jobs to resumes via embeddings, generates cover letters with LLMs.

## Commands

```bash
# Setup (requires Python 3.12+)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Lint + format + typecheck (run in this order)
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/   # untyped third-party imports are silenced via per-module overrides in pyproject.toml

# Auto-fix lint/format
ruff check --fix src/ tests/
ruff format src/ tests/

# Tests — 542 fast unit tests (the green gate); 563 total, the extra 21 are live (need vLLM/GPU)
pytest -m unit -v               # or: pytest tests/unit/ -v   (auto-marked by location)
pytest -m unit -v -k test_name  # single test

# CLI
job-applicator --help
job-applicator doctor                       # Health check: LLM, embeddings, browser, system bins, config
job-applicator match --resume resume.pdf
job-applicator batch --jobs-file jobs.json --resume-run      # Resume an interrupted batch run
```

## Architecture

```
src/job_applicator/
├── cli.py              # Typer CLI entry points
├── config.py           # AppSettings + sub-configs
├── models.py           # Shared Pydantic models
├── exceptions.py       # JobApplicatorError hierarchy
├── state.py            # SQLite application-history store (duplicate-app prevention)
├── batch_state.py      # SQLite batch-progress store (crash recovery)
├── skills/             # Skill-name normalization + hard-negative filtering
├── browser/            # Playwright lifecycle + low-level actions
├── scrapers/           # base.py → linkedin.py, indeed.py
├── applicators/        # base.py → linkedin.py (Easy Apply, dry-run gated), indeed.py
├── documents/          # cover letter, resume parsing/tailoring, style/tone/ATS/OCR
├── embeddings/         # embedding service + job matching
└── utils/              # logging, retry, cookies, region, URL matching, secure store
```

## Conventions

- **Pydantic models cross module boundaries, dicts don't.** Shared payloads go in `models.py`.
- **All exceptions are `JobApplicatorError` subclasses.** No bare `RuntimeError`.
- **Async for I/O, sync for CPU.** Playwright/HTTP = async. Parsing/formatting = sync.
- **Config is centralized.** `AppSettings` in `config.py`. Env prefix: `JOB_APPLICATOR_*`.
- **No global mutable state.** Pass via config/context objects.

## Style

- Line length: 100 chars
- Double quotes (ruff `quote-style = "double"`)
- `from __future__ import annotations` in all files
- Mypy strict mode (`disallow_untyped_defs = true`)

## Gotchas

- **LLM output has thinking process.** Qwen models prepend reasoning; suppress via litellm or strip it. See `utils/llm.py`.
- **Resume tailoring has hallucination guards.** Skill/tool/education validation lives in `resume_tailor.py`; matching uses fuzzy, non-greedy logic in `matching.py`.
- **Embeddings need `openai/` prefix for vLLM.** `model = f"openai/{config.model}"` when calling litellm.
- **Resume PDF parser is fragile.** Verify parsing with `ResumeLoader.load()`; supported formats include PDF (`pdftotext -layout`), DOCX, and image OCR fallback.
- **`sentence-transformers` needs CUDA torch.** If you get `libcudart.so` errors, reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **Embedding cache at `~/.job-applicator/embeddings/`.** Style cache at `~/.job-applicator/styles/`. Clear with `EmbeddingService.clear_cache()`.
- **Tone detection is keyword-based**, not LLM-based. Tone profiles are injected into tailoring/cover-letter prompts; see `documents/tone_detector.py`.
- **`config.toml` is actually loaded.** `AppSettings.settings_customise_sources()` adds it as the lowest-priority source; env vars override it. Point at an alternate file via `JOB_APPLICATOR_CONFIG_FILE`.
- **LinkedIn auth = seed once, reuse the session.** Automated login is disabled for account safety. Run `job-applicator login` once (headed), complete sign-in/CAPTCHA/2FA, and the persistent browser profile retains the session.
- **Cookie JSON is a portable auth backup.** `~/.job-applicator/cookies/{linkedin,indeed}.json` can seed or restore sessions; `import-cookies --from-browser` pulls from your everyday browser.
- **Browser context is shared.** `BrowserManager.persistent_context()` gives one authenticated context for scraper + applicator; use `persistent_page()` for authenticated flows.
- **Indeed requires a headed, ephemeral browser.** It is fronted by a Cloudflare managed challenge; the fix is `BrowserPolicy(headed=True, ephemeral_profile=True, virtual_display=True)`. Region is auto-detected from timezone. See `docs/compose/reports/2026-06-15-indeed-cloudflare-research.md`.
- **Region/UA/timezone are auto-detected at launch** (`utils/region.py`). Pin `browser.timezone` if detection fails.
- **`import-cookies` uses a per-site spec** (`cli.py`). Each board declares required/preferred cookies, session flags, and a post-import check.
- **One host matcher: `utils/url.host_matches(host, base)`.** Use it instead of ad-hoc domain suffix checks.
- **LinkedIn description extraction clicks cards.** Scraper clicks each card, waits for content change, clicks "show more", then extracts.
- **`--verbose` and `--log-file` work both before and after the command.** `job-applicator --verbose match` and `job-applicator match --verbose` both work.
- **JSON output goes to stdout, logs go to stderr.** Enables `job-applicator match --json | jq .` without Rich wrapping corruption.
- **Batch runs persist progress for crash recovery.** State lives in `~/.job-applicator/applications.db` (tables `batch_runs`, `batch_jobs`). Re-run with the same `--resume` / `--jobs-file` / `--query` and `--resume-run` to skip already-tailored jobs. Use `--run-id` to pin or resume a specific run.
- **Skills are normalized before matching/validation.** `src/job_applicator/skills/normalization.py` canonicalizes aliases like `Python 3` → `Python` and `reactjs` → `React`. A hard-negative list drops generic traits (`team player`, `communication skills`) from skill coverage scoring and tailored skill sections.

## LLM Setup

Local vLLM must be running at `http://localhost:8000/v1`. Check with:
```bash
curl -s http://localhost:8000/v1/models
```

Default model: `cyankiwi/Qwen3.5-4B-AWQ-4bit`. Override via `JOB_APPLICATOR_LLM_MODEL` env var or `config.toml`.

## ATS Compatibility Checking

`ATSChecker` in `documents/ats_checker.py` analyzes resumes for email/phone presence, standard sections, text length, and ASCII tables. CLI usage: `job-applicator ats-check --resume resume.pdf [--json]`. Score >= 60% = compatible; warnings are surfaced before `tailor`, `match`, `apply`, and `batch`.

## Testing

- Tests are auto-marked by location (`tests/conftest.py`): `pytest -m unit` / `-m live` / `-m integration` all work. Unit suite (`pytest -m unit`, 542) is fast — no browser/GPU; the green gate.
- The 21 live tests at `tests/` root carry `-m live`; they need vLLM (`localhost:8000`) + GPU; run them manually.
- Tests use fixtures from `tests/conftest.py`.
- Embedding tests mock the model (CPU fallback).

## Files Not to Commit

- `config.toml` (contains credentials)
- `.mimocode/` (local harness/tooling config — kept on disk, not tracked)
- `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
- `output/`, `screenshots/`, `logs/`
