# AGENTS.md

## Project

AI-powered job application tool. Scrapes job boards, matches jobs to résumés via embeddings,
generates cover letters with LLMs, and supports interactive/TUI workflows.

## Commands

```bash
# Setup (requires Python 3.12+)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# First-run setup
job-applicator config-init                  # Create a starter config.toml

# Lint + format + typecheck (run in this order)
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/   # strict on src/; tests are checked by ruff only

# Auto-fix lint/format
ruff check --fix src/ tests/
ruff format src/ tests/

# Release (see RELEASING.md)
bash scripts/release.sh <version>   # bump version, update CHANGELOG.md, tag, build dist

# Tests — ~1271 fast unit tests (the green gate); ~1315 total = ~1271 unit + 9 integration + 35 live
pytest -m unit -v               # or: pytest tests/unit/ -v   (auto-marked by location)
pytest -m unit -v -k test_name  # single test

# CLI
job-applicator                              # bare tty invocation opens the TUI
job-applicator --help
job-applicator doctor                       # Health check: LLM, embeddings, browser, system bins, config, résumé (identity/age/skills), self-host
job-applicator config-init                  # Generate config.toml
job-applicator login                        # Headed sign-in once; reuse session headlessly
job-applicator import-cookies --from-browser chrome
job-applicator check-session                # Verify board session is ready
job-applicator search --site linkedin --query "python developer"
job-applicator status                       # Show saved job funnel
job-applicator match --resume resume.pdf --jobs-file jobs.json
job-applicator rescore                      # Re-score STORED funnel jobs vs the current résumé (no re-scraping)
job-applicator tailor --resume resume.pdf --from <id-or-url> [--style-guide example.txt] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]
job-applicator generate-cover-letter --resume resume.pdf --job-title "..." --company "..." [--style-guide example.txt] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]
job-applicator ats-check --resume resume.pdf [--json] [--strict]
job-applicator apply --query "python" --validate [--style-guide example.txt] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]            # Dry-run Easy Apply and validate it reaches Submit
job-applicator apply --query "python" --submit --limit 5 [--style-guide example.txt] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]    # Send real applications
job-applicator batch --resume resume.pdf --jobs-file jobs.json --top-k 10 --resume-run [--style-guide "ex1.txt,ex2.pdf"] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]
job-applicator tui                          # Full-screen terminal UI over the funnel store
```

Most commands that read a résumé accept `--resume`, `--ocr-mode {auto|on|off}`, and `--force-ocr`.
`apply` is dry-run by default; real submissions require `--submit`. `apply`, `batch`, `tailor`, and
`generate-cover-letter` all accept `--style-guide` with a single file or comma-separated paths,
and now also `--format`, `--template`, and `--category` for PDF rendering.
Example style guides live in `docs/style-guide-examples/`.

- **Cover letters are hard-validated for a proper sign-off.** `documents/sign_off.py` extracts the
  closing word and signature; the signature must match the applicant's full name (or the single known
  part if only one is available). Token-level matching prevents substring false positives like
  `Sam` matching `Samantha`. The name is taken from `profile_name` when set, otherwise from the
  parsed résumé name. Set `profile_name` in `config.toml` if the parsed name is wrong.

## Architecture

```
src/job_applicator/
├── __init__.py         # package version
├── __main__.py         # python -m job_applicator
├── cli.py              # Typer CLI commands (interactive loops live in workflows/)
├── factories.py        # board/browser/scraper/applicator/runtime factories
├── workflows/          # interactive orchestration: cover_letter, tailor, apply
├── config.py           # AppSettings + sub-configs (BrowserConfig, LLMConfig, LLMResilienceConfig,
│                       # EmbeddingConfig, TargetConfig)
├── models.py           # Shared Pydantic models
├── exceptions.py       # JobApplicatorError hierarchy (incl. CookieError)
├── diagnostics.py      # doctor health checks
├── state.py            # SQLite application-history store (duplicate-app prevention)
├── batch_state.py      # SQLite batch-progress store (crash recovery)
├── jobs_store.py       # SQLite job-funnel store (found → matched → tailored → cover_letter)
├── skills/             # Skill-name normalization + hard-negative filtering
├── browser/            # Playwright lifecycle + low-level actions
├── scrapers/           # base.py (BrowserPolicy) → linkedin.py, indeed.py
├── applicators/        # base.py → linkedin.py (Easy Apply, dry-run gated), indeed.py
├── documents/          # cover letter, résumé parsing/tailoring, style/tone/ATS/OCR/sign-off/artifacts
│                       #   grounding_verifier.py (language-agnostic honesty layer)
│                       #   PDF rendering: pdf_renderer.py, formatted_models.py, job_category.py,
│                       #   templates/ (Typst), artifacts.py
├── embeddings/         # embedding service + job matching
├── tui/                # Textual full-screen UI over the funnel store
└── utils/              # logging, LLM retry/breaker/circuit, cookies, console, diff, region,
                        # URL, secure store, text, verbose, profile, path, language (output FR/EN)
```

## Conventions

- **Typed models cross module boundaries; avoid untyped `dict` for business payloads.** Shared
  validation-heavy or serialized contracts go in `models.py` as Pydantic models. Lightweight
  internal structures may use `@dataclass` (e.g., `SearchParams`, `BrowserPolicy`, `MatchResult`).
- **All business-logic exceptions are `JobApplicatorError` subclasses.** A small number of built-ins
  are intentionally raised directly (`IndexError` for out-of-range session access, `OSError` for
  secure-store symlink refusal). No bare `RuntimeError`.
- **No failure-masking fallbacks.** On a failure or unavailable dependency, RAISE a typed error —
  never return a fabricated default/empty that's indistinguishable from a real result (style guide,
  skills, scrape, score). Distinguish a *failure* (raise) from a legitimately-*empty* input/result
  (return empty). The honest-failure scrapers are the template (cards present but none parsed → raise).
- **Async for I/O, sync for CPU.** Playwright/HTTP = async. Parsing/formatting/embeddings = sync.
- **Config is centralized.** `AppSettings` in `config.py`. Env prefix: `JOB_APPLICATOR_*`.
- **No global mutable business state.** Pass `AppSettings`/context objects. A few module-level
  singletons are intentional: Rich consoles (`utils/console.py`) and default DB paths
  (`jobs_store.py`, `state.py`, `batch_state.py`).
- **Pydantic models reject unknown fields by default.** `model_config = {"extra": "forbid"}` is the
  norm. (`documents/cover_letter.py::CoverLetterOutput` now uses it as of v0.3.5.)
- **Use `from typing import TYPE_CHECKING` for imports needed only for annotations.**

## Style

- Line length: 100 chars
- Double quotes (ruff `quote-style = "double"`)
- `from __future__ import annotations` in all non-empty `.py` files
- Mypy strict mode applies to `src/` (`disallow_untyped_defs = true`); the documented typecheck
  command is `mypy src/`, so tests are ruff-checked but not mypy-checked.

## Gotchas

- **LLM output has thinking process.** Qwen models prepend reasoning. Callers suppress it at two
  layers: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` in litellm, and
  post-processing with `strip_thinking_process()` in `utils/llm.py`.
- **LLM calls need an `openai/` prefix for local vLLM.** All completion callers (cover-letter,
  style, tailor, skill-extraction, PDF formatting) build the model id via the single helper
  `utils.llm.litellm_model(config)`, which adds the `openai/` prefix when `llm.api_base` is set.
  Embeddings use `sentence-transformers` directly and do not use this prefix.
- **LLM resilience is configured centrally.** `LLMResilienceConfig` (in `config.py`) drives a shared
  circuit breaker + content-retry runtime in `utils/llm.py` for all LLM consumers.
- **litellm banners are suppressed.** `utils.llm.quiet_litellm()` runs before litellm calls to keep
  feedback/help banners and INFO logs off stdout/stderr.
- **Résumé tailoring has hallucination guards.** `_validate_skills()`, `_strip_hallucinated_tools()`,
  and `_strip_hallucinated_education()` in `documents/resume_tailor.py` keep the output aligned
  with the original résumé. Skill/tool matching uses fuzzy, non-greedy logic in
  `embeddings/matching.py`.
- **Grounding verifier is the language-agnostic honesty layer.** `documents/grounding_verifier.py`:
  an LLM enumerates each claim in a generated doc + cites the SOURCE line; a deterministic audit
  (`audit_report` — token-overlap + numeric backstop + coverage check) overrides ungrounded
  verdicts. SOURCE is ALWAYS the BASE résumé (`resume.raw_text`) — never the JD or the tailored
  intermediate. `CoverLetterGenerator.generate_verified()` regenerates ONCE and keeps the
  strictly-cleaner draft; `ResumeTailor.tailor_verified()` SURFACES the result on
  `TailoredResume.grounding_report` (never auto-strips — the résumé is the document of record).
  Fail-safe: a verifier failure raises `GroundingUnavailableError`, never a clean report. The pure
  audit core is unit-tested (runs on the fast gate); the LLM pass is `-m live`.
- **Output language is a packet-level policy.** `[llm] language` = `auto` (mirror the JD) | `en` |
  `fr`, resolved by `utils/language.py` (small FR/EN heuristic, logged per job). It lives on `[llm]`
  so `cover_letter_llm` inherits it — the CV and cover letter ALWAYS resolve the SAME language.
  French gets an in-language sign-off ("Cordialement,"), a localized PDF date, and recognized French
  closings in `documents/sign_off.py`.
- **Tailoring includes a date audit.** `ResumeDateValidator` checks chronological ordering,
  staleness, and education-date age before generating output.
- **Skills are normalized and hard-negative filtered before matching/validation.**
  `skills/normalization.py` canonicalizes aliases (`Python 3` → `Python`, `reactjs` → `React`) and
  drops generic traits (`team player`, `communication skills`) from skill coverage scoring and
  tailored skill sections.
- **Skill-match threshold is 0.75.** Related-but-different tech terms score below this; genuine
  synonyms/supersets pass. Do not lower without re-tuning against real résumé/job pairs.
- **Default embedding model is `mixedbread-ai/mxbai-embed-large-v1`.** Embeddings default to CUDA
  FP16 with ~1.5 GB VRAM; set `embedding.device="cpu"` for CPU-only boxes.
- **Embedding cache at `~/.job-applicator/embeddings/`.** Style cache at
  `~/.job-applicator/styles/`. Clear with `EmbeddingService.clear_cache()`.
- **`sentence-transformers` needs CUDA torch.** The default PyPI `torch==2.11.0` wheel is already
  the CUDA 13.0 build, so a plain install matches CUDA-13 drivers. Only if your driver needs an
  *older* CUDA (you get `libcudart.so` errors) reinstall from the index matching your driver, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu126` for a CUDA 12.6 driver.
- **Résumé PDF parser uses multi-parser consensus + OCR fallback.** Supports PDF
  (`pdftotext -layout`), DOCX, TXT/MD, and images. `ResumeLoader.load()` dispatches by extension and
  falls back to OCR when extracted text is short. OCR uses PaddleOCR on CPU by default.
- **Tone detection is keyword-based**, not LLM-based. Tone profiles are injected into tailoring and
  cover-letter prompts; see `documents/tone_detector.py`.
- **Per-board browser requirements live in the scraper.** `BrowserPolicy` in `scrapers/base.py`
  declares `headless`, `ephemeral_profile`, and `virtual_display` needs; `factories.py` honors them.
  Indeed sets `headed=True, ephemeral_profile=True, virtual_display=True`. By default the headed
  browser runs windowless via Xvfb; pass `--headed` to disable the virtual display and show a real
  window. The `[indeed]` extra installs `pyvirtualdisplay`; the system `Xvfb` binary is also needed.
- **LinkedIn auth = seed once, reuse the session.** Automated login is disabled for account safety.
  `job-applicator login` opens a headed browser, pre-fills `target.linkedin_email` /
  `target.linkedin_password`, waits for you to click Sign in and solve any CAPTCHA/2FA, then saves
  cookies and the persistent Chrome profile.
- **Cookie JSON is a portable session backup.** `~/.job-applicator/cookies/{linkedin,indeed}.json`
  can seed or restore sessions. For LinkedIn the `li_at` cookie is required. For Indeed (public
  search) the optional `cf_clearance` cookie is only a Cloudflare warm-start; there is no login
  session.
- **`import-cookies` uses a per-site spec with several input modes.** Each board declares required
  and preferred cookies, session flags, and a post-import check. The command supports
  `--from-browser <chrome|chromium|brave|edge|firefox>`, `--li-at` (LinkedIn; pass `-` for stdin),
  `--jsessionid`, `--file <json>`, and `--verify/--no-verify`.
- **Browser context is shared.** `BrowserManager.persistent_context()` returns a single persistent
  context; `persistent_page()` opens pages in it. Scraper and applicator share the authenticated
  session. Do not use `new_context()` or `new_page()` for authenticated flows.
- **Region/UA/timezone are auto-detected at launch** (`utils/region.py`). Pin browser signals with
  `browser.locale`, `browser.timezone`, and `browser.user_agent`. Pin the Indeed regional host with
  `target.indeed_domain` (e.g. `ca.indeed.com`).
- **One host matcher: `utils/url.host_matches(host, base)`.** Use it instead of ad-hoc domain
  suffix checks. It is also used to drop look-alike hosts when importing cookies from a browser.
- **LinkedIn description extraction clicks cards.** The scraper clicks each card, waits for content
  change, clicks the correct "show more" button, then extracts with a 5 000-char cap.
- **`--verbose` and `--log-file` work both before and after the *data* commands** (search, match,
  tailor, apply, batch, generate-cover-letter, ats-check). `--verbose` emits a structured
  `VerboseReport`; `--log-file` persists it to disk. The status-only `status`/`doctor`/`check-session`
  don't take them (they record nothing into a reporter).
- **JSON output goes to stdout, logs go to stderr.** Enables `job-applicator match --json | jq .`
  without Rich wrapping corruption.
- **Batch runs persist progress for crash recovery.** State lives in
  `~/.job-applicator/applications.db` (tables `batch_runs`, `batch_jobs`, and `applications` for
  submitted apps). Re-run with matching parameters and `--resume-run` to skip already-completed or
  explicitly skipped jobs. `TAILORED` jobs are re-processed so their cover letters are generated,
  reusing persisted tailored artifacts when available.
- **`apply` is dry-run by default.** Real applications require the explicit `--submit` flag.
  Without it the Easy Apply form is filled but never submitted.
- **Dry-run `apply` generates cover letters as a preview.** Whenever `--cover-letter` is enabled
  (the default) and a résumé path is available, the CLI calls the LLM, fills the cover-letter field
  in the form, and surfaces the generated text in `--json` output and in the console table notes.
  The application is still not submitted. Use `--no-cover-letter` to skip generation.
- **Apply dry-run validation returns an `ApplicationResult` with a `DryRunValidation` field.** The
  nested object shows whether the Easy Apply button, form fields, résumé upload, cover-letter
  field, and final Submit step were reached. `job-applicator apply --validate` exits non-zero if
  any dry run fails to reach Submit.
- **`search` persists discovered jobs.** Discovered listings flow into `jobs_store.py` and are
  visible via `status` and the TUI.
- **Default `llm.max_tokens` is `4096`**, matching `config.example.toml`. 4096 fits full résumé
  tailoring; style analysis self-caps at 1024 (`STYLE_MAX_TOKENS`). Setting it below ~4096 can
  truncate tailored résumés.
- **`config.toml` is actually loaded.** `AppSettings.settings_customise_sources()` adds it as the
  lowest-priority source; env vars override it. Point at an alternate file via
  `JOB_APPLICATOR_CONFIG_FILE`; a missing file is a no-op.
- **PDF rendering requires the optional `[pdf]` extra.** Install with
  `pip install -e ".[pdf]"` (pulls in `typst`). Without it, `--format pdf` produces a clear
  error message and `doctor` reports PDF rendering as unavailable.
- **PDF templates live in `src/job_applicator/templates/`** (Typst `.typ` files) and are packaged
  into the wheel via `[tool.hatch.build.targets.wheel] include`. Built-ins: `modern`, `classic`,
  `minimal`. Set `output.template_dir` to a directory containing `cv/<name>.typ` and/or
  `cover_letter/<name>.typ`; templates are loaded by that full relative path.
- **PDF artifact basenames include microseconds and the template suffix.** Plain text keeps
  `tailored_<company>_<title>_<YYYYMMDD_HHMMSS>.txt`; the PDF is
  `tailored_<company>_<title>_<YYYYMMDD_HHMMSS>_<microseconds>_<template>.pdf`. With
  `--format both` the `.txt` + `.pdf` + one `.meta.json` sidecar (beside the `.txt`) is produced.

## LLM Setup

Local vLLM must be running at `http://localhost:8000/v1`. Check with:
```bash
curl -s http://localhost:8000/v1/models
```

Default model: `Qwen/Qwen3-8B-AWQ` (genuine AWQ 4-bit, ~6.1 GB — fits the 12 GB card alongside the
embeddings and grounds stack-heavy JDs the 4B couldn't). The smaller, faster
`cyankiwi/Qwen3.5-4B-AWQ-4bit` is a pinnable fallback. Override via `JOB_APPLICATOR_LLM_MODEL` env
var or `config.toml`. Default `llm.max_tokens` is `4096`.

To self-host, install the `[serve]` extra (vLLM 0.23.x, CUDA 13.0 wheel) and run
`scripts/serve-vllm.sh`. The script runs job-applicator's own `.venv/bin/vllm` (or an explicit
`VLLM_BIN`) — for isolation it does NOT silently fall back to a `vllm` on `$PATH`; if neither is
present it errors. Defaults: `GPU_MEM=0.70`, `MAX_MODEL_LEN=8192`, and `ENFORCE_EAGER=1` (needed
on 12 GB cards to avoid vLLM 0.23's V1 cudagraph-profiling OOM). Override with `VLLM_BIN`,
`GPU_MEM`, `MAX_MODEL_LEN`, and `ENFORCE_EAGER` env vars.

`serve-vllm.sh` auto-sets `--tool-call-parser qwen3_xml --enable-auto-tool-choice` for Qwen3
(and Qwen3.5) models, and puts the vLLM venv's bin on `PATH` so flashinfer can JIT-compile a
kernel for a fresh model (the 8B fails with `No such file or directory: 'ninja'` otherwise; `ninja`
ships in the `serve` extra). Cover-letter, grounding-verifier, and style-guide generation use
instructor in TOOLS mode, which needs this parser; without it they still work but fall back to
direct litellm completion. The fallback is less reliable for structured output, so keep the parser
enabled for local vLLM.

## ATS Compatibility Checking

`ATSChecker` in `documents/ats_checker.py` analyzes résumés for email/phone presence, standard
sections (`Experience`, `Education`, `Skills`), optional sections (`Certifications`, `Languages`),
text length, and ASCII tables. CLI usage:
`job-applicator ats-check --resume resume.pdf [--json] [--strict] [--ocr-mode auto|on|off] [--force-ocr]`.
Score >= 60% = compatible. Use `--strict` to exit non-zero in CI when the résumé is incompatible.

Warnings are surfaced before `tailor`, `match`, and `batch`, and before real `apply --submit` runs
that generate cover letters. The default dry-run `apply` does not run the ATS preflight. After
`tailor` and `batch`, the CLI prints a before/after ATS score comparison.

## Testing

- Tests are auto-marked by location (`tests/conftest.py`): `pytest -m unit` / `-m live` /
  `-m integration` all work. Unit suite (`pytest -m unit`, ~1175) is fast — no browser/GPU; the green
  gate.
- 9 integration tests live in `tests/integration/` and exercise browser automation wiring + PDF
  rendering.
- The 35 live tests at `tests/` root carry `-m live`; they need vLLM (`localhost:8000`) + GPU; run
  them manually.
- Tests use fixtures from `tests/conftest.py`.
- Embedding tests mock the model (CPU fallback).

## Files Not to Commit

- `config.toml` (contains credentials)
- `.env`
- `.mimocode/` and `.kimi-code/` (local harness/tooling config — kept on disk, not tracked)
- `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
- `output/`, `screenshots/`, `logs/`, `*.log`
- `docs/comprehensive-application-audit-*.md` (generated working audit reports)
