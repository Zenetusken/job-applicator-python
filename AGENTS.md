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

# Fast quality gate (canonical: lint + format + typecheck + unit tests)
bash scripts/green_gate.sh

# Auto-fix lint/format
ruff check --fix src/ tests/
ruff format src/ tests/

# Release (see RELEASING.md)
bash scripts/release.sh <version>   # bump version, update CHANGELOG.md, tag, build dist

# Tests — 1377 fast unit tests (the green gate); 1440 total = 1377 unit + 28 integration + 35 live
pytest -m unit -v               # or: pytest tests/unit/ -v   (auto-marked by location)
pytest -m unit -v -k test_name  # single test
python scripts/check_matcher_gate_required.py --base HEAD
python scripts/eval_matching.py --required # REQUIRED after matcher/skill/target-role scoring changes
job-applicator document-quality --resume tailored.txt --cover-letter cover.txt --keyword Python

# CLI
job-applicator                              # bare tty invocation opens the TUI
job-applicator --help
job-applicator doctor                       # Health + capability readiness: AI, matching, browser, PDF
job-applicator config-init                  # Generate config.toml
job-applicator login                        # Headed sign-in once; reuse session headlessly
job-applicator import-cookies --from-browser chrome
job-applicator check-session                # Verify board session is ready
job-applicator search --site linkedin --query "python developer"
job-applicator search --site linkedin --query "python developer" --selector-health  # Opt-in live selector preflight
job-applicator status                       # Show saved job funnel
job-applicator match --resume resume.pdf --jobs-file jobs.json
job-applicator rescore                      # Re-score STORED funnel jobs vs the current résumé (no re-scraping)
job-applicator tailor --resume resume.pdf --from <id-or-url> [--style-guide example.txt] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]
job-applicator generate-cover-letter --resume resume.pdf --job-title "..." --company "..." [--style-guide example.txt] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]
job-applicator ats-check --resume resume.pdf [--json] [--strict]
job-applicator document-quality --resume tailored.txt --cover-letter cover.txt --keyword Python [--json]
job-applicator document-quality --private-packet-set --required [--min-cases 15] [--min-manual-reviews-per-category 5] [--max-artifact-age-days 14] [--required-category support] [--required-language en] [--json]
python scripts/eval_llm_sampler.py --dry-run --json  # Plan baseline-vs-Qwen sampler evals
python scripts/eval_llm_sampler.py --required --integrity-only --json # Generate and integrity-check sampler variants
job-applicator apply --query "python" --validate [--style-guide example.txt] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]            # Dry-run Easy Apply and validate it reaches Submit
job-applicator apply --query "python" --submit --limit 5 [--style-guide example.txt] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]    # Send real applications
job-applicator selector-health --site linkedin --surface search --query "python developer"  # Live selector drift report
job-applicator selector-health --site linkedin --surface apply --from <id-or-url>            # Easy Apply selector probe; never submits
job-applicator selector-health --site indeed --surface search --query "python developer"     # Indeed search/description selectors
job-applicator batch --resume resume.pdf --jobs-file jobs.json --top-k 10 --resume-run [--style-guide "ex1.txt,ex2.pdf"] [--format txt|pdf|both] [--template modern|classic|minimal] [--category <category>]
job-applicator tui                          # Full-screen terminal UI over the funnel store
```

Most commands that read a résumé accept `--resume`, `--ocr-mode {auto|on|off}`, and `--force-ocr`.
`apply` is dry-run by default; real submissions require `--submit`. `apply`, `batch`, `tailor`, and
`generate-cover-letter` all accept `--style-guide` with a single file or comma-separated paths,
and now also `--format`, `--template`, and `--category` for PDF rendering.
Example style guides live in `docs/style-guide-examples/`.
Private cover-letter/CV style gold standards, when populated, live under
`~/.job-applicator/document-quality-eval/gold-standards/`; use the prose-only cover-letter fixture
as style-guide input and the JSON/meta files as hard assertions for future coherence checks.

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
├── selector_registry.py # Shared board selector groups for live drift probes
├── selector_health.py  # Live selector-health service + diagnostics
├── state.py            # SQLite application-history store (duplicate-app prevention)
├── batch_state.py      # SQLite batch-progress store (crash recovery)
├── jobs_store.py       # SQLite job-funnel store (found → matched → tailored → cover_letter)
├── skills/             # Skill-name normalization + hard-negative filtering
├── browser/            # Playwright lifecycle + low-level actions
├── scrapers/           # base.py (BrowserPolicy) → linkedin.py, indeed.py
├── applicators/        # base.py → linkedin.py (Easy Apply, dry-run gated), indeed.py
├── documents/          # cover letter, résumé parsing/tailoring, style/tone/ATS/OCR/sign-off/artifacts
│                       #   grounding_verifier.py (standalone claim-audit diagnostic)
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
  layers: `utils.llm.litellm_completion_kwargs()` sends
  `chat_template_kwargs.enable_thinking=false` by default, and post-processing uses
  `strip_thinking_process()` in `utils/llm.py`.
- **LLM calls need an `openai/` prefix for local vLLM.** Completion callers (style analysis,
  skill extraction, and the standalone grounding diagnostic) build the model id via the single helper
  `utils.llm.litellm_model(config)`, which adds the `openai/` prefix when `llm.api_base` is set.
  Embeddings use `sentence-transformers` directly and do not use this prefix.
- **LLM sampler kwargs are centralized.** Completion callers should use
  `utils.llm.litellm_completion_kwargs(config, temperature=..., max_tokens=...)` instead of
  hand-rolling `max_tokens`, `temperature`, or `extra_body`. Optional sampler config (`top_p`,
  `top_k`, `min_p`, `presence_penalty`, `enable_thinking`) is for measured Qwen/vLLM tuning and
  defaults to the previous request shape except for explicit user overrides. Use
  `scripts/eval_llm_sampler.py` to compare baseline vs Qwen-shaped variants before changing
  defaults; its JSON reports baseline-relative overall/per-dimension deltas. These settings can
  affect criteria extraction and diagnostics, but never deterministic applicant claim realization.
- **StyleAnalyzer has live-path observability.** It logs instructor vs direct-litellm JSON paths,
  elapsed time, and fallback reason. Direct fallback failures route through `utils.llm.llm_call_error()`.
  If localhost vLLM is up but the error says the runtime was denied permission to open a network
  socket, the issue is sandbox/network permissions for Python's aiohttp/httpx path, not a dead vLLM.
- **LLM resilience is configured centrally.** `LLMResilienceConfig` (in `config.py`) drives a shared
  circuit breaker + content-retry runtime in `utils/llm.py` for all LLM consumers.
- **litellm banners are suppressed.** `utils.llm.quiet_litellm()` runs before litellm calls to keep
  feedback/help banners and INFO logs off stdout/stderr.
- **Résumé tailoring is a bounded source overlay, not whole-document rewriting.**
  `documents/resume_document.py` canonicalizes the base résumé and makes every non-summary section
  immutable. Structured job requirements, or temperature-zero evidence-span extraction from a
  bounded job description when requirements are absent, feed deterministic ranking of three
  primary source facts. Local realization preserves each fact's wording and adds only terminal
  punctuation.
  `ResumeTailor.verify_tailored()` fails closed unless the body digest, summary, citations, and
  deterministic realization match the overlay. There
  is no phrase replacement, tool
  stripping, metric/grammar repair, section restoration, or context-specific bullet deletion.
  Retry, custom input, and `[S]` regenerate the summary only.
- **PDF formatting is deterministic.** `PDFRenderer` parses canonical résumé sections and cover
  letter paragraphs/sign-off locally, then renders them with Typst. It never sends an artifact back
  through an LLM, so rendering cannot omit, translate, or invent document content.
- **Cover letters separate targeting from deterministic factual realization.** The same grounded
  criteria path feeds deterministic ranking of exactly three relevant primary source facts. Local
  typed realization produces exactly three body statements, one fact ID per statement and each ID used once. The
  application opening, closing request, and sign-off are also assembled deterministically. The
  artifact sidecar stores all statements, citations, source digest, language, and
  architecture version.
- **The standalone grounding verifier is a diagnostic, not a generation dependency.**
  `documents/grounding_verifier.py` can enumerate claims and audit source quotes for non-overlay
  documents. SOURCE is ALWAYS the BASE résumé (`resume.raw_text`) — never the JD or a generated
  intermediate. Source-overlay generation and certification do not depend on this probabilistic
  verifier.
  Cover-letter generation validates each deterministically realized body sentence directly against
  its selected fact before assembly.
  Résumé summaries are checked directly against their three selected facts; the rest of the résumé
  is source text protected by a digest, so the old whole-document claim-enumeration pass is not in
  the tailoring path. Non-interactive saves (`tailor --yes`, `tailor --json`, `batch`, TUI one-shot
  tailoring) fail closed unless overlay verification and contact/ATS integrity are clean.
- **Output language is a packet-level policy.** `[llm] language` = `auto` (mirror the JD) | `en` |
  `fr`, resolved by `utils/language.py` (small FR/EN heuristic, logged per job). It lives on `[llm]`
  so `cover_letter_llm` inherits it — the CV and cover letter ALWAYS resolve the SAME language.
  Résumé tailoring requires the base résumé to already be in that language; cross-language résumé
  generation fails closed instead of relying on unverified machine translation.
  French gets an in-language sign-off ("Cordialement,"), a localized PDF date, and recognized French
  closings in `documents/sign_off.py`.
- **Tailoring includes a date audit.** `ResumeDateValidator` checks chronological ordering,
  staleness, and advisory employment gap/overlap findings before generating output.
- **Skills are normalized and hard-negative filtered before matching/validation.**
  `skills/normalization.py` canonicalizes aliases (`Python 3` → `Python`, `reactjs` → `React`) and
  drops generic traits (`team player`, `communication skills`) from skill coverage scoring and
  tailored skill sections.
- **Skill-match threshold is 0.75.** Related-but-different tech terms score below this; genuine
  synonyms/supersets pass. Do not lower without re-tuning against real résumé/job pairs.
- **Default embedding model is `mixedbread-ai/mxbai-embed-large-v1`.** Embeddings default to CUDA
  FP16 with a 1.3 GB free-VRAM preflight budget; set `embedding.device="cpu"` for CPU-only boxes.
- **Embedding cache at `~/.job-applicator/embeddings/`.** Style cache at
  `~/.job-applicator/styles/`. Clear with `EmbeddingService.clear_cache()`. The first embedding
  model load also depends on the Hugging Face cache (`~/.cache/huggingface` by default); offline or
  sandboxed `match`/`batch` runs load cached model snapshots with `local_files_only=True` to avoid
  Hugging Face metadata probes. If no snapshot is cached, first use still needs network access to
  download `mixedbread-ai/mxbai-embed-large-v1`.
- **`sentence-transformers` needs CUDA torch.** The default PyPI `torch==2.11.0` wheel is already
  the CUDA 13.0 build, so a plain install matches CUDA-13 drivers. Only if your driver needs an
  *older* CUDA (you get `libcudart.so` errors) reinstall from the index matching your driver, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu126` for a CUDA 12.6 driver.
- **Résumé PDF parser uses multi-parser consensus + OCR fallback.** Supports PDF
  (`pdftotext -layout`), DOCX, TXT/MD, and images. `ResumeLoader.load()` dispatches by extension and
  falls back to OCR when extracted text is short. OCR uses PaddleOCR on CPU by default.
- **Tone detection is keyword-based**, not LLM-based. It remains advisory; deterministic
  source-overlay claim realization does not rewrite facts to match a tone profile. See
  `documents/tone_detector.py`.
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
- **Selector health is live-board diagnostics, not `doctor`.** `selector-health` and the opt-in
  `search/apply --selector-health` preflights open real board pages and add traffic, so they are
  explicit only. Required selector misses fail/abort unless `--ignore-selector-health` is supplied;
  optional selector misses warn. JSON reports stay on stdout; logs/diagnostics go to stderr; failure
  artifacts land in `~/.job-applicator/debug/selector-health/`.
- **LinkedIn apply has two live surfaces.** In-product Easy Apply uses an `Easy Apply` button and
  often shows `Next`/required fields before any Submit button. External apply uses an `Apply` button
  with aria like "Apply to ... on company website"; that is reported as SKIPPED/manual follow-up,
  not treated as Easy Apply selector drift.
- **Indeed apply remains search-only/unsupported.** Indeed search and description selectors are
  probed; on-site apply buttons may appear as `#indeedApplyButton`, but automation intentionally
  returns SKIPPED and directs the user to apply manually.
- **`--verbose` and `--log-file` work both before and after the *data* commands** (search, match,
  tailor, apply, batch, generate-cover-letter, ats-check). `--verbose` emits a structured
  `VerboseReport`; `--log-file` persists it to disk. The status-only `status`/`doctor`/`check-session`
  don't take them (they record nothing into a reporter).
- **JSON output goes to stdout, logs go to stderr.** Enables `job-applicator match --json | jq .`
  without Rich wrapping corruption.
- **Batch runs persist progress for crash recovery.** State lives in
  `~/.job-applicator/applications.db` (tables `batch_runs`, `batch_jobs`, and `applications` for
  submitted apps). Re-run with matching output-affecting parameters and `--resume-run` to skip
  already-completed or explicitly skipped jobs; the resume key includes the job source, résumé,
  top-k/min-score, cover-letter flag, style guide, output format, templates, category, and OCR mode.
  `TAILORED` jobs are re-processed so their cover letters are generated, reusing persisted tailored
  artifacts when available. If a CV succeeds but its requested cover letter fails, the row stays
  `TAILORED` and the run is marked `FAILED`; the CLI still writes `batch_summary_*.json` and exits
  non-zero. Duplicate matched URLs are rejected before concurrent document generation because batch
  recovery is keyed by URL.
- **`apply` is dry-run by default.** Real applications require the explicit `--submit` flag.
  Without it the Easy Apply form is filled but never submitted.
- **Dry-run `apply` generates cover letters as a preview.** Whenever `--cover-letter` is enabled
  (the default) and a résumé path is available, the CLI runs the source-overlay generator, fills
  the cover-letter field, and surfaces the generated text in `--json` output and console notes.
  Criteria extraction may call the LLM when requirements are absent; applicant claim prose does
  not. The application is still not submitted. Use `--no-cover-letter` to skip generation.
- **Apply dry-run validation returns an `ApplicationResult` with a `DryRunValidation` field.** The
  nested object shows whether the Easy Apply button, form fields, résumé upload, cover-letter
  field, and final Submit step were reached. It also records matched advance/submit selectors,
  advance step count, modal title, empty required fields, resume upload acceptance evidence, visible
  form validation errors, disabled-submit evidence, and debug artifact paths when available.
  `job-applicator apply --validate` exits non-zero if any dry run fails to reach Submit.
- **`search` persists discovered jobs.** Discovered listings flow into `jobs_store.py` and are
  visible via `status` and the TUI.
- **Default `llm.max_tokens` is `4096`**, matching `config.example.toml`. Individual analysis tasks
  may self-cap; style analysis caps at 1024 (`STYLE_MAX_TOKENS`). Deterministic applicant claim
  realization consumes no completion tokens.
- **`config.toml` is actually loaded.** `AppSettings.settings_customise_sources()` adds it as the
  lowest-priority source; env vars override it. Point at an alternate file via
  `JOB_APPLICATOR_CONFIG_FILE`; a missing file is a no-op.
- **PDF rendering requires the optional `[pdf]` extra.** Install with
  `pip install -e ".[pdf]"` (pulls in `typst`). Without it, `--format pdf` produces a clear
  error message and `doctor` reports PDF rendering as unavailable. `PDFRenderer` compiles Typst
  directly in-process; spawned executor compile paths previously hung in local integration runs.
- **PDF templates live in `src/job_applicator/templates/`** (Typst `.typ` files) and are packaged
  into the wheel via `[tool.hatch.build.targets.wheel] include`. Built-ins: `modern`, `classic`,
  `minimal`. Set `output.template_dir` to a directory containing `cv/<name>.typ` and/or
  `cover_letter/<name>.typ`; templates are loaded by that full relative path.
- **PDF artifact basenames include microseconds and the template suffix.** Plain text keeps
  `tailored_<company>_<title>_<YYYYMMDD_HHMMSS>.txt`; the PDF is
  `tailored_<company>_<title>_<YYYYMMDD_HHMMSS>_<microseconds>_<template>.pdf`. With
  `--format both` the `.txt` + `.pdf` + one `.meta.json` sidecar (beside the `.txt`) is produced;
  that text sidecar is updated after PDF render so it includes `pdf_path`.

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
present it errors. Defaults: `GPU_MEM=0.65`, `MAX_MODEL_LEN=8192`, and `ENFORCE_EAGER=1`; this
keeps enough 8K KV cache for Qwen3-8B-AWQ while leaving the CUDA embedder's 1.3 GB free-VRAM
preflight budget available on the validated 12 GB RTX 4070 profile. Override with `VLLM_BIN`,
`GPU_MEM`, `MAX_MODEL_LEN`, and `ENFORCE_EAGER` env vars.

`serve-vllm.sh` auto-sets `--tool-call-parser qwen3_xml --enable-auto-tool-choice` for Qwen3
(and Qwen3.5) models, and puts the vLLM venv's bin on `PATH` so flashinfer can JIT-compile a
kernel for a fresh model (the 8B fails with `No such file or directory: 'ninja'` otherwise; `ninja`
ships in the `serve` extra). Skill extraction, the standalone grounding verifier, and style-guide
analysis use instructor in TOOLS mode, which needs this parser; without it they still work but fall back to
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
  `-m integration` all work. Unit suite (`pytest -m unit`, 1377) is fast — no browser/GPU; the green
  gate.
- 28 integration tests live in `tests/integration/` and exercise cross-component seams with no
  vLLM/GPU: board browser-policy wiring, PDF rendering, the apply-loop + batch-loop against a real
  SQLite state store (real daily-cap / dedup / resume persistence the mock-state unit tests can't
  reach), and the offline browser-fingerprint self-consistency gate (real Chrome, loopback only).
- The 35 live tests at `tests/` root carry `-m live`; they need vLLM (`localhost:8000`) + GPU; run
  them manually.
- Matcher changes have a private-data companion gate: run `python scripts/eval_matching.py` after
  edits to `embeddings/matching.py`, skill extraction/normalization/grounding, score weights,
  thresholds, or `[matching] target_roles` behavior. Use
  `python scripts/check_matcher_gate_required.py --base <base>` to detect whether the gate is
  required. The eval script reads
  `~/.job-applicator/matching-eval/gold-set.csv` (override with `GOLD_SET_CSV`) and the live funnel
  DB. Use `--required` for matcher-sensitive changes: missing private data, missing résumé, empty
  labels, no labeled jobs, or incomplete coverage exit non-zero so absence of evidence cannot
  certify a matcher change.
- Generated document packet quality has a first-class artifact gate:
  `job-applicator document-quality --resume <txt> --cover-letter <txt> --keyword <job-term>`.
  It checks obvious quality regressions (missing contact/sections/sign-off, placeholders, markdowny
  cover letters, repetition warnings, and basic job-keyword coverage). It complements, not replaces,
  deterministic source integrity and human review. Generated packet changes can also be certified against a private
  packet set with:
  ```bash
  job-applicator document-quality --private-packet-set --required --min-cases 15 \
    --min-manual-reviews-per-category 5 --max-artifact-age-days 14 \
    --required-category support --required-category risk --required-category network \
    --required-language en --required-language fr
  ```
  The default private manifest is `~/.job-applicator/document-quality-eval/packet-set.jsonl`
  (override with `DOCUMENT_QUALITY_SET`).
  Required certification enforces set breadth, freshness, and category/language coverage; missing
  required evidence exits `2`, present failing evidence exits `1`. The TUI's `D` action is a
  selected-job packet quality check or limited smoke check, not private packet-set certification.
  `scripts/eval_document_quality.py` remains a compatibility wrapper for script-based gates.
  Private gold-standard CV/cover-letter bundles live under
  `~/.job-applicator/document-quality-eval/gold-standards/`. See
  `docs/document-quality-eval.md` for the manifest, 0-4 rubric, and gold-standard bundle layout.
- LLM sampler measurement has a private-data companion harness:
  `python scripts/eval_llm_sampler.py --required --integrity-only --json`. It reads local sampler cases from
  `~/.job-applicator/document-quality-eval/sampler-cases.jsonl`, generates fresh packet manifests
  under `~/.job-applicator/document-quality-eval/sampler-runs/`, certifies each variant through
  document-quality, and reports how much each Qwen-shaped criteria-extraction/transport variant
  improves/regresses against `baseline`. Applicant claim prose is deterministic. Start with
  `--dry-run --json` to inspect commands and env overrides.
- Tests use fixtures from `tests/conftest.py`.
- Embedding tests mock the model (CPU fallback).

## Files Not to Commit

- `config.toml` (contains credentials)
- `.env`
- `.mimocode/` and `.kimi-code/` (local harness/tooling config — kept on disk, not tracked)
- `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
- `output/`, `screenshots/`, `logs/`, `*.log`
- `docs/comprehensive-application-audit-*.md` (generated working audit reports)
