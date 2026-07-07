# Job Applicator — Project Conventions

AI-powered job application tool using Playwright browser automation with modern LLM stack.

## Hardware Profile
- CPU: Intel i7-13700 (16C/24T)
- RAM: 32 GB
- GPU: RTX 4070 (12 GB VRAM, CUDA 13.0)
- Python: 3.12+ (system has 3.12.13 and 3.13.14)

## Tech Stack

- **Python 3.12+** — modern features, type hints
- **Playwright** — browser automation
- **litellm** — universal LLM API (OpenAI, Anthropic, vLLM, 100+ providers)
- **instructor** — structured outputs from LLMs
- **sentence-transformers** — mxbai-embed-large-v1 for semantic matching
- **Pydantic v2** — data validation and settings
- **Textual** — full-screen terminal UI (the `tui` command / bare invocation)

## Universal Rules

- **Pydantic models cross module boundaries, dicts don't.** If two modules share a payload, it's a pydantic model in `models.py`.
- **Errors are typed and carry context.** Every raised exception is a `JobApplicatorError` subclass. No bare `RuntimeError`s.
- **Never mask a failure with a fabricated default.** When a dependency is unavailable or an operation fails, RAISE a typed error — a default/empty value that looks like a real result is undetectable downstream (a hallucination). Distinguish a genuine *failure* (raise) from a legitimately-*empty* input/result (return empty).
- **Async for I/O, sync for CPU.** Playwright, HTTP calls — async. Parsing, formatting — sync.
- **Configuration is centralized.** `AppSettings` in `config.py` is the single source. Loaded from `config.toml` + `JOB_APPLICATOR_*` env vars.
- **No global mutable state.** Pass via config/context objects.
- **Type annotations on all public functions.** Mypy strict mode enabled.

## Coding Style

- Line length: 100 chars
- Double quotes for strings
- Ruff for linting + formatting
- Mypy strict for type checking
- Pytest for testing (asyncio_mode = auto)

## Testing

Tests are auto-marked by location in `tests/conftest.py`, so marker selection works:

- `bash scripts/green_gate.sh` — canonical fast gate: ruff check, ruff format --check, mypy
  strict on `src/`, then `pytest -m unit`.
- `pytest -m unit` — fast, isolated unit suite (no browser/GPU/vLLM).
  (`pytest tests/unit/` is equivalent.)
- `pytest -m live` — the live tests at `tests/` root that need vLLM (`localhost:8000`) + GPU;
  kept out of the gate (full suite is green when vLLM is up).
- `pytest -m integration` — the integration tests (`tests/integration/`): cross-component seams
  with no vLLM/GPU — board `browser_policy()` → `_make_browser` wiring (construction-only, no real
  launch), PDF rendering, and the apply/batch loops against a real SQLite state store.
- `pytest` — everything.
- Matcher-sensitive changes use the private companion gate:
  `.venv/bin/python scripts/check_matcher_gate_required.py --base <base>`, then
  `.venv/bin/python scripts/eval_matching.py --required` when required.
- Generated document artifacts can be smoke-checked with
  `.venv/bin/python scripts/eval_document_quality.py --resume <txt> --cover-letter <txt>`.
  Generated packet changes can be certified against the private quality set with
  `.venv/bin/python scripts/eval_document_quality.py --packet-set --required`; default private
  manifest: `~/.job-applicator/document-quality-eval/packet-set.jsonl`.
  Private gold standards live under
  `~/.job-applicator/document-quality-eval/gold-standards/`; the cover-letter v1 bundle contains
  a full letter, a prose-only style-guide fixture, a DOCX rendering, extracted prose JSON, and a
  metadata contract for future CV/cover-letter coherence checks.

## Target Boards

- LinkedIn (Phase 1) — implemented. Session-authenticated (reuse a human-established session).
- Indeed (Phase 2) — implemented. Public search, Cloudflare-fronted; selectors tuned against the
  live DOM (2026-06-15) with region auto-detection. The wall is a Cloudflare *managed JS challenge*
  that blocks headless Chrome (not TLS/JA3, not rate-limit), so Indeed runs **headed** on a clean
  profile (windowless via Xvfb) — declared by `IndeedScraper.browser_policy()`.
  **Indeed is search/match-only:** automated apply is intentionally unsupported (Cloudflare
  anti-bot + ToS risk), not a pending feature — the applicator returns a clean SKIPPED result
  directing the user to apply manually. LinkedIn Easy Apply remains the only automated apply path.
- Selector health is explicit live-board diagnostics. `selector-health` and `search/apply
  --selector-health` reuse the real board browser/session, so they are opt-in and separate from
  `doctor`. Search probes validate card/field/description selectors; LinkedIn apply probes validate
  Easy Apply entry + form controls without submitting; Indeed apply probing is intentionally out of
  scope.

## Key Design Decisions

- Headless browser by default, `--headed` flag for debugging
- AI-powered cover letters via litellm (works with local vLLM or cloud APIs)
- **Output language is a packet-level policy.** `[llm] language` = `auto` (mirror the job posting's
  language) | `en` | `fr` lives on `[llm]` so the cover-letter override (`cover_letter_llm`)
  inherits it — the CV and the cover letter always resolve the SAME language, so one application
  never mixes them. French resolves an in-language sign-off ("Cordialement,"), a localized PDF date,
  and recognized French sign-offs. Resolution (`utils/language.py`) is a deliberately small FR/EN
  heuristic, logged per job so a misdetect is catchable.
- **Grounding verifier — the honesty layer (language-agnostic).** Rather than enumerate banned terms
  per language, an LLM enumerates every claim in a generated document and cites the source line that
  grounds it; a deterministic audit (`documents/grounding_verifier.py`) then overrides any ungrounded
  claim (token-overlap + a numeric backstop) and flags coverage gaps. The SOURCE is always the
  BASE résumé (never the JD or the tailored intermediate). Cover letters route through
  `CoverLetterGenerator.generate_verified()` (regenerate ONCE, then fail closed if the best draft is
  still unclean or verification is unavailable);
  tailored résumés through `ResumeTailor.tailor_verified()`, which SURFACES the report on
  `TailoredResume.grounding_report` for human review (a "claims to review" panel + `--json`) and
  NEVER auto-strips — the résumé is the document of record. Fail-safe: any verifier failure raises
  `GroundingUnavailableError`, so a down endpoint can never pass off an unverified document as clean.
- **LLM endpoint is external by default.** The generation features are a *client* of an
  OpenAI-compatible endpoint (`[llm] api_base`, default `localhost:8000`); the app never starts
  one. Embeddings run in-process. Optional `[serve]` extra (vLLM 0.23.x, CUDA 13.0 wheel) +
  `scripts/serve-vllm.sh` self-host a local vLLM for standalone boxes. The script defaults to
  job-applicator's own `.venv/bin/vllm`, `GPU_MEM=0.70`, `MAX_MODEL_LEN=8192`, and
  `ENFORCE_EAGER=1` (avoids vLLM 0.23's V1 cudagraph-profiling OOM on 12 GB cards).
- Instructor for structured LLM outputs (Pydantic models)
- mxbai-embed-large-v1 for semantic job matching (~1.5 GB VRAM)
- Style analyzer with persistent cache and multi-document support. It tries instructor structured
  output first, logs elapsed time and fallback reason, then falls back to direct litellm JSON
  parsing; direct fallback failures use `utils.llm.llm_call_error()`, including the sandbox/socket
  permission-denied diagnostic for localhost vLLM.
- Combined scoring: 60% semantic similarity + 40% skill coverage; **semantic-only when a job lists no requirements** (skill coverage unknown → no neutral floor)
- **Never automate login.** Seed a session once as a human (`login` headed flow, or
  `import-cookies --from-browser`); the tool only reuses it. Automated sign-in trips anti-bot
  defenses and risks the account.
- **Region-aware browser.** Locale, IANA timezone, and Chrome UA are auto-detected
  (`utils/region.py`) unless pinned in `[browser]` config, so geo-aware boards serve the real region.
- **A board declares its browser needs.** `BaseScraper.browser_policy()` (headed / ephemeral
  profile / virtual display) lives on the scraper, not the CLI, so anti-bot requirements can't drift
  and any caller builds the right browser. `_make_browser` (in `factories.py`) reads it.
- **Easy Apply is dry-run by default;** real submission requires `apply --submit`. Dry runs generate
  cover letters as a preview when `--cover-letter` is enabled and a résumé path is configured; the
  generated letter is surfaced in `--json` and the console table. Dry-run validation also exposes
  upload-acceptance evidence and visible form validation errors so `apply --validate` failures can
  be diagnosed without submitting.
- **LinkedIn apply surfaces are distinct.** Easy Apply is in-product and often starts with
  Next/required fields before any Submit button. External "Apply on company website" is a separate
  button surface; the applicator detects it and returns SKIPPED/manual follow-up without clicking.
- **Selector-health failures are honest diagnostics.** Required misses fail and optional misses warn;
  external LinkedIn apply jobs are `skipped` because Easy Apply form selectors are not applicable.
  JSON goes to stdout, logs/diagnostic artifact paths go to stderr, and failure dumps live under
  `~/.job-applicator/debug/selector-health/`.
- **The job funnel is persisted.** `search`/`match` upsert into a SQLite `JobStore`
  (`jobs_store.py`, in `~/.job-applicator/applications.db`) so jobs flow
  search→match→tailor→cover-letter without re-typing. `ApplicationState` stays the
  authority for "applied" (it drives the daily cap); the `status` command composes both by
  URL (furthest-stage-wins, no double-count). `tailor`/`apply` take `--from <id|url>` to act
  on a stored job, and bare `apply` reads the saved list.
- **The TUI is a presentation layer over the service seams.** `tui/` (Textual) calls the
  factories (`_make_browser`/`_make_scraper`/`_make_applicator`/`_make_runtime`),
  `JobMatcher`, `ResumeTailor`, `CoverLetterGenerator`, and `JobStore` directly — it does
  NOT reuse the terminal-bound `workflows/` functions. Bare `job-applicator` opens it when
  stdout+stdin are a TTY (else prints help). Launching, navigating, and filtering are
  offline/account-safe; the account-touching actions (search/apply) run only behind an
  explicit in-app confirm, and a real apply needs a danger checkbox (dry-run default) — the
  low-friction TUI must never turn an account action into a one-keypress accident.
- **Automated CV saves fail closed.** `tailor --yes`, `tailor --json`, and the TUI one-shot
  `tailor_job` path refuse to save if grounding did not complete cleanly, if contact info disappears,
  or if an ATS-compatible base résumé becomes incompatible. CLI non-interactive runs prepend strict
  source-only instructions and retry dirty grounding drafts before refusing. Interactive review can
  still accept a surfaced warning because the user is the document-of-record authority.
- **Doctor reports capability readiness.** `doctor` keeps its narrow blocking `ok` verdict tied to
  LLM `/models` HTTP 200, but also renders capability readiness for AI generation, matching,
  browser workflows, and PDF output so first-use dependency gaps are visible without changing the
  historical exit semantics.

## GPU Memory Layout

Default base model is **`Qwen/Qwen3-8B-AWQ`** (genuine AWQ 4-bit, text-only, ~6.1 GB) — it
fits the 12 GB card alongside the embeddings and grounds stack-heavy JDs the 4B couldn't
(measured: cover-letter employer-stack overclaim 5/6 → 0/5). The smaller, faster
`cyankiwi/Qwen3.5-4B-AWQ-4bit` stays a fallback (pin via `JOB_APPLICATOR_LLM_MODEL` / `[llm]
model`, or `MODEL=… scripts/serve-vllm.sh`). The 4B and 8B can't co-reside on 12 GB, so it's
one base model at a time; a per-step bigger model (the `[cover_letter]` override) is for a
**cloud** endpoint.

| Component | Allocation |
|---|---|
| vLLM (Qwen3-8B-AWQ, eager mode, GPU_MEM=0.70) | ~8.4 GB (6.1 GB weights + KV) |
| Embeddings (mxbai-embed-large-v1) | ~1.5 GB |
| Free VRAM | ~2.4 GB |

## Embedding Service

- Model: `mixedbread-ai/mxbai-embed-large-v1` (1024 dimensions)
- Cache: `~/.job-applicator/embeddings/` (numpy arrays)
- First model load also needs the Hugging Face model cache (`~/.cache/huggingface` by default).
  If a snapshot is cached, `EmbeddingService` loads it with `local_files_only=True` so
  offline/sandboxed matching does not block on Hugging Face metadata probes. If no snapshot is
  cached, first use still needs network access to download the model.
- Matching: Cosine similarity with combined scoring
- Skill threshold: 0.75 for semantic match (empirically tuned; 0.55 matched unrelated same-domain skills)
