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

- `pytest -m unit` — fast, isolated unit suite (no browser/GPU/vLLM). The green gate.
  (`pytest tests/unit/` is equivalent.)
- `pytest -m live` — the live tests at `tests/` root that need vLLM (`localhost:8000`) + GPU;
  kept out of the gate (full suite is green when vLLM is up).
- `pytest -m integration` — the integration tests (`tests/integration/`): board
  `browser_policy()` → `_make_browser` wiring (construction-only, no real launch).
- `pytest` — everything.

## Target Boards

- LinkedIn (Phase 1) — implemented. Session-authenticated (reuse a human-established session).
- Indeed (Phase 2) — implemented. Public search, Cloudflare-fronted; selectors tuned against the
  live DOM (2026-06-15) with region auto-detection. The wall is a Cloudflare *managed JS challenge*
  that blocks headless Chrome (not TLS/JA3, not rate-limit), so Indeed runs **headed** on a clean
  profile (windowless via Xvfb) — declared by `IndeedScraper.browser_policy()`.
  **Indeed is search/match-only:** automated apply is intentionally unsupported (Cloudflare
  anti-bot + ToS risk), not a pending feature — the applicator returns a clean SKIPPED result
  directing the user to apply manually. LinkedIn Easy Apply remains the only automated apply path.

## Key Design Decisions

- Headless browser by default, `--headed` flag for debugging
- AI-powered cover letters via litellm (works with local vLLM or cloud APIs)
- **LLM endpoint is external by default.** The generation features are a *client* of an
  OpenAI-compatible endpoint (`[llm] api_base`, default `localhost:8000`); the app never starts
  one. Embeddings run in-process. Optional `[serve]` extra (vLLM 0.23.x, CUDA 13.0 wheel) +
  `scripts/serve-vllm.sh` self-host a local vLLM for standalone boxes. The script defaults to
  job-applicator's own `.venv/bin/vllm`, `GPU_MEM=0.70`, `MAX_MODEL_LEN=8192`, and
  `ENFORCE_EAGER=1` (avoids vLLM 0.23's V1 cudagraph-profiling OOM on 12 GB cards).
- Instructor for structured LLM outputs (Pydantic models)
- mxbai-embed-large-v1 for semantic job matching (~1.5 GB VRAM)
- Style analyzer with persistent cache and multi-document support
- Combined scoring: 60% semantic similarity + 40% skill coverage
- **Never automate login.** Seed a session once as a human (`login` headed flow, or
  `import-cookies --from-browser`); the tool only reuses it. Automated sign-in trips anti-bot
  defenses and risks the account.
- **Region-aware browser.** Locale, IANA timezone, and Chrome UA are auto-detected
  (`utils/region.py`) unless pinned in `[browser]` config, so geo-aware boards serve the real region.
- **A board declares its browser needs.** `BaseScraper.browser_policy()` (headed / ephemeral
  profile / virtual display) lives on the scraper, not the CLI, so anti-bot requirements can't drift
  and any caller builds the right browser. `_make_browser` (in `factories.py`) reads it.
- **Easy Apply is dry-run by default;** real submission requires `apply --submit`. Dry runs generate cover letters as a preview when `--cover-letter` is enabled and a résumé path is configured; the generated letter is surfaced in `--json` and the console table.
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

## GPU Memory Layout

| Component | Allocation |
|---|---|
| vLLM (Qwen3.5-4B-AWQ, eager mode) | ~6.5 GB |
| Embeddings (mxbai-embed-large-v1) | ~1.5 GB |
| Free VRAM | ~4.0 GB |

## Embedding Service

- Model: `mixedbread-ai/mxbai-embed-large-v1` (1024 dimensions)
- Cache: `~/.job-applicator/embeddings/` (numpy arrays)
- Matching: Cosine similarity with combined scoring
- Skill threshold: 0.75 for semantic match (empirically tuned; 0.55 matched unrelated same-domain skills)
