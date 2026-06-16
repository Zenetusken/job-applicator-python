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

- `pytest tests/unit/` — fast, isolated unit suite (461 tests; no browser/GPU/vLLM). The green gate.
- `pytest` — everything (482); the extra 21 are live tests at `tests/` root that need vLLM
  (`localhost:8000`) + GPU to run, so they're kept out of the fast gate (full suite is green when
  vLLM is up).
- Split is by directory (`tests/unit/` vs the live `tests/test_*_live.py`); the `unit`/`integration`
  markers in `pyproject.toml` are declared but not applied, so `pytest -m unit` collects nothing.

## Target Boards

- LinkedIn (Phase 1) — implemented. Session-authenticated (reuse a human-established session).
- Indeed (Phase 2) — implemented. Public search, Cloudflare-fronted; selectors tuned against the
  live DOM (2026-06-15) with region auto-detection. The wall is a Cloudflare *managed JS challenge*
  that blocks headless Chrome (not TLS/JA3, not rate-limit), so Indeed runs **headed** on a clean
  profile (windowless via Xvfb) — declared by `IndeedScraper.browser_policy()`.

## Key Design Decisions

- Headless browser by default, `--headed` flag for debugging
- AI-powered cover letters via litellm (works with local vLLM or cloud APIs)
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
  and any caller builds the right browser. `cli._make_browser` reads it.
- **Easy Apply is dry-run by default;** real submission requires `apply --submit`.

## GPU Memory Layout

| Component | Allocation |
|---|---|
| vLLM (Qwen3.5-4B-AWQ) | ~7.2 GB |
| Embeddings (mxbai-embed-large-v1) | ~1.5 GB |
| Free VRAM | ~3.3 GB |

## Embedding Service

- Model: `mixedbread-ai/mxbai-embed-large-v1` (1024 dimensions)
- Cache: `~/.job-applicator/embeddings/` (numpy arrays)
- Matching: Cosine similarity with combined scoring
- Skill threshold: 0.55 for semantic match
