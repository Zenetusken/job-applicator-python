# MEMORY.md

Project memory for job-applicator-python. Consolidated facts about the codebase,
decisions, and current state. Keep under ~200 lines; prune stale entries when adding.

_Last synced: 2026-06-15_

## Snapshot

- **Stats:** 27 source modules (`src/job_applicator/`), 365 unit tests (all fast, no browser/GPU).
- **Python:** 3.12+ (dev box 3.12.8). Mypy strict; ruff (100-char lines, double quotes).
- **Quality gates (all must pass, in order):**
  `ruff check src/ tests/` → `ruff format --check src/ tests/` →
  `mypy src/job_applicator/ --ignore-missing-imports` → `pytest tests/unit/ -v`.
- **Install:** `python3.12 -m venv .venv && pip install -e ".[dev]"`. The optional
  `[embeddings]` extra (sentence-transformers + CUDA torch) is NOT needed for gates.
- **Browser flows:** `playwright install chromium` once.

## Architecture (single source of truth: AGENTS.md)

- `cli.py` — Typer CLI: search, apply, match, batch, generate-cover-letter, tailor, ats-check, config-init.
- `config.py` — `AppSettings` + sub-configs; loads `config.toml` (lowest priority) + `JOB_APPLICATOR_*` env.
- `models.py` — all shared Pydantic contracts (`extra="forbid"`).
- `documents/` — resume parsing, tailoring, cover letters, style/tone, ats_checker, ocr.
- `browser/` `scrapers/` `applicators/` — Playwright lifecycle + LinkedIn/Indeed.
- `embeddings/` — mxbai-embed-large-v1 service + job matching.
- `utils/` — logging, retry, diff, verbose, **llm (strip_thinking_process), text (contains_word)**.

## Key Decisions / Invariants

- Pydantic models cross module boundaries, never dicts. All exceptions subclass `JobApplicatorError`.
- Async for I/O, sync for CPU. Config centralized in `AppSettings`; no global mutable state.
- Combined match score = 60% semantic + 40% skill coverage. Skill semantic threshold 0.55.
- LLM via litellm + instructor; vLLM at `http://localhost:8000/v1` (model `cyankiwi/Qwen3.5-4B-AWQ-4bit`),
  `openai/` prefix for local. Suppress Qwen reasoning via `enable_thinking: False` + `strip_thinking_process()`.
- Resume-tailoring hallucination guards must be preserved (skills/tools/education validation,
  fuzzy `_skills_match()` ratio ≥ 0.85, `KNOWN_HEADERS` frozenset). See AGENTS.md gotchas.

## Audit (code sanity check) — status

Full audit produced 4 HIGH, 7 MEDIUM, 10 LOW findings. All fixed across three stacked PRs:

- **PR #6** — config (`config.toml` now actually loads), credential message, PII removal from matching,
  non-greedy skill matching, word-boundary tool stripping, `max_tokens` honored, parser/tailor header
  alignment, mypy green.
- **PR #7** — H-4/L-2: scraper + applicator now share one authenticated browser context via
  `BrowserManager.persistent_context()`/`persistent_page()`; removed `_browser._browser` leak;
  error screenshot captures the real failure page.
- **PR #8** — LOW findings: `LLMError` → direct `JobApplicatorError`; `strip_thinking_process` moved to
  `utils/llm.py` (re-exported); no filesystem side effects in config (`ensure_output_dir()`);
  word-boundary matching for tone/ATS (`utils/text.contains_word`); single ATS model
  (`ATSCompatibilityResult.is_compatible` computed); dead-code removal; `detect_seniority` uses
  description fallback; PaddleOCR `<3.0` pin documented; ATS suggestions skip optional sections.

## Recurring Gotchas (see AGENTS.md for the full list)

- vLLM/embedding models are not on the CI/dev VM — LLM/embedding paths are exercised with mocks.
- LinkedIn login uses Playwright locator API (`input[type="email"]`), not removed `name=` attributes.
- Authenticated browser work must use `persistent_context()`/`persistent_page()`, never `new_page()`.
- `config.toml` holds credentials — do not commit it (`.gitignore`d).

## Workflow

- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`; scopes like `fix(ocr):`).
- Keep AGENTS.md authoritative (architecture tree, test count, gotchas) and in sync with code.
- Feature flow: spec → plan → report under `docs/compose/`. Custom commands in `.mimocode/command/`.
