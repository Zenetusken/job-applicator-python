# Production-Readiness Assessment — Job Applicator

- **Date:** 2026-06-15
- **Branch:** main (with uncommitted in-flight cookie-auth/stealth work)
- **Author:** investigation pass (Claude) — empirically verified where noted

## TL;DR

The app is in **much better shape than "broken" implies**. The core document/LLM/
embeddings/match/tailor/batch pipeline is healthy and the refactored browser layer
runs. There was **one confirmed app-killing defect** — an undeclared `playwright-stealth`
dependency introduced by the uncommitted work — which is **fixed in this pass**
(`pyproject.toml`). The remaining risk concentrates in the **live-browser layer**
(LinkedIn scraping + Easy Apply), which is inherently fragile (CAPTCHA) and has **not been
re-validated end-to-end since the refactor**.

## Verified health snapshot (this session)

| Check | Command | Result |
|---|---|---|
| Unit tests | `pytest tests/unit/ -q` | **375 passed** in 5.55s |
| Lint | `ruff check src/ tests/` | clean |
| Format | `ruff format --check src/ tests/` | 63 files formatted |
| Types | `mypy src/job_applicator/ --ignore-missing-imports` | no issues, 35 files |
| Imports | all declared deps + package modules | import OK |
| Dependency closure | AST scan of `src/` imports vs declared deps | only `playwright_stealth` was undeclared (now fixed) |
| Browser refactor | headless `launch_persistent_context` + stealth + navigate + stop | **runs**; `navigator.webdriver=False`, realistic UA |
| LLM backend | `curl localhost:8000/v1/models` | vLLM live, `cyankiwi/Qwen3.5-4B-AWQ-4bit` |

> Confidence note: unit tests use **mocked** LLM/embeddings/browser. The core pipeline is
> "unit-tested + previously live-validated (live report dated 2026-06-14, **pre-refactor**),
> **not re-validated end-to-end this session**." Subsystem maps below are *mapped, not audited.*

## Architecture & key patterns (the map)

**Layering** (`src/job_applicator/`):
- `cli.py` — Typer CLI orchestrator (~2.3k lines). Commands: `search`, `apply`, `match`,
  `batch`, `tailor`, `generate-cover-letter`, `ats-check`, `config-init`. Heavy commands
  (`match`/`batch`/`tailor`/`apply`) wire together loaders, matcher, tailor, cover-letter,
  ATS, tone. `batch` runs up to 3 concurrent LLM ops via a semaphore.
- `config.py` — `AppSettings` + sub-configs (`Browser/LLM/Embedding/Target`). Single source
  of truth, loaded from `config.toml` + `JOB_APPLICATOR_*` env (env wins). No FS side effects
  on construct; `ensure_output_dir()` is explicit.
- `models.py` — all shared Pydantic v2 contracts (`extra="forbid"`). Sessions (`TailorSession`,
  `CoverLetterSession`) are plain in-memory classes (no persistence).
- `exceptions.py` — `JobApplicatorError` hierarchy; `LLMError` is a direct base subclass
  (not under `CoverLetterError`) by design.
- `browser/` — `manager.py` (Playwright lifecycle) + `actions.py` (navigate/click/fill/...).
- `scrapers/` — `base.py` (ABC) → `linkedin.py`; `indeed.py` is a **stub**.
- `applicators/` — `base.py` (ABC) → `linkedin.py` (Easy Apply); `indeed.py` is a **stub**.
- `documents/` — `resume.py` (parser, OCR fallback), `resume_tailor.py` (1.1k lines, LLM
  tailoring + date audit + hallucination guards), `cover_letter.py`, `style_analyzer.py`,
  `tone_detector.py` (keyword-based), `ats_checker.py`, `ocr.py` (PaddleOCR, lazy).
- `embeddings/` — `service.py` (mxbai-embed-large-v1, disk cache, FP16) + `matching.py`
  (`JobMatcher`: score = 0.6·semantic + 0.4·skill; skill threshold 0.55; non-greedy 1:1).
- `utils/` — logging (Rich→stderr), retry (`@async_retry`), diff, verbose (structured
  `--verbose`/`--json` reports), `llm.strip_thinking_process`, `text.contains_word`.

**Cross-cutting conventions** (enforced, observed green): Pydantic models cross module
boundaries (not dicts); typed exceptions only; async for I/O, sync for CPU; `from __future__
import annotations` everywhere; mypy strict; line length 100; double quotes; JSON→stdout /
logs→stderr.

**LLM stack:** litellm (universal) + instructor (structured) with manual-JSON fallback;
thinking suppressed via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
plus `strip_thinking_process()`; `max_tokens` from config (default 4096); vLLM needs
`openai/` model prefix. Hallucination guards in `resume_tailor.py`: `_validate_skills`,
`_strip_hallucinated_tools` (two-pass, alnum word boundaries), `_strip_hallucinated_education`.

**Browser/auth model (in-flight refactor):** single `launch_persistent_context` profile at
`~/.job-applicator/browser-profile/` (persists cookies/localStorage/etc.), stealth via
`playwright-stealth`, realistic default UA, `--disable-blink-features=AutomationControlled`.
Scraper + applicator **share one authenticated context** (`persistent_context()`/
`persistent_page()`). Cookie cache at `~/.job-applicator/cookies/linkedin.json`;
`search --save-cookies` forces fresh login.

## In-flight uncommitted work (cookie-auth + stealth/persistent-context refactor)

Plan: `docs/compose/plans/2026-06-15-cookie-auth.md`. Touches `browser/manager.py`,
`scrapers/linkedin.py`, `cli.py`, `tests/unit/test_browser_context.py`,
`tests/unit/test_scrapers.py`, `AGENTS.md`. Status: **coherent, tests pass, runs headless.**
It introduced the P0 dep gap (below) and some minor redundancy.

## Prioritized production-readiness gaps

### P0 — app-breaking (FIXED this pass)
- **`playwright-stealth` undeclared.** `browser/manager.py:10` imports it; it was missing from
  `pyproject.toml` and only present in the venv transitively via an unrelated `Crawl4AI`
  install. A clean `pip install -e ".[dev]"` would `ImportError` on startup → entire CLI dead
  (cli → scrapers → manager). **Fix applied:** added `playwright-stealth>=2.0` to dependencies
  (pin matters — `Stealth`/`apply_stealth_async` are 2.x-only; <2.0 exposes a different API).
  **Follow-up:** validate via a *fresh* venv install (current venv is polluted, so it can't
  prove the fix alone).

### P1 — core functional / validation
- **Re-validate end-to-end post-refactor.** Run the live E2E suite (`tests/test_tier1_live.py`,
  `test_tier2_live.py`, `test_batch_live.py`, `test_live_tailor.py`) — they need vLLM (up) +
  GPU embeddings + local fixtures, **not** network/LinkedIn, so they're runnable now. This is
  the real production check for the document/LLM/match/tailor/batch pipeline.
- **Finalize + commit the in-flight refactor** once P0 + E2E are green (currently a large
  uncommitted blob spanning 6 files).
- **Live LinkedIn scraping is best-effort, not reliable.** CAPTCHA/challenge blocking is the
  documented reality (it's why cookie-auth exists). Real `search`/`apply` against LinkedIn
  depends on a successful initial login or manually-exported cookies.
- **Easy Apply (`applicators/linkedin.py`) is unvalidated against real LinkedIn** and relies on
  brittle hard-coded CSS selectors / multi-step form assumptions.

### P2 — quality / robustness (check in before changing)
- **Indeed is a stub** (`scrapers/indeed.py`, `applicators/indeed.py`) though README lists
  "LinkedIn and Indeed."
- **`resume_tailor.tailor()` hardcodes `JobMatcher(EmbeddingConfig(device="cpu",
  memory_limit_gb=0.5))` when `matcher` is None** (`resume_tailor.py:642`) — ignores user
  config and can double-load models. Likely an intentional VRAM-contention guard; confirm CLI
  paths always pass a `matcher`.
- **Live test files are unmarked** — they escape both `-m unit` and `-m integration`; only bare
  `pytest` collects them (396 vs 375). Add an `integration`/`live` marker + a vLLM skip guard.
- **Redundant stealth application** — applied to the context in `start()` (which auto-hooks new
  pages) *and* again per-page in `persistent_page()` and the scraper's `_new_stealth_page()`.
  Harmless but noisy.
- **Vestigial `self._browser`** field + dead `if self._browser:` branch in `stop()`, and an
  unused `new_page()` alias, after the persistent-context refactor.
- **`cover_letter.generate_from_template()` appears orphaned** (sync, no LLM, no callers).
- **Pervasive silent `except Exception` fallbacks** (OCR, style cache, JSON parsing,
  `_summarize_changes`) — by design (graceful degradation) but reduce observability.

## Recommended next steps (ordered)

1. **P0 fix is applied.** Confirm with a fresh-venv install (gold standard).
2. **Run the live E2E suite** (tier1 + tier2 first) to re-validate the core pipeline against
   the real LLM post-refactor.
3. **Finalize + commit** the cookie-auth/stealth refactor (now that the dep is declared).
4. Decide the **scope of "production-ready"** — offline pipeline only, vs. also hardening live
   LinkedIn scraping/Easy Apply, vs. also delivering Indeed — then work P1/P2 accordingly.
