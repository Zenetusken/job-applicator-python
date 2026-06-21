# MEMORY.md

Project memory for job-applicator-python. Consolidated facts about the codebase,
decisions, and current state. Keep under ~200 lines; prune stale entries when adding.

_Last synced: 2026-06-19_

## Snapshot

- **Stats:** 44 source modules (`src/job_applicator/`), 544 fast unit tests (`pytest -m unit` — the green gate, no browser/GPU); 565 total, the extra 21 are live tests (`-m live`) needing vLLM (`localhost:8000`) + GPU. Tests auto-marked by location in `tests/conftest.py`. Live tests now skip cleanly when the configured LLM endpoint is unreachable.
- **Python:** 3.12+ (dev box 3.12.8). Mypy strict; ruff (100-char lines, double quotes).
- **Quality gates (all must pass, in order):**
  `ruff check src/ tests/` → `ruff format --check src/ tests/` →
  `mypy src/` → `pytest -m unit`.
  (Untyped third-party imports — paddleocr, fitz, playwright_stealth, browser_cookie3 —
  are silenced via per-module `ignore_missing_imports` overrides in `pyproject.toml`,
  so no `--ignore-missing-imports` flag is needed.)
- **Install:** `python3.12 -m venv .venv && pip install -e ".[dev]"`. Optional extras:
  `[embeddings]` (sentence-transformers + CUDA torch), `[browser]` (browser-cookie3, for
  `import-cookies --from-browser`) — neither is needed for the gates.
- **Browser flows:** `playwright install chromium` once.

## Architecture (single source of truth: AGENTS.md)

- `cli.py` — Typer CLI: search, login, import-cookies, apply, match, batch, generate-cover-letter, tailor, ats-check, config-init, doctor. `doctor` (→ `diagnostics.py`) probes the LLM endpoint (`/v1/models`) + embeddings cache + self-host prereqs; only endpoint reachability is blocking.
- `config.py` — `AppSettings` + sub-configs; loads `config.toml` (lowest priority) + `JOB_APPLICATOR_*` env. `BrowserConfig` has `locale`/`timezone` (empty=auto); `TargetConfig` has `indeed_domain`.
- `models.py` — all shared Pydantic contracts (`extra="forbid"`).
- `documents/` — resume parsing, tailoring, cover letters, style/tone, ats_checker, ocr.
- `browser/` `scrapers/` `applicators/` — Playwright lifecycle + LinkedIn (session) / Indeed (public, Cloudflare). Both scrapers/applicators live.
- `embeddings/` — mxbai-embed-large-v1 service + job matching.
- `utils/` — logging, retry, diff, verbose, **llm (strip_thinking_process + CircuitBreaker + ValidatedOutput), text (contains_word), cookies (save/load/read), region (locale/tz/UA detect), url (host_matches), secure_store (atomic 0600)**.
- `state.py` / `batch_state.py` — SQLite stores for application history and batch-run progress (crash recovery).
- `skills/` — skill-name normalization and hard-negative filtering for matching/validation.

## Key Decisions / Invariants

- Pydantic models cross module boundaries, never dicts. All exceptions subclass `JobApplicatorError`.
- Async for I/O, sync for CPU. Config centralized in `AppSettings`; no global mutable state.
- Combined match score = 60% semantic + 40% skill coverage. Skill semantic threshold 0.55.
- Skills are normalized before matching/validation (`Python 3` → `Python`, `reactjs` → `React`); generic traits (`team player`, `communication`) are hard-negative filtered so they don't distort skill scores.
- Apply is dry-run by default; `--submit` opt-in required. `--validate` exits non-zero if a dry run doesn't reach the Submit button. `DryRunValidation` records reachability, fields filled, resume upload, and cover-letter field presence.
- LLM via litellm + instructor; **client of an external** OpenAI-compatible endpoint (`[llm] api_base`,
  default `http://localhost:8000/v1`, model `cyankiwi/Qwen3.5-4B-AWQ-4bit`) — the app never starts one
  (optional `[serve]` extra + `scripts/serve-vllm.sh` self-host a local vLLM). `openai/` prefix for local.
  Suppress Qwen reasoning via `enable_thinking: False` + `strip_thinking_process()`.
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

## Auth, Indeed & Region (recent work, PRs #9–#14)

- **Auth model: never automate login.** `LinkedInScraper.login()`/`IndeedScraper.login()` never
  submit credentials (automated login trips anti-bot + risks the account). Seed a session once via
  `job-applicator login` (headed) or `import-cookies --from-browser <chrome|…>` (reuses the everyday
  browser's cookie store, incl. httpOnly `li_at`/`cf_clearance`). Sessions persist via the Chrome
  profile + `~/.job-applicator/cookies/{linkedin,indeed}.json`.
- **`import-cookies` per-site `_SiteSpec`** — `required_cookie` (LinkedIn `li_at`, hard-fail) vs
  `preferred_cookie` (Indeed `cf_clearance`, warn only — search is public), `session_flags`,
  `feed_verify`. Add a board = add a spec entry, not `if site == …` branches.
- **Indeed = live; runs HEADED + ephemeral profile (Cloudflare managed challenge).** The wall is
  a Cloudflare JS challenge that blocks headless Chrome — NOT TLS/JA3 (bundled Chromium's JA4 ==
  real Chrome) and NOT rate-limit. Fix needs no special engine. **The browser policy lives on the
  board**: `BaseScraper.browser_policy() -> BrowserPolicy` (default headless/persistent),
  `IndeedScraper` overrides → headed+ephemeral+virtual_display. `cli._make_browser` READS the
  policy (no `if site == "indeed"`) and `_scraper_class(site)` validates the board before any
  launch. Windowless via Xvfb (`virtual_display`, optional `[indeed]` extra = pyvirtualdisplay;
  else ambient `$DISPLAY` / `xvfb-run`); `--headed` shows a real window. LinkedIn stays headless
  persistent. `scrape()` warns if given a headless browser. Indeed `search`/`batch` validated;
  `apply` wired-but-unvalidated. Full matrix: `docs/compose/reports/2026-06-15-indeed-cloudflare-research.md`.
- **Region auto-detect (`utils/region.py`)** — timezone from `TZ`→`/etc/localtime`→`/etc/timezone`,
  `posix/`/`right/` prefixes stripped, validated against the IANA db before reaching Playwright
  (a bad `timezone_id` crashes the launch). UA matches host Chrome major (`lru_cache`d). Windows
  w/o `TZ` falls back to default — pin `browser.timezone`. `detect_indeed_domain()` maps the
  timezone → ISO country via `/usr/share/zoneinfo/zone1970.tab`, then to `<cc>.indeed.com` only for
  countries in the `_INDEED_COUNTRIES` allowlist (else `www.indeed.com` — never a dead host);
  timezone, not the often-`en_US` locale, is the geo signal. `target.indeed_domain` pins explicitly.
- **Shared `utils/url.host_matches`** — single exact-or-subdomain matcher (strips leading `.`);
  used by the cookie look-alike filter and `_is_indeed_host`. Don't re-implement.
- **Easy Apply is dry-run by default.** `apply` fills forms but does NOT submit unless `--submit`;
  the final submit routes through `BaseApplicator._gated_submit`.

## Recurring Gotchas (see AGENTS.md for the full list)

- vLLM/embedding models are not on the CI/dev VM — LLM/embedding paths are exercised with mocks.
- LinkedIn login uses Playwright locator API (`input[type="email"]`), not removed `name=` attributes.
- Authenticated browser work must use `persistent_context()`/`persistent_page()`, never `new_page()`.
- `config.toml` holds credentials — do not commit it (`.gitignore`d).

## Round 2 Hardening (2026-06-19)

Completed a second systematic hardening pass with baseline capture and unit tests; live tests skip cleanly because vLLM is unavailable on the dev box.

- **Batch crash recovery** — `BatchState` in `batch_state.py` persists per-job progress in `~/.job-applicator/applications.db`. `job-applicator batch --resume-run` skips already-tailored jobs after an interruption; `--run-id` pins/resumes a specific run.
- **Skill normalization + hard negatives** — `skills/normalization.py` canonicalizes aliases (`Python 3` → `Python`, `reactjs` → `React`) and drops generic traits (`team player`, `communication`) from skill coverage scoring and tailored skill sections.
- **LinkedIn Easy Apply dry-run validation** — `DryRunValidation` reports whether the Easy Apply flow reached the Submit button, which fields were filled, resume upload status, and cover-letter field presence. `job-applicator apply --validate` exits non-zero if any dry run fails to reach Submit.

## Workflow

- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`; scopes like `fix(ocr):`).
- Keep AGENTS.md authoritative (architecture tree, test count, gotchas) and in sync with code.
- Feature flow: spec → plan → report under `docs/compose/`. Local harness/tooling config (`.mimocode/`) is gitignored, not tracked.
