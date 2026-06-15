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

# Tests (437 unit tests, all fast; 458 total incl. integration-marked)
pytest tests/unit/ -v
pytest tests/unit/ -v -k test_name  # single test

# CLI
job-applicator --help
job-applicator match --resume resume.pdf
```

## Architecture

```
src/job_applicator/
├── cli.py              # Typer CLI (search, login, import-cookies, apply, match, batch, generate-cover-letter, tailor, ats-check, config-init)
├── config.py           # AppSettings + sub-configs (BrowserConfig, LLMConfig, EmbeddingConfig, TargetConfig)
├── models.py           # All shared Pydantic models (JobListing, ResumeData, StyleGuide, TailoredResume, DateAuditResult, etc.)
├── exceptions.py       # JobApplicatorError hierarchy
├── browser/            # Playwright lifecycle (manager.py) + low-level actions (actions.py)
├── scrapers/           # base.py (ABC) → linkedin.py, indeed.py (both live; Indeed selectors tuned 2026-06-15)
├── applicators/        # base.py (ABC) → linkedin.py (Easy Apply, dry-run gated), indeed.py
├── documents/          # cover_letter.py (LLM), resume.py (parser), resume_tailor.py (tailoring), style_analyzer.py, tone_detector.py, ats_checker.py, ocr.py
├── embeddings/         # service.py (mxbai-embed-large-v1), matching.py (job matching)
└── utils/              # logging, retry, diff, verbose, llm (strip_thinking_process), text (contains_word),
                        #   cookies (save/load/read), region (locale/tz/UA detection), url (host_matches), secure_store (atomic 0600 writes)
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

- **LLM output has thinking process.** Qwen models prepend reasoning. Use `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` in litellm calls to suppress. Fallback: `strip_thinking_process()` lives in `utils/llm.py` (re-exported from `cover_letter.py` for backward compatibility); import it from `utils.llm`.
- **Resume tailoring hallucination guards.** `_validate_skills()` strips hallucinated skills. `_strip_hallucinated_tools()` replaces tools not in original resume. `_strip_hallucinated_education()` removes education if original has none.
- **Education extraction must be explicit.** LLMs silently drop education entries. `_extract_education_entries()` injects a numbered checklist into the prompt to force inclusion.
- **Embeddings need `openai/` prefix for vLLM.** `model = f"openai/{config.model}"` when calling litellm.
- **Resume PDF parser is fragile.** Skills extraction breaks on bullet-per-line PDFs. The parser handles this, but verify with `ResumeLoader.load()`. `_extract_skills_section()` recognizes qualified headers too (`Technical/Core/Key/Professional/Relevant/Soft Skills`, `Competencies`, `Proficiencies`) — keep this aligned with the tailor's section headers.
- **`sentence-transformers` needs CUDA torch.** If you get `libcudart.so` errors, reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **Embedding cache at `~/.job-applicator/embeddings/`.** Style cache at `~/.job-applicator/styles/`. Clear with `EmbeddingService.clear_cache()`.
- **Skill matching threshold is 0.55.** Lower = more matches, higher = stricter. Tune in `matching.py:_match_skills()`.
- **`parse_sections()` uses known headers.** Matches against `KNOWN_HEADERS` frozenset (case-insensitive) and Title Case with colon suffix. ALL CAPS names (e.g. "JOHN DOE") are NOT matched as headers. Add new headers to the frozenset in `resume_tailor.py` if needed.
- **Skill validation uses fuzzy matching.** `_skills_match()` in `resume_tailor.py` checks exact match, token containment (subset), and `SequenceMatcher` ratio >= 0.85. Prevents "ai" matching "training" while catching typos.
- **Tool hallucination has two passes.** Pass 1: checks job requirements not in original. Pass 2: checks `tool_replacements` keys in tailored text not in original AND not in requirements. Catches LLM-invented tools.
- **Tool stripping uses alphanumeric word boundaries, not `re.escape` substrings.** `_alnum_boundary_pattern()` in `resume_tailor.py` wraps terms in `(?<![A-Za-z0-9])…(?![A-Za-z0-9])` so a `Java` requirement never corrupts `JavaScript` (and `React` isn't considered present just because the original says `Reactive`). It still matches symbol-ending terms like `C++`.
- **`LLMConfig.max_tokens` is honored everywhere (default 4096).** `resume_tailor.py`, `cover_letter.py`, and `style_analyzer.py` all pass `self._config.max_tokens` to litellm/instructor — do not hardcode `1024`/`4096`. 4096 fits a full résumé; cover letters/style analysis stay well under it.
- **`config.toml` is actually loaded.** `AppSettings.settings_customise_sources()` adds a `TomlConfigSettingsSource` as the lowest-priority source (env vars override it). Point at an alternate file via `JOB_APPLICATOR_CONFIG_FILE`; a missing file is a no-op.
- **Skill matching is not greedy 1:1.** `_match_skills()` skips skills already claimed by an earlier requirement and falls back to the next-best *available* skill, so two requirements competing for one skill no longer falsely mark the loser as "missing".
- **No hardcoded PII in matching.** `JobMatcher._is_pii_or_noise()` filters bullet glyphs, the candidate's own name, and bare email/contact lines generically — never hardcode a specific name/email.
- **`tailor()` accepts optional `tone_profile`.** When provided, skips internal `ToneDetector.detect()`. Eliminates double detection when CLI already computed the profile.
- **`refine()` accepts optional `tone_profile`.** When provided, injects tone directives into the refinement prompt. Without it, refinement loses tone alignment.
- **Tone directives in system prompts.** `TAILOR_SYSTEM_PROMPT` has a `TONE` section telling the LLM to use specified verbs, emphasize themes, avoid patterns. `cover_letter.py` `SYSTEM_PROMPT` replaces static "professional but personable" with dynamic tone awareness. Both generators follow tone when provided, fall back to professional tone when not.
- **`_detect_tone(job)` is the shared helper.** Used by `tailor`, `batch`, `generate-cover-letter`, and `_cover_letter_workflow`. Deterministic keyword-based detection, no LLM.
- **`format_for_prompt()` produces actionable directives.** Returns `"Use these action verbs: ..."` / `"Emphasize: ..."` / `"Avoid: ..."` instead of labels. Returns `"Match the job posting's natural tone."` for `unknown` tone.
- **Batch per-job tone.** Each job in batch gets its own `ToneProfile` via `_detect_tone(job)` and its own `TailoringReport`. `record_batch_tailoring()` accumulates per-job results. `VerboseReport.batch_tailoring` holds the list.
- **`refine()` recomputes match scores.** Creates synthetic `ResumeData` from refined text and runs `JobMatcher.match_resume_to_job()`. No more stale scores.
- **`CoverLetterGenerator.refine()` exists.** Uses same structured generation pipeline as `generate()` — system prompt, style guide, tone section, instructor fallback. `_refine_cover_letter()` in cli.py delegates to it.
- **Tone detection is keyword-based, not LLM-based.** `ToneDetector.detect()` in `tone_detector.py` uses keyword frequency heuristics — fast, but may misclassify edge cases (e.g. a startup posting heavy on compliance jargon).
- **Max tailor retry limit is 10.** A warning prints at attempt 8. The limit is hardcoded in `cli.py` and `tailor_cgi.py` — search for `attempt > 10` to adjust.
- **`TailorSession` is in-memory only.** Version history is lost when the session ends. No persistence to disk.
- **Cover letter sub-loop has no `[S] Section` option.** Cover letters lack parseable sections, so the section-editing prompt is skipped in the cover letter flow.
- **`CoverLetterResult` is simpler than `TailoredResume`.** No `match_score`, `matched_skills`, or `semantic_score` — cover letters don't go through embedding-based matching.
- **Resume meta.json write is deferred until after cover letter flow.** The CLI waits until the cover letter sub-loop completes (or is skipped) before writing the resume's sidecar metadata, so `cover_letter_path` can be included.
- **`cover_letter_path` in `TailoredResume` links resume to cover letter.** After the cover letter is saved, its path is stored in the resume model for downstream consumers.
- **`MatchResult` has `semantic_score` and `skill_score` fields.** Raw component scores stored alongside the combined `score`. `resume_tailor.py` uses these directly — never recompute from combined score.
- **`_refine_cover_letter()` returns `bool`.** `True` on success, `False` on failure. Caller checks `if not result:` — not `if result is None:`.
- **Batch command loads style guide independently of `--cover-letter`.** Providing `--style-guide --no-cover-letter` still applies the style guide to resume tailoring.
- **`detect_seniority()` is a standalone utility.** Not auto-called on `JobListing` creation. Consumers call it explicitly or populate `seniority` field manually. The title is the strongest signal and takes precedence; the `description` arg is consulted only as a fallback when the title is inconclusive.
- **Config validation has no filesystem side effects.** `AppSettings` does not create the output dir on construction. Callers create it explicitly right before writing via `settings.ensure_output_dir()` (returns the `Path`).
- **One ATS result model.** `ATSCompatibilityResult` is the single ATS contract; `is_compatible` is a `@computed_field` (`score >= 0.6`) so it serializes. There is no separate `ATSReport`.
- **Word-boundary matching for tone + ATS sections.** `tone_detector.py` and `ats_checker.py` match keywords/section headers via `utils.text.contains_word()` (alphanumeric boundaries), so `api` never matches inside `therapist` and `education` never matches inside `educational`.
- **ATS suggestions skip optional sections.** `_generate_suggestions()` never nags to add optional `Certifications`/`Languages` sections; those still appear as informational checks but not as suggestions.
- **`pdftotext` uses `-layout` flag.** Preserves multi-column resume formatting. Temp files cleaned up via `try/finally`.
- **DOCX support via `python-docx`.** `ResumeLoader.load()` dispatches `.docx` to `_load_docx()` using `Document(path).paragraphs`.
- **OCR fallback via `paddleocr`.** `ResumeLoader.load()` accepts `ocr_mode={auto,on,off}`. Auto mode falls back to OCR when extracted text is < 100 chars. Image resumes (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`) use OCR directly. `--force-ocr` CLI flag forces OCR on all resume-loading commands.
- **OCR models are lazy-loaded.** `OCRService` initializes PaddleOCR only on first extraction. First import triggers ~500MB+ model downloads.
- **PyMuPDF is required for PDF OCR.** `extract_text_from_pdf` uses `fitz.open()` + page pixmap → temp PNG → OCR. Temp files cleaned up via `try/finally`.
- **LinkedIn login uses Playwright locator API.** LinkedIn removed `name` attributes from login form. Use `page.locator('input[type="email"]').last` and `page.locator('button:has-text("Sign in"]').last`. Do NOT use `input[name="session_key"]` — it no longer exists.
- **Browser uses stealth + persistent profile.** `BrowserManager` uses `launch_persistent_context()` with a Chrome user data directory at `~/.job-applicator/browser-profile/`. This preserves ALL browser state (cookies, localStorage, IndexedDB, service workers, history) between runs. Stealth patches from `playwright-stealth` are applied to every page. Anti-automation flags are disabled via `--disable-blink-features=AutomationControlled`. The persistent context is closed in `BrowserManager.stop()`.
- **LinkedIn auth = seed once, reuse the session. Automated login is DISABLED (account safety).** `LinkedInScraper.login()` no longer submits credentials — programmatic login is exactly what trips LinkedIn's risk-based CAPTCHA and raises the account's risk score. Instead run `job-applicator login` once: it opens a HEADED browser, pre-fills config creds, and waits while the human clicks Sign in + solves any CAPTCHA/2FA. The authenticated session is retained by the persistent Chrome profile (`~/.job-applicator/browser-profile/`) and reused headlessly across runs (verified to persist across separate processes). `scrape()` calls `_ensure_session()` (loads the optional cookie JSON, checks `/feed`) and raises `LoginRequiredError` pointing at `job-applicator login` when there's no session — it never auto-logs-in. `_scrape_listings()` (the retried inner method) only retries on `NavigationError`, so a no-session run fails fast without repeated hits.
- **Cookie JSON is both a portable backup and a first-class auth path.** `~/.job-applicator/cookies/{linkedin,indeed}.json` is written by `interactive_login` after sign-in AND by `import-cookies`, and loaded by `_ensure_session`/`scrape`. The persistent profile already carries the LinkedIn session so the JSON is redundant *there*, but `import-cookies --from-browser` is the supported way to seed a session from your everyday browser (incl. httpOnly cookies like `li_at`/`cf_clearance` that page scripts can't read).
- **Region/UA/timezone are auto-detected at launch (`utils/region.py`).** `detect_timezone()` reads `TZ` → `/etc/localtime` → `/etc/timezone`, strips the `posix/`/`right/` zoneinfo prefixes, and **validates each candidate against the IANA db before returning** — a non-canonical/bogus value must never reach Playwright's `timezone_id` (it raises at launch). Windows without `TZ` falls back to the default; pin `browser.timezone`. `detect_chrome_user_agent()` is `lru_cache`d (shells out to `chrome --version`; anchor the major-version regex on the `Chrome`/`Chromium` keyword so Brave's `1.71.x Chromium: 130` yields 130). `detect_locale()` tolerates numeric/script subtags (`es-419`, `zh_Hans_CN`→`zh-CN`).
- **`import-cookies` is driven by a per-site `_SiteSpec` (cli.py).** Each board declares `required_cookie` (hard-fail if absent — LinkedIn `li_at`), `preferred_cookie` (warn only — Indeed `cf_clearance`, since search is public), `session_flags` (LinkedIn-only `--li-at`/`--jsessionid`), and `feed_verify` (post-import logged-in check; off for Cloudflare-fronted Indeed). Add a board by adding a spec entry — don't sprinkle `if site == ...` branches.
- **One host matcher: `utils/url.host_matches(host, base)`.** Used by `import-cookies`' look-alike filter (browser_cookie3's domain filter is a substring match that sweeps in `notlinkedin.com`) and Indeed's `_is_indeed_host`. Strips a leading `.` (cookie-domain form) and matches exact-or-subdomain only. Don't re-implement domain-suffix checks.
- **Indeed is public + Cloudflare-fronted.** No login (`IndeedScraper.login()` returns False). `scrape()` loads `cf_clearance` + the warm session via `load_cookies`, follows Indeed's region redirect (pins the regional host it lands on, e.g. `ca.indeed.com`, in `_resolved_base`; `target.indeed_domain` pins one explicitly), and raises `ScraperError` on an active challenge. Cookie+UA reuse improves the odds but Cloudflare's TLS-layer fingerprinting can't be fully reproduced by Playwright — a challenge is still possible.
- **Secure cookie writes (`utils/secure_store.write_secret_json`).** Atomic (`mkstemp`+`os.replace`) at mode `0600`, and refuses a symlinked path/parent. `utils/cookies.save_cookies` wraps it with the `{"cookies": [...]}` envelope; `load_cookies` is all-or-nothing-tolerant (falls back to per-cookie add so one malformed entry can't void the batch).
- **If a session won't establish:** re-run `job-applicator login` and complete the sign-in; solve any challenge in the window (a human solving it clears suspicion). A VPN/clean IP helps if the account/IP is already flagged. Stealth lowers fingerprint detection but cannot solve a CAPTCHA or clear an existing block. Note: automated scraping/Easy-Apply still carries inherent LinkedIn-ToS risk — keep volume low.
- **Scraper AND applicator share one authenticated context.** `BrowserManager.persistent_context()` returns a single context that lives for the manager's lifetime; `persistent_page()` opens a page in it and closes only the page. The scraper's `login()`/`scrape()` and the applicator's `apply()`/`check_already_applied()` all go through it, so the login session is reused for Easy Apply. Do NOT use `self._browser.new_page()` / `new_context()` for authenticated flows (those are isolated and logged-out), and do NOT reach into `self._browser._browser` — use the public `persistent_context()`/`persistent_page()` API. The persistent context is closed in `BrowserManager.stop()`. When `user_agent`/`locale`/`timezone` are unset (the default), `BrowserManager.start()` auto-detects them via `utils/region.py` (`detect_chrome_user_agent`/`detect_locale`/`detect_timezone`) so sites see a real desktop UA matching the host's installed Chrome and the host's real region — not "HeadlessChrome" or a hardcoded US default. There is no longer a `DEFAULT_USER_AGENT` constant in `manager.py`.
- **Applicator screenshots the failed page in place.** On error, `LinkedInApplicator.apply()` screenshots the *same* page that failed (it no longer re-navigates a fresh page, which used to hide the real failure state).
- **LinkedIn description extraction clicks cards.** Scraper clicks each job card, waits for content change, clicks "show more" button, then extracts. Search page cards only have title/company/location — descriptions come from the detail panel.
- **`--verbose` and `--log-file` work both before and after the command.** `job-applicator --verbose match` and `job-applicator match --verbose` both work. `_merge_verbose_ctx()` in cli.py handles merging subcommand flags with global callback.
- **JSON output goes to stdout, logs go to stderr.** `sys.stdout.write()` for JSON, `RichHandler(console=Console(file=sys.stderr))` for logging. Enables `job-applicator match --json | jq .` without Rich wrapping corruption.

## LLM Setup

Local vLLM must be running at `http://localhost:8000/v1`. Check with:
```bash
curl -s http://localhost:8000/v1/models
```

Default model: `cyankiwi/Qwen3.5-4B-AWQ-4bit`. Override via `JOB_APPLICATOR_LLM_MODEL` env var or `config.toml`.

## ATS Compatibility Checking

`ATSChecker` in `documents/ats_checker.py` analyzes resumes for ATS compatibility:
- Email/phone presence
- Standard section headers (Experience, Education, Skills)
- Text length (minimum 200 chars)
- ASCII table detection (ATS can't parse these)

CLI usage: `job-applicator ats-check --resume resume.pdf [--json]`
Score >= 60% = compatible. Returns warnings and actionable suggestions.

**Integrated checks:** ATS compatibility is automatically checked before `tailor`, `match`, `apply`, and `batch` commands. Warnings shown if score < 60%. Post-tailor verification shows before/after comparison.

## Testing

- All tests are `pytest -m unit` (no browser, no GPU needed)
- Tests use fixtures from `tests/conftest.py`
- Embedding tests mock the model (CPU fallback)
- `scripts/smoke_test_match.py` — real resume matching (needs GPU)
- `scripts/detailed_match_report.py` — rich per-skill match breakdown
- `scripts/tailor_cgi.py` — resume tailoring for CGI job (needs vLLM)
- `scripts/test_e2e.py` — full pipeline (needs vLLM running)

## Files Not to Commit

- `config.toml` (contains credentials)
- `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
- `output/`, `screenshots/`, `logs/`
