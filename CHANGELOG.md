# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Target-role preference boosts** (`[[matching.target_roles]]`). Declared role families — a
  title regex + a boost — lift preference-important jobs the CV is lexically far from (measured:
  an AI-red-team and an IAM posting ranked below the review floor on an SOC CV; embedding
  interest-terms could NOT discriminate them from decoys, deterministic title patterns fired with
  zero false tags). Ranking-only: the boost never enters generated documents; fit scores
  (semantic/skill) stay pure. First match wins, clamp at 1.0; tagged 🎯 in the `match` table and
  as `target_role` in `match --json`. The boosted score is what `match` persists — the stored
  ranking is the preference-adjusted one. Gold-set validation: Spearman +0.736 → **+0.852**,
  0 missed-cyber / 0 false-positives at the fixed review floor.
- **Matching regression harness** (`scripts/eval_matching.py`). Scores the labeled gold set
  (`~/.job-applicator/matching-eval/`, personal data, not in the repo) through the LIVE pipeline —
  embeddings + extraction + boosts — and fails (exit 1) when Spearman drops below the STRONG bar
  or a non-security job crosses the fixed review floor. Run it whenever matching logic changes.
- **Live selector-health diagnostics.** New `job-applicator selector-health` probes LinkedIn/Indeed
  selector groups against live pages without scraping persistence or submitting applications. Search
  probes validate card/field/description selectors; LinkedIn apply probes validate Easy Apply entry
  and form controls. `search --selector-health` and `apply --selector-health` add opt-in preflights
  that abort on required selector drift unless `--ignore-selector-health` is supplied. JSON reports
  stay stdout-clean, and failure diagnostics land under
  `~/.job-applicator/debug/selector-health/`.
- **Private generated-document packet quality set.** The document-quality gate now supports a
  populated local packet manifest at `~/.job-applicator/document-quality-eval/packet-set.jsonl`,
  scored across usefulness, specificity, coherence, writing quality, and formatting polish. The
  private WSP IT-support seed packet is backed by fresh generated CV/cover-letter artifacts and a
  CV-coherent cover-letter gold-standard style fixture.
- **Packet-level CV/cover-letter coherence scoring.** Private document-quality packets now get a
  fifth 0-4 dimension that checks applicant identity, target role/company alignment, language
  consistency, and source-backed terms shared by the CV and cover letter. Manifests can set
  `coherence_terms` when the narrative bridge should be narrower than the broad keyword list.

- **`status` now shows which search surfaced each job.** The stored `source_query` (previously
  captured but never displayed) is surfaced as a **Found via** column in the `status` table, a
  `source_query` field in `status --json`, and a `· via 'X'` note on the TUI job card — so you can
  see a job's provenance. First-seen-wins, so post-dedup it reads "first found via X".
- **Scraper anti-detection hardening.** The browser now launches your **real host Chrome**
  (`[browser] channel = "chrome"`, the new default) instead of the bundled headless Chromium — no
  `HeadlessChrome` fingerprint leak, self-consistent version + WebGL (falls back to bundled Chromium
  with a warning if no host Chrome). And `search` honors a **proactive daily volume budget**
  (`[target] max_searches_per_day`, default 30; optional `search_cooldown_s`), refusing past the cap
  *before* launching the browser — so an authenticated LinkedIn session stays low-footprint. Basis:
  an audit found search was uncapped in code and the headless browser leaked fingerprint tells; a
  reused authenticated session means the goal is *unremarkable*, not undetected.
- **Résumé parser now extracts structured experience & education.** `parse_text` populates per-role
  `experience`/`education` entries (title / company / dates / bullets · degree / institution / dates)
  via a conservative, **multi-format** extractor (YYYY / Month YYYY en+fr / MM/YYYY / Present·présent),
  **degrading to empty** on formats it can't confidently parse rather than fabricating a field. This
  sharpens the cover-letter "returning candidate" company-match; job MATCHING is deliberately
  unchanged (guarded off — pending a gold-set re-validation). Previously these fields were declared
  but never populated (always empty).
- **`rescore` command — account-safe in-place re-scoring of the funnel.** Recomputes the match
  scores of *stored* jobs against the current résumé and writes them back in place (funnel stage
  preserved), **without re-scraping** — the only prior way to refresh scores was re-`search`, which
  re-scrapes (account-touching for LinkedIn). Fails closed (nothing written if matching raises) and
  reports before→after deltas (`--json` supported). Use it after the résumé changes so the saved
  scores reflect the current CV.
- **Tailored ATS + contact green-check in the interactive `tailor` view.** Each version now surfaces
  the tailored output's ATS score (before → after) and a contact-preservation check (email/phone
  survived), alongside the grounding report. Surfaced for review, never a gate; best-effort (an
  advisory line never aborts the review loop). ATS-on-tailored previously ran only in the verbose
  reporter + `batch`.
- **`doctor` surfaces the configured résumé's identity + age + parsed-skill count.** So a
  `resume_path` silently pointing at a stale/wrong CV (e.g. a 2-yr-old file, or one whose parse
  yields ~no skills) is *visible* — a filename/age a human catches instantly, where a threshold
  can't. A soft warning also fires on a thin parse (0 skills) or a >12-month-old file. Motivated by
  a real stale-config CV that had silently mis-scored the whole funnel.

### Fixed
- **Generated CV/cover-letter packet honesty hardening.** The v1 PDF résumé parser now handles the
  fixed-width skills grid without leaking row labels or splitting parenthetical skills, tailored CVs
  preserve source-owned Education/Languages sections, and generated summaries are rebuilt from
  source-backed phrases when the LLM draft is too dirty.
- **Batch CV saves now share the fail-closed grounding path.** A dirty batch-tailored CV gets one
  strict source-only refinement; if grounding/contact/ATS integrity is still not clean, the CV is
  not saved and the job records an error instead of producing a stale-looking artifact.
- **Cover letters reject unsupported JD-side and source-merge overclaims.** The generator now
  pre-prompts and validates against absent job-side terms such as workstation deployment, ITIL, and
  asset inventory, plus observed source-merge phrases such as unsupported technical-support
  coursework or critical-systems claims. Target company/role mentions are required as application
  context without implying prior employment.
- **Grounding audit false positives are narrower.** The deterministic verifier can supplement an
  incomplete model quote only with overlapping source fragments that carry the exact missing
  percentage, accepts supported numeric skill names such as Microsoft 365 / 802.1X, accepts
  source-backed employment-heading years, and ignores cover-letter application framing/courtesy
  lines as non-factual claims.
- **Style-analysis live diagnostics are no longer a black box.** `StyleAnalyzer` now logs which LLM
  path it is using (instructor structured output vs direct litellm JSON fallback), how long each path
  took, and the fallback reason. Direct litellm fallback failures now go through the shared
  `llm_call_error()` classifier, including a specific "socket permission denied" message for
  sandboxed localhost calls where vLLM is running but the runtime cannot open a network socket.
- **Glued-word repair for corrupted postings** (`scrapers/text_repair.py`). Some postings arrive
  with words mashed together (`Senti\nnelKQL`, `ge)Création`) — the board's own rich-text markup
  is misaligned mid-word (verified live in two render paths; not an extraction bug). The
  evidence-span verifier rightly refused those glued spans, silently costing real skills (KQL,
  Microsoft Security (E5)) on the best-fit posting. Both scrapers now apply a corruption-GATED
  space-insertion repair (split-only — it can never fuse two tokens into a fabricated skill;
  mid-word line breaks stay broken on purpose). Clean descriptions pass byte-identical; measured:
  gate fires on 2/39 real JDs, recovery = the full SOC skill set on the corrupted posting.
- **`match` is now reproducible.** Skill extraction is a factual task, but the extractor inherited
  the `[llm]` temperature (0.7, tuned for cover-letter prose), so the same JD grounded different
  skills across runs and match scores wandered. It now runs extraction at temperature 0 (greedy) —
  measured reproducible with no recall loss; the evidence-span verification still guards
  hallucinations.
- **LinkedIn reposts no longer inflate the funnel.** The funnel dedups by job URL, but LinkedIn
  serves the same job under many tracking-decorated URLs (`?eBP=…&trackingId=…`) that differ per
  search, so one job stored as several rows (measured: **53% of a 92-job funnel was phantoms**). The
  scraper now canonicalizes each job URL to its stable `/jobs/view/<id>` identity, collapsing reposts
  while keeping genuinely-distinct jobs separate.
- **LinkedIn search is now geo-correct.** `search --location` sent a raw location string with no
  numeric `geoId`, so LinkedIn ignored it and returned a global/remote feed (a `Montréal, QC` search
  measured **89% France/EMEA/EU-remote**). It now resolves the location to LinkedIn's `geoId` via the
  guest typeahead (region-aware, with a bare-city retry for `"City, ST"`) — Montréal searches return
  Montréal/Canada jobs. Falls back to the raw location (logged) if resolution fails, never a wrong id.
- **The LinkedIn scrape no longer drops job cards.** Card handles were captured once, then each card
  click re-rendered LinkedIn's virtualized list and detached the rest ("element not attached to the
  DOM"). The scrape now snapshots all card metadata first, then loads each description by re-resolving
  the card fresh by exact job id — a description that won't load degrades to a metadata-only listing
  instead of losing the whole job.
- **LinkedIn external apply is no longer mistaken for selector drift.** Live recon showed external
  postings render a button like "Apply to ... on company website", not the in-product Easy Apply
  button. The applicator now detects that surface and returns SKIPPED/manual follow-up without
  clicking it. Selector health reports external apply jobs as `skipped/ok`, while Easy Apply probes
  accept initial Next/Continue/Review controls because Submit often appears only after the form is
  advanced and filled.
- **Bare `match` ranks the saved funnel.** `match` with no `--jobs-file` errored "provide
  --jobs-file" instead of ranking the jobs `search` already stored (and that it writes scores back
  to). It now reads the funnel — `search` → `match` → `tailor` flows without re-exporting a JSON file.
- **`doctor` reports the browser engine actually used.** With the new default real-Chrome channel,
  `doctor`'s Browser check verified only the bundled Chromium; it now resolves and reports the host
  Chrome (and warns if the channel is set but no host Chrome is found → a silent fallback).
- **`--json` output no longer corrupted by the progress spinner under forced color.** `match` (and
  other data commands) routed their "Computing…" Rich status spinner to stdout; under a color-forcing
  environment (`FORCE_COLOR`, some CIs) that leaked ANSI/spinner frames into piped `--json`, breaking
  parsers. All progress now goes to stderr, keeping stdout pure regardless of the caller's color env.
  (The matching itself was always correct — only the JSON framing was affected.)
- **`tailor` no longer aborts on a valid CV; robust section-header matching.** Section headers were
  matched case-sensitively / exact-match, so all-caps qualified headers (`PROFESSIONAL EXPERIENCE`,
  `EDUCATION & CERTIFICATIONS`) silently fell through — making `summary` swallow ~97% of the document
  and, via a false "ordering issue" from mis-attributed date sections, **aborting `tailor`** with a
  "Proceed anyway?" prompt on a perfectly valid CV. A shared case-insensitive/qualifier-tolerant
  header matcher now drives both the summary boundary and the date-section attribution, and the
  tailor date-check is **advisory, never blocking** (the date parser is heuristic — it also drops
  MM/YYYY, "Current", and French formats). Also removed the education-age staleness heuristic (noise
  for an experienced candidate) and corrected the `ResumeDateValidator` docstring, which claimed
  gap/overlap detection it never implemented.
- **Skill extraction: keep short skills + stop paren-comma mangling.** A parenthetical skill with
  commas (`Linux (Fedora, CLI, Bash)`) was split into stray-paren garbage tokens; it now stays one
  skill (commas inside parentheses aren't split). And the hard-negative filter dropped any skill of
  length ≤ 2, silently losing `C#`/`Go`/`R`/`AI`/`ML` from coverage; it now drops only
  empty/pure-punctuation noise while keeping short real skills.
- **DOCX résumé parser now extracts table cells (contact header + skills).** `_load_docx` read only
  `doc.paragraphs`, silently dropping content in tables — real résumés put the contact header
  (name/email/phone) and the skills section in tables, so they never reached `raw_text`, and
  matching / grounding / tailoring all ran on a CV missing its contact and skills. It now walks the
  document body in order, extracting paragraphs **and** table cells (merged-cell-deduped, best-effort
  per table so a malformed grid can't fail the whole résumé). Recovers the contact block on tailored
  outputs and corrects match scores.
- **Markdown bold no longer leaks into the secondary `tailor` views.** The `[D]` full diff, `[V]`
  history preview, and `[S]` section preview rendered raw `**header**` markers; they now strip them
  like the main preview does.

### Changed
- **Grounding verifier now grounds faithful paraphrases and cross-language translations by meaning.**
  The honesty-verifier prompt was tuned so a faithful restatement of a résumé fact — a reworded or
  French↔English-translated version that adds and inflates nothing — is recognized as grounded
  rather than flagged for low word overlap (it previously over-flagged faithful French translations
  and low-overlap rephrasings). The added leniency is paired with an explicit inflation + **scope**
  guard: a claim that broadens scope (e.g. a metric the résumé limits to "the sales team" claimed
  for "the entire company") is still flagged. Validated EN+FR against adversarial inflations (now
  permanent gold-set recall guards); the deterministic English honesty floor still applies on top.

## [0.5.0] - 2026-06-29

### Added
- **Grounding verifier — a language-agnostic honesty layer for generated documents.** An LLM
  enumerates every claim in a generated CV or cover letter and cites the source line that grounds
  it; a deterministic audit (`documents/grounding_verifier.py`) then overrides any ungrounded claim
  (token-overlap + a numeric backstop covering percentages **and** standalone integers — years,
  counts, team sizes) and flags coverage gaps. The SOURCE is always the BASE résumé, never the job
  description or the tailored intermediate. Cover letters route through
  `CoverLetterGenerator.generate_verified()` (regenerate once, keep the strictly-cleaner draft);
  tailored résumés through `ResumeTailor.tailor_verified()`, which **surfaces** the report on
  `TailoredResume.grounding_report` for human review (a "claims to review" panel + a key in
  `tailor --json`) and **never auto-strips** — the résumé is the document of record. Fail-safe: any
  verifier failure raises `GroundingUnavailableError`, never passing off an unverified document as
  clean. Replaces per-language hardcoded honesty blocklists (which were unbounded and inert on
  French) with one entailment pass; the English deterministic floor is kept as augmentation.
- **Output-language policy.** `[llm] language` = `auto` (mirror the job posting's language) | `en` |
  `fr`. It lives on `[llm]` so the cover-letter override inherits it — the generated CV and cover
  letter always resolve the **same** language, so one application never mixes them. French resolves
  an in-language sign-off ("Cordialement,"), a localized PDF date, and recognized French closings.
- **Structured cover-letter generation.** Cover letters are generated as three connected paragraphs
  via a structured (instructor) step, with deterministic honesty guards and an enforced sign-off.

### Changed
- **Default base model is now `Qwen/Qwen3-8B-AWQ`** (genuine AWQ 4-bit, ~6.1 GB), replacing
  `cyankiwi/Qwen3.5-4B-AWQ-4bit` (kept as a pinnable fallback via `JOB_APPLICATOR_LLM_MODEL` /
  `[llm] model`). The 8B grounds stack-heavy job descriptions the 4B couldn't (measured: cover-letter
  employer-stack overclaim 5/6 → 0/5) while still fitting the 12 GB card alongside the embeddings
  (`GPU_MEM=0.70`, eager mode: ~8.4 GB vLLM + ~1.5 GB embeddings). Generation guardrails were
  re-tuned for the larger model, and `llm.max_tokens` raised to `4096`.
- **`scripts/serve-vllm.sh` puts the vLLM venv's `bin` on `PATH`** so flashinfer can JIT-compile a
  kernel for a fresh model (the 8B fails with `No such file or directory: 'ninja'` otherwise);
  `ninja` now ships in the `serve` extra.
- **Single source of truth for the litellm model id.** All completion callers build it via
  `utils.llm.litellm_model(config)` instead of re-deriving the `openai/` prefix in six places.
- **Docs corrected on how the CUDA 13.0 wheel is obtained.** `vllm 0.23` pins `torch==2.11.0`,
  whose PyPI wheel is the cu13 build (bundles `nvidia-*-cu13`), so a plain `pip install` lands the
  cu13 stack with no extra index; pip selects it unconditionally and it needs a CUDA-13 driver to
  run. (README, pyproject, AGENTS.md.)

### Fixed
- **Serve script no longer silently piggybacks on a `$PATH` vLLM.** `scripts/serve-vllm.sh` now
  uses only job-applicator's own `.venv/bin/vllm` or an explicit `VLLM_BIN`; if neither exists it
  errors with install/opt-in guidance instead of adopting whatever `vllm` is first on `$PATH`
  (historically a sibling project's), closing the last cross-project coupling in the self-host path.
- **`config.example.toml` no longer caps `max_tokens` at 1024.** It now ships `4096` to match the
  built-in default, so bootstrapping from the example doesn't silently truncate tailored résumés.

## [0.4.1] - 2026-06-25

### Changed

- **vLLM stack upgrade and isolation.** The `[serve]` extra now pins `vllm>=0.23,<0.24`, which
  pulls the CUDA 13.0 wheel on modern NVIDIA drivers. `scripts/serve-vllm.sh` now defaults to
  job-applicator's own `.venv/bin/vllm` so its CUDA/runtime configuration is fully isolated from
  other projects; `VLLM_BIN` can still point at a shared executable if desired.
- `scripts/serve-vllm.sh` defaults changed for 12 GB desktop GPUs:
  - `GPU_MEM=0.70` (was `0.60`).
  - `ENFORCE_EAGER=1` by default. vLLM 0.23's V1 engine cudagraph profiler allocates a large
    minimal KV-cache tensor during startup that ignores `--gpu-memory-utilization`, causing an
    OOM on 12 GB cards with Qwen3.5-style hybrid models. Eager mode avoids this and keeps the
    server stable. Users with more VRAM can set `ENFORCE_EAGER=0` for higher throughput.
  - `MAX_MODEL_LEN=8192` is now exposed as an env override for tuning context length vs. memory.
- `scripts/serve-vllm.sh` exports `LD_LIBRARY_PATH` so vLLM's optional `deep_gemm` path can find
  the CUDA 13.0 runtime libraries shipped with the vLLM cu130 wheel.
- Documentation (`README.md`, `AGENTS.md`, `CLAUDE.md`, `MEMORY.md`) updated to reflect the new
  vLLM version, isolated binary defaults, and realistic 12 GB VRAM allocations.
- `StyleAnalyzer` caps its per-call output budget at 1024 tokens (down from the global 4096
  default). Style-guide analysis produces a short JSON object, so the lower cap reduces KV
  reservation without affecting quality.
- Documented why instructor's default TOOLS mode is used for structured outputs: it relies on
  vLLM's `--tool-call-parser` (auto-set to `qwen3_xml` for Qwen3.5) and is faster and more
  schema-accurate than `json_object` or `guided_json` in our vLLM 0.23 tests.

## [0.4.0] - 2026-06-25

### Added

- PDF rendering support via the new optional `[pdf]` extra (`pip install 'job-applicator[pdf]'`),
  powered by Typst and Jinja2 templates.
  - Built-in résumé and cover-letter templates: `modern`, `classic`, and `minimal`.
  - `PDFRenderer` renders `TailoredResume` / `CoverLetterResult` to PDF through a
    structured LLM formatting step, escaped Typst source generation, and a
    `ProcessPoolExecutor` compile step.
  - `--format {txt|pdf|both}`, `--template`, and `--category` flags added to
    `tailor`, `generate-cover-letter`, `batch`, and `apply`.
  - TUI key bindings `T` (tailor résumé PDF), `C` (cover-letter PDF), and `p`
    (open the generated PDF in the default viewer).
  - PDF filenames include microseconds and the chosen template suffix to avoid
    collisions (e.g. `tailored_Acme_Dev_20260625_120000_123456_modern.pdf`).
  - `--format both` writes `.txt` + `.pdf` + a single `.meta.json` sidecar.
  - `OutputConfig` in `config.toml` controls default format and templates.
  - `job-applicator doctor` verifies the PDF rendering stack.
  - Property-based fuzz tests (`tests/unit/test_pdf_renderer_fuzz.py`) verify Typst
    escaping is idempotent and complete.
  - Opt-in visual regression tests (`tests/integration/test_pdf_regression.py`) gated
    by `JOB_APPLICATOR_PDF_REGRESSION=1`.

## [0.3.6] - 2026-06-24

### Fixed

- Cover letters no longer sign off as the placeholder `default` or invent alternate names. The applicant name now falls back to the parsed résumé name when `profile_name` is unset or left as the default placeholder.
- `_voice_tells` ignores the trailing sign-off block so a valid `Sincerely,\n<name>` closing does not suppress the short-sentence robotic-writing tell.
- Token-level signature matching rejects names that only appear as substrings of other words (e.g. `Sam` inside `Samantha`).
- The TUI style-guide modal now correctly clears the configured style guide when the user saves an empty path.
- `profile_name` handling now strips whitespace and treats `default` case-insensitively as the unset sentinel.
- The `refine()` prompt now includes the applicant profile and an explicit sign-off requirement, matching the generation path.
- The startup-warm style-guide example now uses a closing (`Best,`) that the validator recognizes.
- Security hardening: replaced `random` with `secrets.SystemRandom` for jitter/backoff and browser typing delays; resolved `pdftotext` and browser-version probes via full executable paths; marked cache-key MD5 uses as non-security.
- Removed all pre-existing `vulture` dead-code findings.

### Added

- `documents/sign_off.py`: structured sign-off extraction and validation. Generated cover letters are now hard-validated for a recognized closing word plus a signature matching the applicant's full name (or the single known part when only one is available).
- Explicit sign-off instruction in the cover-letter prompt, including an example block and a note that the sign-off must be the very last text in the letter.
- Style-guide prompts now include a reminder that the letter must still end with a sign-off and the applicant's name, preserving voice while enforcing the signature rule.
- `_load_user_profile` accepts an optional `resume_name` fallback and all cover-letter call sites (CLI `generate-cover-letter`, `batch`, `apply`; TUI actions; interactive workflow `refine`) pass the parsed résumé name.
- Empirical test-first coverage: `tests/unit/test_sign_off.py` pins extraction, validation, name-fallback behavior, style-guide example closings, and substring false-positives.
- Live end-to-end sign-off tests in `tests/test_sign_off_e2e_live.py` validate real CLI cover letters are signed with the résumé name and respect a configured `profile_name` override.
- TUI action test verifies `cover_letter_job` resolves the applicant name from the parsed résumé for correct signing.
- Added `pytest-rerunfailures` to the dev dependencies; live tests that depend on the small local vLLM model are auto-retried up to 2 times to keep CI stable without masking real failures.

## [0.3.5] - 2026-06-24

### Fixed

- `batch --no-cover-letter` is now respected even when `--style-guide` is provided; cover letters are no longer generated in that mode.
- `apply`, `batch`, and `tailor` now forward `--ocr-mode` and `--force-ocr` to the style-guide loader.
- `apply` dry-run and style-guide status messages are emitted to stderr so `--json` stdout remains clean.
- Interactive `tailor` refinements (`[R]efine`, `[I]nput`, `[S]tyle`) now preserve the loaded style guide.

### Changed

- `CoverLetterGenerator.load_style_guide` uses `ResumeLoader` for all file types and raises `DocumentError` for missing or unreadable paths.
- `CoverLetterOutput` and `StyleGuide` models now reject unknown fields (`extra="forbid"`).
- `config-init` template now includes `style_guide_path`.
- `AGENTS.md` now documents `--style-guide` support for `apply`, `tailor`, `generate-cover-letter`, and `batch`.

## [0.3.4] - 2026-06-24

### Added

- Universal multi-file style-guide support: `CoverLetterGenerator.load_style_guide` now accepts a comma-separated list of paths, loads `.pdf` files via `ResumeLoader`, and merges multiple examples with `StyleAnalyzer.analyze_multiple`.
- `apply --style-guide` now loads the style guide and passes it through to `CoverLetterGenerator.generate`.

### Changed

- `generate-cover-letter` now uses the same shared style-guide loader as `apply`, `batch`, and `tailor`, removing the duplicated inline loading logic.
- `config.example.toml` and `README.md` document comma-separated `--style-guide` usage.


## [0.3.3] - 2026-06-24

### Fixed

- Help-text unit tests now use introspection for option registration and set `COLUMNS=200` via `monkeypatch` before rendering, avoiding failures caused by `CliRunner` forcing an 80-column terminal width in CI.

## [0.3.2] - 2026-06-24

### Fixed

- Unit tests that assert on `--help` output now invoke the CLI with `COLUMNS=200` so option names are not truncated on narrow CI terminals.

## [0.3.1] - 2026-06-24

### Fixed

- Release workflow now installs the `[embeddings]` extra so `mypy src/` passes in CI.

## [0.3.0] - 2026-06-24

### Added

- `apply` now generates AI cover letters during dry runs whenever `--cover-letter` is enabled (the default) and a résumé path is configured. The generated letter is surfaced in `--json` output and in the console table as a preview before the user opts in with `--submit`.
- New live end-to-end tests (`tests/test_apply_dry_run_cover_letter_live.py`) that exercise the real `apply` CLI with vLLM cover-letter generation.

### Fixed

- Console table notes in `workflows/apply.py` are now escaped with `rich.markup.escape()` so bracketed labels like `[submit ✓]` and `[cover letter: N chars]` render literally instead of being stripped as invalid Rich markup.

## [0.2.0] - 2026-06-19

### Added

- Initial structured release baseline.

[0.3.4]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.3...v0.3.4
[0.3.6]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.4...v0.3.5
[0.4.0]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.6...v0.4.0
[0.4.1]: https://github.com/Zenetusken/job-applicator-python/compare/v0.4.0...v0.4.1
[0.5.0]: https://github.com/Zenetusken/job-applicator-python/compare/v0.4.1...v0.5.0
[Unreleased]: https://github.com/Zenetusken/job-applicator-python/compare/v0.5.0...HEAD
[0.3.3]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Zenetusken/job-applicator-python/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Zenetusken/job-applicator-python/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Zenetusken/job-applicator-python/releases/tag/v0.2.0
