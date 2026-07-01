# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Fixed
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
