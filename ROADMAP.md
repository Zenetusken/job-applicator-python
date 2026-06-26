# Roadmap

Living tracker of planned work. Detailed specs/plans/reports live under `docs/compose/`
(spec → plan → report). Substantial changes follow a gated cadence: empirics → spec →
implement → gate (`ruff` · `ruff format --check` · `mypy src/` · `pytest -m unit` ·
`pytest -m live`) → code-review → commit → PR.

## Shipped

- **Style-guide + sign-off release** (PR #35 / v0.3.6): structured cover-letter sign-off
  extraction/validation (`documents/sign_off.py`), applicant-name fallback from the parsed
  résumé, style-guide modal in the TUI (`g` key), and example guides in
  `docs/style-guide-examples/`. Includes adversarial review by architecture, lead-dev, QA,
  product/UX, and HR-domain subagents; all findings fixed. Gate: unit 876 · integration 5 ·
  live 34.
- **Hardening-arc audit follow-up** (PR #24): clusters 1–3 audit fixes + the six deferred
  design enhancements — half-open circuit breaker, `LLMRuntime` DI, `LLMResilienceConfig`
  policy, `ValidatedOutput` error-feedback, `StyleAnalyzer.format` staticmethod,
  `resume_tailor` folded onto the shared breaker, `BatchRunSpec`, batch mid-job resume.
  Gate at merge: unit 576 · live 21.
- **CLI decomposition** (PRs #26–#28, #30, #31, #33): extracted six layers from the
  ~3,200-line `cli.py` — board/browser/runtime factories + shared console (`factories.py`,
  `utils/console.py`), all cookie logic (`utils/cookies.py`), and the three orchestration
  loops (`workflows/{cover_letter,tailor,apply}.py`). `cli.py` ~3,200 → 2,303 (−~900).
  Each loop extracted tests-first: characterization tests pin behavior → guarded/verbatim
  move → zero drift.
- **Cover-letter + apply `NameError` fixes** (PRs #29, #32): two latent production crashes
  — `CoverLetterResult` / `ApplicationResult` constructed at runtime but imported only
  under `TYPE_CHECKING` — surfaced by the tests-first decomposition work and fixed via lazy
  imports.

## In progress / next (gated cycles — PR #25 + follow-ups)

- **A — doctor hygiene** *(done)*: `check_config` no longer creates the output dir
  (side-effect removed). resume.py L7/L8/L9 nits were already resolved in the cluster rework.
- **B — tier-2 test refresh** *(done)*: un-skipped E2 batch / D2 OCR / D4 ATS in
  `test_tier2_live` (all three are implemented; D4 now runs `ATSChecker` functionally).
- **C — integration smoke tests** *(done)*: first `tests/integration/` tests — board
  `browser_policy()` → `_make_browser` wiring, construction-only (no real launch).
- **D — Indeed search-only** *(done)*: see Out of scope.
- **E — CLI decomposition (incremental)** *(done — see Shipped)*: six gated increments
  extracted factories/console, cookie logic, and the cover-letter/tailor/apply loops, each
  validated green between. Only `batch`'s `_process_one` remains (banked — see Out of scope).

## Out of scope / accepted

- **Indeed automated apply**: scoped to search/match-only — automated apply is intentionally
  unsupported (Cloudflare anti-bot + ToS); the applicator returns a clean SKIPPED result
  directing the user to apply manually. LinkedIn Easy Apply is the only automated apply path.
- **Non-LinkedIn automated login**: not planned — collides with the standing "never
  automate login; reuse a human-seeded session" account-safety policy (see `CLAUDE.md`).
- **Browser selectors**: hardcoded against live DOM snapshots; inherent ongoing
  maintenance, not a discrete deliverable.
- **Batch `_process_one` extraction**: banked — the last orchestration loop, but not a
  clean lift like cover-letter/tailor/apply. It's a ~160-line closure in `batch`'s `_run`
  capturing ~18 locals incl. three shared mutable lists (`tailoring_scores`,
  `batch_reports`, `written_paths`) appended by concurrent workers under `Semaphore(3)`; a
  clean extraction needs a context-object refactor (not a verbatim move) plus careful
  handling of the concurrent-append seam — diminishing returns vs risk. Revisit only if
  `_run` needs changes the entanglement obstructs, and do it tests-first (characterization
  tests driving `batch` through `_process_one`) if so.

## Operational

- Re-validate the live E2E suite (Tier-1/Tier-2, batch, tailor) whenever vLLM or a
  selector changes; live LinkedIn/Easy-Apply flows are inherently fragile.
