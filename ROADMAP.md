# Roadmap

Living tracker of planned work. Detailed specs/plans/reports live under `docs/compose/`
(spec → plan → report). Substantial changes follow a gated cadence: empirics → spec →
implement → gate (`ruff` · `ruff format --check` · `mypy src/` · `pytest -m unit` ·
`pytest -m live`) → code-review → commit → PR.

## Shipped

- **Hardening-arc audit follow-up** (PR #24): clusters 1–3 audit fixes + the six deferred
  design enhancements — half-open circuit breaker, `LLMRuntime` DI, `LLMResilienceConfig`
  policy, `ValidatedOutput` error-feedback, `StyleAnalyzer.format` staticmethod,
  `resume_tailor` folded onto the shared breaker, `BatchRunSpec`, batch mid-job resume.
  Gate at merge: unit 576 · live 21.

## In progress / next (gated cycles — PR #25 + follow-ups)

- **A — doctor hygiene** *(done)*: `check_config` no longer creates the output dir
  (side-effect removed). resume.py L7/L8/L9 nits were already resolved in the cluster rework.
- **B — tier-2 test refresh** *(done)*: un-skipped E2 batch / D2 OCR / D4 ATS in
  `test_tier2_live` (all three are implemented; D4 now runs `ATSChecker` functionally).
- **C — integration smoke tests** *(done)*: first `tests/integration/` tests — board
  `browser_policy()` → `_make_browser` wiring, construction-only (no real launch).
- **D — Indeed search-only** *(done)*: see Out of scope.
- **E — CLI decomposition (incremental)** *(next)*: extract ONE low-risk layer from the
  ~3,100-line `cli.py` per gated cycle (e.g. browser/scraper factories, then cookie
  handling), validating green between each. Deliberately incremental to bound regression risk.

## Out of scope / accepted

- **Indeed automated apply**: scoped to search/match-only — automated apply is intentionally
  unsupported (Cloudflare anti-bot + ToS); the applicator returns a clean SKIPPED result
  directing the user to apply manually. LinkedIn Easy Apply is the only automated apply path.
- **Non-LinkedIn automated login**: not planned — collides with the standing "never
  automate login; reuse a human-seeded session" account-safety policy (see `CLAUDE.md`).
- **Browser selectors**: hardcoded against live DOM snapshots; inherent ongoing
  maintenance, not a discrete deliverable.

## Operational

- Re-validate the live E2E suite (Tier-1/Tier-2, batch, tailor) whenever vLLM or a
  selector changes; live LinkedIn/Easy-Apply flows are inherently fragile.
