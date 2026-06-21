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

## Near-term (gated cycles)

- **A — code hygiene** *(in progress)*: `doctor`'s `check_config` no longer creates the
  output dir (side-effect removed). (resume.py L7/L8/L9 nits were already resolved in the
  cluster rework.)
- **B — test hygiene**: replace the stale `tests/test_tier2_live.py` `NOT IMPLEMENTED`
  skips (E2 batch, D2 OCR, D4 ATS — all three are implemented) with real assertions; add
  first `tests/integration/` smoke tests (BrowserManager, session health, scraper launch).

## Decisions pending (need a scope call before implementing)

- **Indeed apply** (`applicators/indeed.py` returns FAILED): implement the "Easily apply"
  form flow, or officially scope Indeed to search-only.
- **CLI decomposition**: extract orchestration/service layers from the ~3,100-line
  `cli.py` (highest architectural risk; needs an agreed approach + incremental plan).

## Out of scope / accepted

- **Non-LinkedIn automated login**: not planned — collides with the standing "never
  automate login; reuse a human-seeded session" account-safety policy (see `CLAUDE.md`).
- **Browser selectors**: hardcoded against live DOM snapshots; inherent ongoing
  maintenance, not a discrete deliverable.

## Operational

- Re-validate the live E2E suite (Tier-1/Tier-2, batch, tailor) whenever vLLM or a
  selector changes; live LinkedIn/Easy-Apply flows are inherently fragile.
