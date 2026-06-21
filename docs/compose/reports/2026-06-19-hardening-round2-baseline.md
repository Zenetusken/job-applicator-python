---
date: 2026-06-19
commit: 02d3302
branch: main
---

# Hardening Round 2 Baseline Report

## Goal

Establish a reproducible baseline before the next systematic hardening pass.
This round focuses on:

1. vLLM availability skip guard for live tests.
2. Batch crash recovery + progress persistence.
3. Skill normalization + hard-negative skill list.
4. LinkedIn Easy Apply real dry-run validation.

## Environment

- **Host:** Linux, Python 3.12.13
- **Package version:** `0.2.0`
- **vLLM endpoint:** `http://localhost:8000/v1` — **not running**
- **Working tree:** clean, `main` synced with `origin/main`

## Quality Gates

| Gate | Command | Result |
|---|---|---|
| Lint | `ruff check src/ tests/` | clean |
| Format | `ruff format --check src/ tests/` | clean |
| Type check | `mypy src/` | clean |
| Unit tests | `pytest -m unit -q` | **518 passed, 21 deselected** |
| Live tests | `pytest -m live -q` | **2 failed, 19 passed, 518 deselected** — fails because vLLM is unreachable |

## Live Test Failure Detail

When the external vLLM endpoint is not running, live tests that exercise the
LLM path fail with a connection error instead of skipping:

- `tests/test_live_tailor.py::test_live_tailor`
- `tests/test_tier2_live.py::test_parallel_cover_letters` (and other LLM tests)

Root cause: live tests have no `pytest.skipif` guard for missing vLLM.

## Baseline Verdict

Unit suite is green. Live suite is unstable when the external LLM endpoint is
offline. The first hardening item is to add an availability guard so live
tests skip cleanly when vLLM is not available.
