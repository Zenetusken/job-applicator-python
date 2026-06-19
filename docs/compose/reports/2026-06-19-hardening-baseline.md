---
date: 2026-06-19
commit: 95bae1f
branch: main
---

# Hardening Baseline Report

## Goal

Establish a reproducible baseline before a systematic hardening pass across the
`job-applicator-python` stack.

## Environment

- **Host:** Linux, Python 3.12.13
- **Package version:** `0.1.0` (runtime) / `0.2.0` (`pyproject.toml`)
- **vLLM endpoint:** `http://localhost:8000/v1` — reachable
- **Model served:** `cyankiwi/Qwen3.5-4B-AWQ-4bit`
- **Working tree:** clean, `main` synced with `origin/main`

## Quality Gates

| Gate | Command | Result |
|---|---|---|
| Lint | `ruff check src/ tests/` | clean |
| Format | `ruff format --check src/ tests/` | clean |
| Type check | `mypy src/` | clean |
| Unit tests | `pytest -m unit -q` | **488 passed, 21 deselected** |
| Live tests | `pytest -m live -q` | **21 passed, 488 deselected** (30.50s) |

## Live Test Coverage

- `tests/test_batch_live.py` — batch pipeline
- `tests/test_live_tailor.py` — interactive tailor workflow
- `tests/test_tier1_live.py` — style analyzer, cache, query prefix, DOCX, OCR, JSON flag, seniority, dependencies
- `tests/test_tier2_live.py` — few-shot examples, temperature tuning, parallel cover letters, async I/O, ATS

## Known Pre-Existing Gaps (from prior audits)

1. Indeed `apply` is wired but not validated end-to-end.
2. LinkedIn live scraping/Easy Apply remains fragile (CAPTCHA/challenge).
3. `doctor` only checks LLM/embeddings/self-host — no browser/config/session checks.
4. Resume parsing has no confidence score or multi-parser consensus.
5. LLM pipeline lacks structured-output validation retry and circuit breaker.
6. No persistent application-state DB or duplicate-apply prevention.

## Baseline Verdict

The core AI pipeline is healthy against the live test suite. All quality gates
pass. This is a solid foundation for the hardening pass.
