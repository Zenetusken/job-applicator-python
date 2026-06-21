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

## Quality Gates — Baseline

| Gate | Command | Result |
|---|---|---|
| Lint | `ruff check src/ tests/` | clean |
| Format | `ruff format --check src/ tests/` | clean |
| Type check | `mypy src/` | clean |
| Unit tests | `pytest -m unit -q` | **488 passed, 21 deselected** |
| Live tests | `pytest -m live -q` | **21 passed, 488 deselected** (30.50s) |

## Quality Gates — After Hardening Pass

| Gate | Command | Result |
|---|---|---|
| Lint | `ruff check src/ tests/` | clean |
| Format | `ruff format --check src/ tests/` | clean |
| Type check | `mypy src/` | clean |
| Unit tests | `pytest -m unit -q` | **518 passed, 21 deselected** |
| Live tests | `pytest -m live -q` | **21 passed, 518 deselected** (31.73s) |

## Live Test Coverage

- `tests/test_batch_live.py` — batch pipeline
- `tests/test_live_tailor.py` — interactive tailor workflow
- `tests/test_tier1_live.py` — style analyzer, cache, query prefix, DOCX, OCR, JSON flag, seniority, dependencies
- `tests/test_tier2_live.py` — few-shot examples, temperature tuning, parallel cover letters, async I/O, ATS

## Live UI Smoke Tests Performed

- `job-applicator doctor` — reports LLM, embeddings, browser, system binaries, config, and plaintext-credential warning
- `job-applicator check-session linkedin` — confirms active authenticated LinkedIn session
- `job-applicator ats-check --resume /tmp/test_resume.txt` — parses with confidence 0.94, score 100%
- `job-applicator match --resume /tmp/test_resume.txt --json` — end-to-end match with embeddings returns ranked results

## What Was Hardened

1. **Doctor** — Playwright/Chromium, `pdftotext`/`Xvfb`, config parseability, plaintext credential warning, resume/output paths.
2. **Session health** — `check-session` command, `BaseScraper.check_session()`, LinkedIn feed verification, card-level failure tracking.
3. **Resume parsing** — `parse_confidence`/`parse_method`, multi-parser PDF consensus, password-protected PDF detection.
4. **LLM pipeline** — `CircuitBreaker` and `ValidatedOutput` helpers, cover-letter output validation (empty/placeholders) with retry.
5. **Application state** — SQLite store at `~/.job-applicator/applications.db`, duplicate detection, daily-cap enforcement in `apply`.

## Remaining Known Gaps

- Indeed `apply` is wired but not validated end-to-end.
- LinkedIn live scraping/Easy Apply remains inherently fragile to CAPTCHA/challenge.

## Verdict

All hardening items implemented, tested, and validated. Unit test count grew
from 488 → 518. Live suite remains green. Live UI smoke tests confirm the
end-to-end pipeline works.
