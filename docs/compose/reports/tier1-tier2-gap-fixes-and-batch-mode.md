---
feature: tier1-tier2-gap-fixes-and-batch-mode
status: delivered
specs:
  - docs/compose/specs/2026-06-13-enhanced-tailor-workflow-design.md
  - docs/compose/specs/2026-06-13-cover-letter-integration-design.md
plans:
  - docs/compose/plans/2026-06-13-enhanced-tailor-workflow.md
  - docs/compose/plans/2026-06-13-cover-letter-integration.md
  - docs/compose/plans/2026-06-13-resume-tailor.md
  - docs/compose/plans/2026-06-13-batch-mode.md
branch: main
commits: 3b693ec..7891f8f
---

# Tier 1+2 Gap Fixes + Batch Mode — Final Report

## What Was Built

A comprehensive set of improvements to the job-applicator-python tool, delivered across three pull requests. The changes span the entire stack: embedding pipeline, LLM prompt engineering, CLI UX, document processing, and a new batch processing command.

**PR #1 — Enhanced Tailor Workflow**: Interactive resume tailoring with diff view, version history, section editing, auto tone detection, and production hardening (max 10 retries).

**PR #2 — Cover Letter Integration**: Post-tailor cover letter generation with shared tone/style, accept/retry/input/diff/history workflow, and linked metadata between resume and cover letter.

**PR #3 — Tier 1+2 Gap Fixes + Batch Mode**: 15 targeted fixes from a comprehensive codebase audit (instructor integration, embedding cache key, mxbai query prefix, DOCX support, pdftotext -layout flag, score population, --json flag, seniority detection, dependency cleanup, few-shot examples, temperature tuning, parallel cover letters, --min-score gate, async file I/O) plus a new `batch` CLI command for non-interactive multi-job processing.

## Architecture

### Batch Pipeline (`cli.py:batch`)

```
Input (JSON file or live search)
  → JobMatcher.rank_jobs() (single resume embedding, batch job embeddings)
  → Filter by --min-score, take --top-k
  → Parallel per-job (Semaphore(3)):
      ResumeTailor.tailor() → TailoredResume
      CoverLetterGenerator.generate() → cover letter (optional)
  → Save per-job files + batch_summary.json
```

Key components reused: `JobMatcher` (embedding + scoring), `ResumeTailor` (LLM tailoring + hallucination guards), `CoverLetterGenerator` (LLM cover letters + instructor structured output). No new modules — the batch command orchestrates existing components.

### Score Decomposition (`matching.py`)

`MatchResult` now exposes raw `semantic_score` and `skill_score` alongside the combined `score` (60% semantic + 40% skill). `resume_tailor.py` stores these directly — no recomputation.

### Instructor Integration (`style_analyzer.py`)

`StyleAnalyzer` uses `instructor.from_litellm(acompletion)` with `response_model=StyleGuide` for structured output. Falls back to manual 5-strategy JSON parser on failure. Temperature 0.1 for deterministic extraction.

### Per-Task Temperature Tuning

| Task | Temperature | Rationale |
|------|------------|-----------|
| Style analysis | 0.1 | Structured JSON extraction — deterministic |
| Change summaries | 0.2 | Factual descriptions — low creativity |
| Resume refinement | 0.3 | Conservative edits — preserve user intent |
| Initial tailoring | 0.4 | Creative rewriting — moderate variation |
| Cover letters | 0.7 | Narrative prose — higher variation |

## Usage

```bash
# Match resume to jobs
job-applicator match --resume resume.pdf --jobs-file jobs.json --top-k 10 --json

# Interactive tailor
job-applicator tailor --resume resume.pdf --job-title "Python Dev" --company "Acme" \
  --requirements "Python,FastAPI" --min-score 0.5

# Batch processing (new)
job-applicator batch --resume resume.pdf --jobs-file jobs.json --top-k 10 --min-score 0.5
job-applicator batch --resume resume.pdf --query "python developer" --top-k 5 --no-cover-letter
```

## Verification

- **301 unit tests** passing (280 unit + 21 live integration tests)
- **3 review cycles** on PR #3 — 16 issues found and resolved (3 critical, 5 medium, 8 low)
- **Live tests**: Tier 1 (54 tests), Tier 2 (33 tests), batch pipeline (2 parallel jobs in 8.2s)
- **Critical bugs caught**: Score decomposition (combined × 0.6 instead of raw), `_refine_cover_letter` returning None, `rank_jobs()` missing new MatchResult fields

## Journey Log

- [lesson] `MatchResult` fields must be propagated at ALL construction sites — `rank_jobs()` was missed because it constructs independently from `match_resume_to_job()`
- [lesson] Functions typed `-> None` that are checked for return values silently produce wrong control flow — always verify return type matches caller expectations
- [pivot] `parse_sections()` evolved from regex to `KNOWN_HEADERS` frozenset — regex had too many false positives on ALL CAPS names
- [lesson] `NamedTemporaryFile(delete=False)` needs `try/finally` — success-only cleanup leaks files on error paths
- [lesson] Few-shot examples in system prompts have enormous ROI for 4B models — concrete before/after examples dramatically improve output quality

## Source Materials

| File | Role | Notes |
|------|------|-------|
| `docs/compose/specs/2026-06-13-enhanced-tailor-workflow-design.md` | PR #1 design | ToneDetector, parse_sections, TailorSession |
| `docs/compose/specs/2026-06-13-cover-letter-integration-design.md` | PR #2 design | CoverLetterResult/Session, post-tailor workflow |
| `docs/compose/plans/2026-06-13-enhanced-tailor-workflow.md` | PR #1 plan | 8 tasks, all complete |
| `docs/compose/plans/2026-06-13-cover-letter-integration.md` | PR #2 plan | 5 tasks, all complete |
| `docs/compose/plans/2026-06-13-batch-mode.md` | PR #3 plan | 4 tasks, all complete |
| `docs/compose/plans/2026-06-13-resume-tailor.md` | Initial tailor plan | Superseded by enhanced workflow |
