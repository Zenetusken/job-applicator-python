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
mypy src/job_applicator/ --ignore-missing-imports

# Auto-fix lint/format
ruff check --fix src/ tests/
ruff format src/ tests/

# Tests (236 unit tests, all fast)
pytest tests/unit/ -v
pytest tests/unit/ -v -k test_name  # single test

# CLI
job-applicator --help
job-applicator match --resume resume.pdf
```

## Architecture

```
src/job_applicator/
├── cli.py              # Typer CLI (search, apply, match, generate-cover-letter, tailor, config-init)
├── config.py           # AppSettings + sub-configs (BrowserConfig, LLMConfig, EmbeddingConfig, TargetConfig)
├── models.py           # All shared Pydantic models (JobListing, ResumeData, StyleGuide, TailoredResume, DateAuditResult, etc.)
├── exceptions.py       # JobApplicatorError hierarchy
├── browser/            # Playwright lifecycle (manager.py) + low-level actions (actions.py)
├── scrapers/           # base.py (ABC) → linkedin.py, indeed.py (stub)
├── applicators/        # base.py (ABC) → linkedin.py, indeed.py (stub)
├── documents/          # cover_letter.py (LLM), resume.py (parser), resume_tailor.py (tailoring), style_analyzer.py, tone_detector.py
├── embeddings/         # service.py (mxbai-embed-large-v1), matching.py (job matching)
└── utils/              # logging.py, retry.py, diff.py
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

- **LLM output has thinking process.** Qwen models prepend reasoning. Use `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` in litellm calls to suppress. Fallback: `strip_thinking_process()` in `cover_letter.py`.
- **Resume tailoring hallucination guards.** `_validate_skills()` strips hallucinated skills. `_strip_hallucinated_tools()` replaces tools not in original resume. `_strip_hallucinated_education()` removes education if original has none.
- **Education extraction must be explicit.** LLMs silently drop education entries. `_extract_education_entries()` injects a numbered checklist into the prompt to force inclusion.
- **Embeddings need `openai/` prefix for vLLM.** `model = f"openai/{config.model}"` when calling litellm.
- **Resume PDF parser is fragile.** Skills extraction breaks on bullet-per-line PDFs. The parser handles this, but verify with `ResumeLoader.load()`.
- **`sentence-transformers` needs CUDA torch.** If you get `libcudart.so` errors, reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **Embedding cache at `~/.job-applicator/embeddings/`.** Style cache at `~/.job-applicator/styles/`. Clear with `EmbeddingService.clear_cache()`.
- **Skill matching threshold is 0.55.** Lower = more matches, higher = stricter. Tune in `matching.py:_match_skills()`.
- **`parse_sections()` uses known headers.** Matches against `KNOWN_HEADERS` frozenset (case-insensitive) and Title Case with colon suffix. ALL CAPS names (e.g. "JOHN DOE") are NOT matched as headers. Add new headers to the frozenset in `resume_tailor.py` if needed.
- **Skill validation uses fuzzy matching.** `_skills_match()` in `resume_tailor.py` checks exact match, token containment (subset), and `SequenceMatcher` ratio >= 0.85. Prevents "ai" matching "training" while catching typos.
- **Tool hallucination has two passes.** Pass 1: checks job requirements not in original. Pass 2: checks `tool_replacements` keys in tailored text not in original AND not in requirements. Catches LLM-invented tools.
- **`tailor()` accepts optional `tone_profile`.** When provided, skips internal `ToneDetector.detect()`. Eliminates double detection when CLI already computed the profile.
- **`refine()` recomputes match scores.** Creates synthetic `ResumeData` from refined text and runs `JobMatcher.match_resume_to_job()`. No more stale scores.
- **`CoverLetterGenerator.refine()` exists.** Uses same structured generation pipeline as `generate()` — system prompt, style guide, tone section, instructor fallback. `_refine_cover_letter()` in cli.py delegates to it.
- **Tone detection is keyword-based, not LLM-based.** `ToneDetector.detect()` in `tone_detector.py` uses keyword frequency heuristics — fast, but may misclassify edge cases (e.g. a startup posting heavy on compliance jargon).
- **Max tailor retry limit is 10.** A warning prints at attempt 8. The limit is hardcoded in `cli.py` and `tailor_cgi.py` — search for `attempt > 10` to adjust.
- **`TailorSession` is in-memory only.** Version history is lost when the session ends. No persistence to disk.
- **Cover letter sub-loop has no `[S] Section` option.** Cover letters lack parseable sections, so the section-editing prompt is skipped in the cover letter flow.
- **`CoverLetterResult` is simpler than `TailoredResume`.** No `match_score`, `matched_skills`, or `semantic_score` — cover letters don't go through embedding-based matching.
- **Resume meta.json write is deferred until after cover letter flow.** The CLI waits until the cover letter sub-loop completes (or is skipped) before writing the resume's sidecar metadata, so `cover_letter_path` can be included.
- **`cover_letter_path` in `TailoredResume` links resume to cover letter.** After the cover letter is saved, its path is stored in the resume model for downstream consumers.

## LLM Setup

Local vLLM must be running at `http://localhost:8000/v1`. Check with:
```bash
curl -s http://localhost:8000/v1/models
```

Default model: `cyankiwi/Qwen3.5-4B-AWQ-4bit`. Override via `JOB_APPLICATOR_LLM_MODEL` env var or `config.toml`.

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
- `output/`, `screenshots/`
