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

# Tests (47 unit tests, all fast)
pytest tests/unit/ -v
pytest tests/unit/ -v -k test_name  # single test

# CLI
job-applicator --help
job-applicator match --resume resume.pdf
```

## Architecture

```
src/job_applicator/
├── cli.py              # Typer CLI (search, apply, match, generate-cover-letter)
├── config.py           # AppSettings + sub-configs (BrowserConfig, LLMConfig, EmbeddingConfig)
├── models.py           # All shared Pydantic models (JobListing, ResumeData, StyleGuide, etc.)
├── exceptions.py       # JobApplicatorError hierarchy
├── browser/            # Playwright lifecycle (manager.py) + low-level actions (actions.py)
├── scrapers/           # base.py (ABC) → linkedin.py, indeed.py (stub)
├── applicators/        # base.py (ABC) → linkedin.py, indeed.py (stub)
├── documents/          # cover_letter.py (LLM), resume.py (parser), style_analyzer.py
├── embeddings/         # service.py (mxbai-embed-large-v1), matching.py (job matching)
└── utils/              # logging.py, retry.py
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

- **LLM output has thinking process.** Qwen models prepend reasoning. `strip_thinking_process()` in `cover_letter.py` handles this.
- **Embeddings need `openai/` prefix for vLLM.** `model = f"openai/{config.model}"` when calling litellm.
- **Resume PDF parser is fragile.** Skills extraction breaks on bullet-per-line PDFs. The parser handles this, but verify with `ResumeLoader.load()`.
- **`sentence-transformers` needs CUDA torch.** If you get `libcudart.so` errors, reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **Embedding cache at `~/.job-applicator/embeddings/`.** Style cache at `~/.job-applicator/styles/`. Clear with `EmbeddingService.clear_cache()`.
- **Skill matching threshold is 0.55.** Lower = more matches, higher = stricter. Tune in `matching.py:_match_skills()`.

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
- `scripts/test_e2e.py` — full pipeline (needs vLLM running)

## Files Not to Commit

- `config.toml` (contains credentials)
- `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`
- `output/`, `screenshots/`
