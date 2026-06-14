# Job Applicator

AI-powered job application tool using Playwright browser automation with modern LLM stack and semantic embeddings.

## Features

- **Job Search**: Scrape job listings from LinkedIn and Indeed
- **Auto-Apply**: Automatically fill and submit job applications
- **AI Cover Letters**: Generate personalized cover letters using LLM (litellm - supports 100+ providers)
- **Resume Parsing**: Load and parse PDF/text/image resumes with intelligent skill extraction; OCR fallback for scanned PDFs
- **Semantic Job Matching**: Match resumes to jobs using mxbai-embed-large-v1 embeddings
- **Resume Tailoring**: LLM-powered resume rewriting for specific jobs with hallucination guards
- **Date Audit**: Pre-ingestion CV validation — checks ordering, staleness, timeline coherence
- **Style Analysis**: Mimic writing style from example resumes/cover letters
- **Structured Outputs**: Instructor for type-safe LLM responses

## Tech Stack

- Python 3.12+
- Playwright (browser automation)
- litellm (universal LLM API)
- instructor (structured outputs)
- sentence-transformers (mxbai-embed-large-v1 embeddings)
- Pydantic v2 (data validation)

## Hardware Requirements

| Component | Allocation |
|---|---|
| GPU | NVIDIA RTX 4070 (12 GB VRAM) |
| vLLM (Qwen3.5-4B) | ~7.2 GB |
| Embeddings (mxbai-embed-large-v1) | ~1.5 GB |
| Free VRAM | ~3.3 GB |

## Installation

```bash
# Requires Python 3.12+
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Initialize config
job-applicator config-init

# Search for jobs
job-applicator search --site linkedin --query "python developer"

# Auto-apply with AI cover letters
job-applicator apply --site linkedin --query "python" --limit 5

# Generate a cover letter
job-applicator generate-cover-letter --job-title "Python Dev" --company "Acme"

# Match resume to jobs using embeddings
job-applicator match --resume resume.pdf --jobs-file jobs.json --top-k 10

# Force OCR for scanned PDFs or image resumes
job-applicator match --resume scanned.pdf --force-ocr
job-applicator match --resume resume.png --ocr-mode on

# Batch tailor resumes for multiple jobs (non-interactive)
job-applicator batch --resume resume.pdf --jobs-file jobs.json --top-k 10 --min-score 0.5
job-applicator batch --resume resume.pdf --query "python developer" --top-k 5 --no-cover-letter

# Tailor resume for a specific job (interactive session)
job-applicator tailor --resume resume.pdf --job-title "Tech Support" --company "CGI" \
  --requirements "Troubleshooting,Windows,Office 365" --location "Montreal, QC"

# Generate cover letter with style guide
job-applicator generate-cover-letter --resume resume.pdf --style-guide example.txt

# Detailed match report with per-skill breakdown
python scripts/detailed_match_report.py
```

### Enhanced Tailor Workflow

The `tailor` command runs an interactive session that lets you iteratively refine your resume:

- **Diff View**: After each attempt, a unified diff is shown so you can see exactly what changed. Press `[D]` at the prompt to see the full diff of any attempt.
- **Version History**: Press `[V]` to browse all previous attempts and select one to revert to or compare against.
- **Section Editing**: Press `[S]` to target a specific resume section (e.g. Experience, Skills, Summary) for focused rewriting instead of regenerating the entire resume.
- **Auto Tone Detection**: The tailor automatically detects the job posting's tone (corporate, startup, technical, or creative) and adjusts vocabulary and phrasing accordingly.
- **Error Handling**: Up to 10 retry attempts on LLM failures, with a warning at attempt 8. The session gracefully recovers from transient LLM errors.
- **Post-Tailor Cover Letter**: After accepting a tailored resume, the CLI offers to generate a matching cover letter. The same tone, style guide, and job data are shared between both documents. The cover letter follows the same accept/retry/input/diff/history workflow as the resume, and is saved alongside it with linked metadata.

### Batch Mode

The `batch` command runs the full match→tailor→cover-letter pipeline non-interactively across multiple jobs:

```bash
# From a JSON file
job-applicator batch --resume resume.pdf --jobs-file jobs.json --top-k 10 --min-score 0.5

# From a live search
job-applicator batch --resume resume.pdf --query "python developer" --top-k 5

# Without cover letters
job-applicator batch --resume resume.pdf --jobs-file jobs.json --no-cover-letter
```

- **Smart matching**: Jobs are ranked by semantic similarity + skill coverage, filtered by `--min-score`, then only the top `--top-k` are processed.
- **Parallel execution**: Tailoring and cover letter generation run concurrently (up to 3 simultaneous LLM calls).
- **Per-job output**: Each job produces `tailored_*.txt` + `.meta.json` and optionally `cover_letter_*.txt` + `.meta.json`.
- **Batch summary**: A `batch_summary_{timestamp}.json` file contains all results with scores, paths, and errors.

## Configuration

Copy `config.example.toml` to `config.toml` and fill in your details, or use environment variables with `JOB_APPLICATOR_*` prefix.

### LLM Configuration

```toml
[llm]
api_base = "http://localhost:8000/v1"  # vLLM endpoint
api_key = "not-needed-for-local"
model = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
```

### Embedding Configuration

```toml
[embedding]
model_name = "mixedbread-ai/mxbai-embed-large-v1"
device = "cuda"
memory_limit_gb = 1.5
```

## Development

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/job_applicator/

# Test
pytest -m unit
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory (12 GB)                       │
├─────────────────────────────────────────────────────────────┤
│  vLLM Orchestrator (Qwen3.5-4B-AWQ)     ~7.2 GB            │
│  ├── Cover letter generation                               │
│  ├── Style analysis                                        │
│  └── Job description understanding                         │
├─────────────────────────────────────────────────────────────┤
│  Embedding Model (mxbai-embed-large-v1)  ~1.5 GB           │
│  ├── Resume embedding                                      │
│  ├── Job matching                                          │
│  └── Skill similarity                                      │
├─────────────────────────────────────────────────────────────┤
│  Free VRAM                           ~3.3 GB               │
└─────────────────────────────────────────────────────────────┘
```
