# Job Applicator

AI-powered job application tool using Playwright browser automation with modern LLM stack and semantic embeddings.

## Features

- **Job Search**: Scrape job listings from LinkedIn and Indeed
- **Auto-Apply**: Automatically fill and submit job applications
- **AI Cover Letters**: Generate personalized cover letters using LLM (litellm - supports 100+ providers)
- **Resume Parsing**: Load and parse PDF/text resumes with intelligent skill extraction
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

# Tailor resume for a specific job (interactive: accept/retry/input)
job-applicator tailor --resume resume.pdf --job-title "Tech Support" --company "CGI" \
  --requirements "Troubleshooting,Windows,Office 365" --location "Montreal, QC"

# Generate cover letter with style guide
job-applicator generate-cover-letter --resume resume.pdf --style-guide example.txt

# Detailed match report with per-skill breakdown
python scripts/detailed_match_report.py
```

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
