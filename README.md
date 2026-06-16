# Job Applicator

AI-powered job application tool using Playwright browser automation with modern LLM stack and semantic embeddings.

## Features

- **Job Search**: Scrape job listings from LinkedIn (session-authenticated) and Indeed (public, Cloudflare-fronted)
- **Session Reuse**: Sign in once in your real browser; the tool reuses the session — it never automates login (which would trip anti-bot defenses and risk your account)
- **Region-Aware Browser**: Auto-detects the host's locale, IANA timezone, and Chrome version so geo-aware boards serve your real region
- **Auto-Apply**: Automatically fill and submit job applications (dry-run by default; `--submit` to send)
- **AI Cover Letters**: Generate personalized cover letters using LLM (litellm - supports 100+ providers)
- **Resume Parsing**: Load and parse PDF/text/image resumes with intelligent skill extraction; OCR fallback for scanned PDFs
- **Semantic Job Matching**: Match resumes to jobs using mxbai-embed-large-v1 embeddings
- **Resume Tailoring**: LLM-powered resume rewriting for specific jobs with hallucination guards
- **Date Audit**: Pre-ingestion CV validation — checks ordering, staleness, timeline coherence
- **Style Analysis**: Mimic writing style from example resumes/cover letters
- **ATS Compatibility Check**: Validate resumes against ATS heuristics (contact info, standard sections, length, no ASCII tables) with a score and actionable suggestions
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
job-applicator search --site indeed --query "python developer" --location "Montreal, QC"

# Auto-apply with AI cover letters (dry run — fills forms but does NOT submit)
job-applicator apply --site linkedin --query "python" --limit 5
# Add --submit to actually send applications
job-applicator apply --site linkedin --query "python" --limit 5 --submit

# Generate a cover letter
job-applicator generate-cover-letter --job-title "Python Dev" --company "Acme"

# Check resume ATS compatibility (score >= 60% = compatible)
job-applicator ats-check --resume resume.pdf

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

### Authentication & Sessions

The tool **never automates login** — programmatic sign-in is exactly what trips a
job board's risk-based CAPTCHA and raises your account's risk score. Instead you
establish a session once as a human and the tool reuses it:

```bash
# Option A — sign in once in a real (headed) browser window; the session is
# saved to the persistent Chrome profile and reused headlessly afterwards.
job-applicator login

# Option B — reuse the session already in your everyday browser. Reads (decrypts)
# that browser's cookie store; only runs when you pass --from-browser.
job-applicator import-cookies --site linkedin --from-browser chrome
job-applicator import-cookies --site indeed   --from-browser chrome

# Other import sources:
job-applicator import-cookies --li-at "<value>"          # paste the LinkedIn li_at cookie
job-applicator import-cookies --li-at -                   # ...or read it from stdin (no shell history)
job-applicator import-cookies --site indeed --file cookies.json   # a cookie-manager JSON export
```

- **LinkedIn** needs the `li_at` session cookie (required — nothing authenticates without it),
  and runs fully headless on the shared persistent profile.
- **Indeed** search is public; cookie import is optional. Indeed sits behind a Cloudflare
  *managed challenge* that blocks headless Chrome, so the Indeed scraper runs **headed** on a
  fresh profile — kept windowless via a virtual display (Xvfb). Install the optional `[indeed]`
  extra for that to be automatic on any host: `pip install -e ".[indeed]"` (needs the system
  `Xvfb` binary, e.g. `apt install xvfb`). Without it, Indeed uses your ambient display, or run
  the command under `xvfb-run`. `--headed` shows a real window instead.
- **Region** is auto-detected: the browser advertises the host locale + timezone + Chrome UA,
  and the Indeed host is derived from your timezone (e.g. `ca.indeed.com` in Canada). Pin one
  explicitly with `target.indeed_domain` (e.g. `ca.indeed.com`) if needed.

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

### Browser & Region Configuration

```toml
[browser]
headless = true
# Empty = auto-detect from the host. Set to pin a region for geo-aware boards.
locale = ""          # e.g. "en-CA"
timezone = ""        # e.g. "America/Toronto" (IANA name)
# user_agent = ""    # empty = match the host's installed Chrome major version

[target]
# Indeed redirects by region; the scraper auto-detects, or pin one here.
indeed_domain = "www.indeed.com"   # e.g. "ca.indeed.com", "uk.indeed.com"
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
