# Job Applicator

AI-powered job application tool using Playwright browser automation with modern LLM stack and semantic embeddings.

## Features

- **Job Search**: Scrape job listings from LinkedIn (session-authenticated) and Indeed (public, Cloudflare-fronted)
- **Session Reuse**: Sign in once in your real browser; the tool reuses the session — it never automates login (which would trip anti-bot defenses and risk your account)
- **Region-Aware Browser**: Auto-detects the host's locale, IANA timezone, and Chrome version so geo-aware boards serve your real region
- **LinkedIn Easy Apply**: Fill LinkedIn Easy Apply forms in a dry run by default; real submission requires `--submit`. LinkedIn external-apply jobs and Indeed applications are reported for manual follow-up rather than guessed through
- **Selector Health Diagnostics**: Probe live LinkedIn/Indeed selectors on demand, or as an opt-in `search` / `apply` preflight, so board DOM drift is reported before a real run
- **Source-Backed Cover Letters**: A temperature-zero, job-only pass extracts exact responsibility spans without seeing applicant data. The local mxbai embedder then ranks immutable résumé facts with stable tie-breaking and selects exactly three. Typed local realization preserves those facts in the body; the application frame and sign-off are deterministic. Dry runs generate the letter as a preview before you opt in with `--submit`
- **Output-Language Policy**: The generated CV and cover letter resolve the *same* language — `[llm] language = "auto"` mirrors the job posting, or force `"en"` / `"fr"`. The base résumé must already use that language; cross-language résumé generation fails closed rather than saving unverified machine translation.
- **Grounding Verification (honesty layer)**: Résumé targeting changes only a three-statement summary; each statement is an exact deterministic realization of one selected source fact, while every non-summary source section is digest-protected. Cover-letter statements follow the same typed source contract. Both artifact sidecars retain citations, source/job digests, exact target criteria, ranked scores, language, and architecture version.
- **Resume Parsing**: Load and parse PDF/text/image resumes with intelligent skill extraction; OCR fallback for scanned PDFs
- **Semantic Job Matching**: Match resumes to jobs using mxbai-embed-large-v1 embeddings
- **Resume Tailoring**: source-preserving summary overlay with immutable experience, education, skills, projects, dates, metrics, and contact content
- **Date Audit**: Pre-ingestion CV validation — checks ordering, staleness, and advisory employment gap/overlap findings
- **Style Analysis**: Analyze writing style from one or more example resumes/cover letters
  (comma-separated paths); the analyzer tries instructor structured output first, falls back to
  direct litellm JSON parsing with timing logs, and fails loudly instead of fabricating a style
  guide. Source-overlay factual statements are not rewritten to mimic a style guide. Example guides
  live in `docs/style-guide-examples/`.
- **Cover-Letter Sign-Off Enforcement**: Every generated or refined cover letter is assembled with a recognized closing word and a signature matching the applicant's name
- **ATS Compatibility Check**: Validate resumes against ATS heuristics (contact info, standard sections, length, no ASCII tables) with a score and actionable suggestions
- **PDF Résumé & Cover Letters**: Deterministically parse and render generated text with Typst (optional `[pdf]` extra); PDF formatting makes no LLM call. Built-in `modern`, `classic`, and `minimal` templates
- **Structured Outputs**: Instructor for type-safe LLM responses

## Tech Stack

- Python 3.12+
- Playwright (browser automation)
- litellm (universal LLM API)
- instructor (structured outputs)
- sentence-transformers (mxbai-embed-large-v1 embeddings)
- Typst (PDF rendering, optional `[pdf]` extra)
- Pydantic v2 (data validation)

## Hardware Requirements

| Component | Allocation |
|---|---|
| GPU | NVIDIA RTX 4070 (12 GB VRAM) |
| vLLM (Qwen3-8B-AWQ, eager mode, GPU_MEM=0.65) | ~7.9 GB (6.1 GB weights + KV) |
| Embeddings (mxbai-embed-large-v1) | ~1.4 GB |
| Runtime headroom | ~1.5-1.8 GB free on the validated RTX 4070 desktop profile |

## Installation

```bash
# Requires Python 3.12+
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## LLM backend (required for model-assisted analysis)

Style analysis, skill extraction, fallback job-criteria extraction, and the standalone grounding
diagnostic call an **OpenAI-compatible chat endpoint**. Résumé and cover-letter applicant claims
are deterministic source realizations, but document generation may still need criteria extraction
when a posting has no structured requirements. job-applicator is a *client*: it does **not** start
an endpoint. Embeddings for job matching run in-process via `sentence-transformers`, so they need no
separate service — just the `[embeddings]` extra.

Pick one:

**1. Point at an existing endpoint (recommended).** Set `api_base` + `model` under
`[llm]` in `config.toml`. It uses litellm, so any provider works — a shared local
vLLM, cloud OpenAI/Anthropic, Ollama, LM Studio, etc.

```toml
[llm]
api_base = "http://localhost:8000/v1"   # the endpoint you point at
model = "Qwen/Qwen3-8B-AWQ"
```

**2. Self-host a local vLLM.** For a standalone box with no shared/remote LLM
(needs a CUDA GPU). job-applicator ships with `scripts/serve-vllm.sh` which uses the
project's own vLLM binary and CUDA 13.0 wheel, isolated from any other app:

```bash
pip install -e ".[serve]"          # installs vLLM 0.23.x (CUDA 13.0 wheel)
scripts/serve-vllm.sh               # serves :8000
```

Environment overrides: `MODEL`, `HOST`, `PORT`, `GPU_MEM` (default `0.65`),
`MAX_MODEL_LEN` (default `8192`), `ENFORCE_EAGER` (default `1`), and `VLLM_BIN`
(default: this project's `.venv/bin/vllm`). For isolation the script uses **only**
that in-project binary or an explicit `VLLM_BIN`; if neither is present it errors
rather than silently adopting a `vllm` found on `$PATH` (which could be another
project's). Set `VLLM_BIN` to share a vLLM executable from another venv without
touching that project's config.

The defaults are tuned for a 12 GB desktop GPU running local CUDA embeddings
alongside the 8B LLM. `GPU_MEM=0.65` keeps enough 8K KV cache for Qwen3-8B-AWQ
while leaving the embedder's 1.3 GB free-VRAM preflight budget available. On tighter cards
vLLM 0.23's V1 cudagraph profiler can OOM during startup; `ENFORCE_EAGER=1` avoids
that by disabling CUDA graphs. If you have ample VRAM you can raise `GPU_MEM` or run
with `ENFORCE_EAGER=0` for higher throughput.

The first launch **auto-downloads the model** from Hugging Face Hub (~6 GB for the
default; cached to `~/.cache/huggingface`) — no separate step. Needs network on first
run. Embeddings likewise fetch `mxbai-embed-large-v1` (~640 MB) on first use; after
that, cached snapshots are loaded in local-only mode so offline/sandboxed `match` or
`batch` runs do not block on Hugging Face metadata probes.

The default model is **public**. A *gated* model additionally needs a Hugging Face
token — run **`huggingface-cli login`** once (it validates the token and persists it;
vLLM and the embedder then pick it up automatically — no app config needed).

Leave it running in its own terminal (or wrap it in a process manager / systemd unit
for always-on), then run job-applicator against it as usual.

**Verify the connection:** `job-applicator doctor` probes the endpoint and reports
capability readiness for AI generation, matching, browser workflows, and PDF output —
plus exactly what to fix if something is not ready. Run it any time the AI features
misbehave.

If a localhost LLM call fails with "denied permission to open a network socket" while
`curl http://localhost:8000/v1/models` works elsewhere, the problem is the current runtime
environment, not vLLM. This can happen inside a sandbox that allows `curl` but blocks Python's
aiohttp/httpx socket path. Re-run `job-applicator doctor` or the affected CLI command from the real
runtime.

## Usage

```bash
# Initialize config
job-applicator config-init

# Check the AI backend is reachable (LLM endpoint, embeddings, self-host prereqs)
job-applicator doctor

# Search for jobs
job-applicator search --site linkedin --query "python developer"
job-applicator search --site indeed --query "python developer" --location "Montreal, QC"
# Optional live selector preflight before scraping (extra board traffic)
job-applicator search --site linkedin --query "python developer" --selector-health

# Auto-apply with AI cover letters (dry run — fills forms, previews the cover letter, but does NOT submit)
job-applicator apply --site linkedin --query "python" --limit 5
# Preview the generated cover letter as JSON
job-applicator apply --site linkedin --query "python" --limit 1 --resume resume.pdf --json
# Add --submit to actually send applications
job-applicator apply --site linkedin --query "python" --limit 5 --submit
# Skip cover-letter generation entirely
job-applicator apply --site linkedin --query "python" --limit 5 --no-cover-letter
# Optional live selector preflight before filling a stored/target job
job-applicator apply --from <id-or-url> --selector-health

# Standalone live selector diagnostics (no scraping persistence, no submission)
job-applicator selector-health --site linkedin --surface search --query "python developer"
job-applicator selector-health --site linkedin --surface apply --from <id-or-url>
job-applicator selector-health --site indeed --surface search --query "python developer"

# Generate a cover letter
job-applicator generate-cover-letter --resume resume.pdf --job-title "Python Dev" --company "Acme"

# Check resume ATS compatibility (score >= 60% = compatible)
job-applicator ats-check --resume resume.pdf

# Check generated CV/cover-letter quality
job-applicator document-quality --resume tailored.txt --cover-letter cover.txt --keyword Python
job-applicator document-quality --private-packet-set --required --min-cases 15 \
  --min-manual-reviews-per-category 5 --max-artifact-age-days 14 \
  --required-category support --required-category risk --required-category network \
  --required-language en --required-language fr --json
# Compare current baseline against measured Qwen/vLLM sampler variants
.venv/bin/python scripts/eval_llm_sampler.py --required --integrity-only --json

# Match resume to jobs using embeddings
job-applicator match --resume resume.pdf --jobs-file jobs.json --top-k 10

# Force OCR for scanned PDFs or image resumes
job-applicator match --resume scanned.pdf --force-ocr
job-applicator match --resume resume.png --ocr-mode on

# Re-score STORED funnel jobs against the current résumé — in place, WITHOUT re-scraping
# (use after your résumé changes; account-safe, never touches LinkedIn/Indeed)
job-applicator rescore

# Batch tailor resumes for multiple jobs (non-interactive)
job-applicator batch --resume resume.pdf --jobs-file jobs.json --top-k 10 --min-score 0.5
job-applicator batch --resume resume.pdf --query "python developer" --top-k 5 --no-cover-letter

# Tailor resume for a specific job (interactive session)
job-applicator tailor --resume resume.pdf --job-title "Tech Support" --company "CGI" \
  --requirements "Troubleshooting,Windows,Office 365" --location "Montreal, QC"

# Generate cover letter with style guide (single file or comma-separated)
job-applicator generate-cover-letter --resume resume.pdf --style-guide example.txt
job-applicator generate-cover-letter --resume resume.pdf \
  --style-guide "cover_letter_example.txt,resume_example.pdf"
# Private gold-standard prose/style fixture, if populated locally:
# ~/.job-applicator/document-quality-eval/gold-standards/cover-letter-v1/cover-letter-prose-only.txt

# Apply with a style guide
job-applicator apply --site linkedin --query "python" --limit 5 \
  --resume resume.pdf --style-guide example.txt

# Batch tailor + cover letters with a style guide
job-applicator batch --resume resume.pdf --jobs-file jobs.json --top-k 10 \
  --style-guide "cover_letter_example.txt,resume_example.pdf"

# Tailor with a style guide
job-applicator tailor --resume resume.pdf --job-title "Tech Support" --company "CGI" \
  --style-guide example.txt

# Render PDF artifacts (requires the [pdf] extra)
pip install -e ".[pdf]"
job-applicator tailor --resume resume.pdf --job-title "Python Dev" --company "Acme" --format pdf
job-applicator generate-cover-letter --resume resume.pdf --job-title "Python Dev" --company "Acme" --format pdf
job-applicator batch --resume resume.pdf --jobs-file jobs.json --top-k 5 --format both
job-applicator apply --query "python" --limit 3 --format pdf

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

### Selector Health

`selector-health` is a live diagnostic surface, separate from `doctor`. It opens real
LinkedIn/Indeed pages and checks the selector groups the scraper/applicator depends on, but it does
not persist scraped jobs and never submits an application. Use it when a board layout looks suspect:

```bash
job-applicator selector-health --site linkedin --surface search --query "SOC" --location "Montreal, QC"
job-applicator selector-health --site linkedin --surface apply --from <stored-id-or-url>
job-applicator selector-health --site indeed --surface search --query "python developer" --json
```

Search/apply preflights are opt-in via `--selector-health`; failed required selector groups abort
before scraping/filling unless `--ignore-selector-health` is also provided. JSON reports are written
to stdout, with logs/diagnostics on stderr. Failure artifacts are saved under
`~/.job-applicator/debug/selector-health/`.

LinkedIn apply checks distinguish in-product Easy Apply from external "Apply on company website"
postings. External apply jobs are reported as `skipped` because Easy Apply form selectors do not
apply. Easy Apply probes require the entry button plus form controls such as Next/Continue/Review or
Submit; Submit itself can be absent until the form has been advanced and filled.

Indeed selector health covers search results and the description pane only. Indeed automated apply
remains intentionally unsupported; live recon showed on-site apply buttons are best identified by
`#indeedApplyButton`, but the app still directs the user to apply manually.

### Enhanced Tailor Workflow

The `tailor` command runs an interactive session that lets you iteratively refine your resume:

- **Diff View**: After each attempt, a unified diff is shown so you can see exactly what changed. Press `[D]` at the prompt to see the full diff of any attempt.
- **Version History**: Press `[V]` to browse all previous attempts and select one to revert to or compare against.
- **Summary Focus**: Press `[S]` to reselect evidence for the generated summary. Experience, skills, education, projects, dates, metrics, and contact content remain immutable.
- **Targeting Context**: The job posting and optional user input affect fact selection only; they cannot rewrite source claims.
- **Non-Interactive Integrity Gate**: `tailor --yes`, `tailor --json`, batch tailoring, and TUI one-shot tailoring save only when the source-body digest, deterministic summary realization, and citations all validate.
- **Error Handling**: Up to 10 retry attempts on LLM failures, with a warning at attempt 8. The session gracefully recovers from transient LLM errors.
- **Post-Tailor Cover Letter**: After accepting a tailored resume, the CLI offers to generate a matching cover letter. Job and user focus affect source-fact selection; fixed local realization produces the factual body. The letter is saved alongside the résumé with linked source-overlay metadata.

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
- **Parallel execution**: Up to three document tasks run concurrently. Criteria extraction may call the LLM when a posting has no structured requirements; applicant claim realization does not.
- **Fail-closed CV saves**: Batch refuses to save a CV unless its immutable source-body digest, summary citations, deterministic realization, contact data, and ATS integrity validate.
- **Recovery-safe identity**: Resume state is keyed by the job source, résumé, score/filter inputs, style guide, output format, templates, category, and OCR mode. `--resume-run` only resumes an incomplete run with the same output-affecting parameters.
- **Partial failure semantics**: A verified CV is kept as a resumable `TAILORED` job if only the cover letter fails, so a later resume-run retries the cover letter without re-tailoring. Any requested document failure still writes the batch summary and exits non-zero.
- **Unique job URLs required**: Batch rejects duplicate matched URLs before concurrent document generation because recovery state is keyed by URL.
- **Per-job output**: Each job produces `tailored_*.txt` + `.meta.json` and optionally `cover_letter_*.txt` + `.meta.json`. With `--format both`, the text sidecar is the authoritative metadata file and includes the generated `pdf_path`.
- **Batch summary**: A `batch_summary_{timestamp}.json` file contains all results with scores, paths, and errors.

## Configuration

Copy `config.example.toml` to `config.toml` and fill in your details, or use environment variables with `JOB_APPLICATOR_*` prefix.

### Profile Configuration

```toml
profile_name = "default"   # leave as "default"/empty to derive from the parsed résumé
resume_path = "/path/to/your/resume.pdf"
# style_guide_path = "docs/style-guide-examples/01_enterprise-formal.txt"
# For a private, CV-coherent cover-letter standard:
# style_guide_path = "~/.job-applicator/document-quality-eval/gold-standards/cover-letter-v1/cover-letter-prose-only.txt"
```

### LLM Configuration

```toml
[llm]
api_base = "http://localhost:8000/v1"  # vLLM endpoint
api_key = "not-needed-for-local"
model = "Qwen/Qwen3-8B-AWQ"
max_tokens = 4096
temperature = 0.7
# Optional sampler knobs for measured Qwen/vLLM tuning; omitted by default to preserve baseline.
# top_p = 0.8
# top_k = 20
# min_p = 0.0
# presence_penalty = 1.2
enable_thinking = false
# Output language for the generated CV + cover letter: "auto" mirrors the job
# posting's language, or force "en" / "fr". The two documents always resolve the
# SAME language. The source resume must already be in that language; automatic
# cross-language resume translation is intentionally unavailable.
language = "auto"
```

The smaller, faster `cyankiwi/Qwen3.5-4B-AWQ-4bit` remains a pinnable fallback (via
`JOB_APPLICATOR_LLM_MODEL` / `[llm] model`) — the 8B grounds stack-heavy job descriptions
the 4B couldn't, while still fitting the 12 GB card alongside the embeddings.
Sampler fields are configurable for A/B testing of factual analysis and transport behavior against
private document-quality packets. Applicant claim realization is deterministic, so these settings
do not tune résumé or cover-letter prose. They are not model fine-tuning, and unset fields are
omitted from requests.

The standalone claim-enumeration verifier remains available as a diagnostic. Deterministic
source-overlay generation and certification do not depend on it.

### Embedding Configuration

```toml
[embedding]
model_name = "mixedbread-ai/mxbai-embed-large-v1"
device = "cuda"
memory_limit_gb = 1.3
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
# Fast quality gate: lint, format check, mypy, unit tests
bash scripts/green_gate.sh

# Arc-end isolated CLI sanity check (CORE + LIVE; LIVE skips when vLLM is unavailable)
.venv/bin/python .agents/skills/qa-sanity/qa.py

# Matcher-sensitive changes
.venv/bin/python scripts/check_matcher_gate_required.py --base HEAD
.venv/bin/python scripts/eval_matching.py --required

# Generated document artifact quality smoke gate
job-applicator document-quality --resume tailored.txt --cover-letter cover.txt --keyword Python

# Private generated-packet quality certification
job-applicator document-quality --private-packet-set --required --min-cases 15 \
  --min-manual-reviews-per-category 5 --max-artifact-age-days 14 \
  --required-category support --required-category risk --required-category network \
  --required-language en --required-language fr

# LLM sampler A/B measurement against private generated-packet cases
.venv/bin/python scripts/eval_llm_sampler.py --dry-run --json
.venv/bin/python scripts/eval_llm_sampler.py --required --integrity-only --json
```

The private packet-set gate is set-level certification, not just per-packet smoke scoring: required
mode enforces a minimum number of passing cases, freshness, and requested category/language
coverage. Missing required private evidence exits `2`; present-but-failing evidence exits `1`.
The sampler harness keeps production defaults unchanged while measuring baseline vs Qwen-shaped
criteria-extraction/transport variants. Repeated held-out runs also report ranking completeness,
criteria and selection stability, résumé/cover alignment, TXT identity across templates, PDF text
retention, and page counts. Applicant claim prose remains deterministic; prose qualification stays
in the separate manual-review sidecar. See `docs/llm-sampler-eval.md`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory (12 GB)                       │
├─────────────────────────────────────────────────────────────┤
│  vLLM Orchestrator (Qwen3-8B-AWQ)       ~7.9 GB            │
│  ├── Cover letter generation                               │
│  ├── Style analysis                                        │
│  └── Job description understanding                         │
├─────────────────────────────────────────────────────────────┤
│  Embedding Model (mxbai-embed-large-v1)  ~1.4 GB           │
│  ├── Resume embedding                                      │
│  ├── Job matching                                          │
│  └── Skill similarity                                      │
├─────────────────────────────────────────────────────────────┤
│  Runtime headroom                    ~1.5-1.8 GB           │
└─────────────────────────────────────────────────────────────┘
```
