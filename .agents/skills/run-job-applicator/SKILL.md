---
name: run-job-applicator
description: Build, run, smoke-test, and drive the job-applicator CLI. Use when asked to run job-applicator, start it, build it, install it, run its tests, check its health (doctor), generate a cover letter, or verify it works after a change.
---

job-applicator is a Python (Typer) CLI for AI-assisted job applications. The agent
path is the committed smoke driver **`.agents/skills/run-job-applicator/driver.sh`**,
which generates a throwaway résumé and exercises the safe surface in two tiers — an
offline CORE tier and a vLLM-gated LIVE tier. All paths below are relative to the
repo root (the unit).

> **Account safety — load-bearing.** This tool runs against the user's *real*
> LinkedIn account. The driver NEVER runs `login`, `import-cookies`, `search`,
> `match`, `apply`, `batch`, or `check-session` — those touch the live account or
> stored credentials. Cover-letter generation is driven with the job text passed
> **inline** (`--description`), so no browser launches and no board is scraped. Keep
> it that way.

## Prerequisites

System binaries (verified via `dpkg -S`: `pdftotext`←`poppler-utils`, `Xvfb`←`xvfb`):

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils xvfb
```

Python 3.12+ is required. (The CLI uses `enum.StrEnum`, which is 3.11+ — see Gotchas.)

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,embeddings,browser,indeed]"
.venv/bin/playwright install chromium
```

`-e ".[dev,embeddings,browser,indeed]"` pulls the test tools, the embedding model
stack, browser-cookie import, and the Xvfb wrapper. The heavy `serve` extra installs
vLLM 0.23.x (CUDA 13.0 wheel) and is only for self-hosting the LLM; skip it if you
have an endpoint.

The LIVE tier needs an OpenAI-compatible LLM at `http://localhost:8000/v1`. Either
point at an existing endpoint (`[llm] api_base` in config), or self-host:

```bash
curl -s http://localhost:8000/v1/models    # is one already up?
bash scripts/serve-vllm.sh                 # otherwise self-host — project's own script (needs `serve` extra + GPU; not run this session)
```

`scripts/serve-vllm.sh` defaults to job-applicator's own `.venv/bin/vllm`, model
`Qwen/Qwen3-8B-AWQ`, `GPU_MEM=0.70`, `MAX_MODEL_LEN=8192`, and `ENFORCE_EAGER=1`. The
eager default avoids a vLLM 0.23 V1 cudagraph-profiling OOM on 12 GB cards. It also puts
the venv bin on `PATH` so flashinfer can JIT-compile a kernel for a fresh model (`ninja`,
from the `serve` extra).

## Run (agent path) — the driver

```bash
bash .agents/skills/run-job-applicator/driver.sh           # CORE + LIVE (LIVE auto-skips if vLLM is down)
bash .agents/skills/run-job-applicator/driver.sh --core    # offline only
bash .agents/skills/run-job-applicator/driver.sh --live    # live only
```

It resolves the repo from its own location (run it from any CWD), uses
`.venv/bin/job-applicator`, and runs every command inside a fresh `mktemp` workdir
(so no real `config.toml` is loaded and nothing is clobbered). Exit 0 = all CORE
checks passed; a CORE failure exits 1; a LIVE *skip* (vLLM down) is not a failure.

Real output of `bash driver.sh` on a provisioned box (vLLM up):

```
[CORE] offline checks (no vLLM / GPU)
  ✓ version
  ✓ help lists commands
  ✓ ats-check (docx)
  ✓ ats-check --json
  ✓ config-init
[LIVE] needs vLLM at localhost:8000 (+GPU)
  ✓ doctor (LLM reachable)
  ✓ generate-cover-letter (real LLM)

PASS=8  FAIL=0  SKIP=0
SMOKE OK
```

| check | tier | what it proves |
|---|---|---|
| `version` / `help` | core | CLI imports and dispatches |
| `ats-check` (docx + `--json`) | core | résumé parsing + ATS scoring, offline |
| `config-init` | core | config scaffolding (written to the temp dir) |
| `doctor` | live | LLM endpoint reachable + model loaded |
| `generate-cover-letter` | live | end-to-end LLM generation against vLLM |

## Direct invocation

The driver's checks, individually (all via the venv entrypoint):

```bash
.venv/bin/job-applicator --version                       # → job-applicator v0.5.0
.venv/bin/job-applicator doctor                          # health: LLM, embeddings, browser, bins, config
.venv/bin/job-applicator ats-check --resume r.docx --json | jq .is_compatible
.venv/bin/job-applicator generate-cover-letter \
  --job-title "Senior Python Engineer" --company "Initech" \
  --description "Async data pipelines; asyncio, Pydantic, PostgreSQL." --resume r.docx
.venv/bin/job-applicator document-quality --private-packet-set --required --min-cases 3 \
  --max-artifact-age-days 14 --required-category support --required-category risk \
  --required-language en --required-language fr --json
```

`doctor` is the fastest "is the whole stack wired?" check — it prints a green/red
line per subsystem and exits 0 when healthy.
The private packet-set document-quality command is the generated-packet certification gate. It
uses local private data under `~/.job-applicator/document-quality-eval/` and should not be used as
a generic smoke test on a machine without that manifest.

## TUI (interactive home screen)

Bare `job-applicator` in a terminal (stdout **and** stdin a TTY) opens the full-screen
Textual UI; `job-applicator tui` is the explicit form. Piped / non-TTY prints help, and
`tui` in a non-TTY exits 1 with a clean message (so pipes/CI never hang). It's the
navigable home over the job funnel — drive it with keys:

| key | action |
|---|---|
| ↑↓ / j k · `/` · `r` · `q` | navigate · filter · refresh · quit |
| `e` | set your résumé in-app (saved to `config.toml`, atomically) |
| `s` | search (modal) → scrape + score against the résumé → results land ranked |
| `t` / `c` | tailor / cover-letter the selected job (local LLM, background worker) |
| `a` | apply — dry-run by default; a real submit needs the danger checkbox |

> **Account safety.** Launching, navigating, and filtering touch only local state. `s`/`a`
> touch the real LinkedIn account ONLY after an explicit in-app confirm; apply is dry-run
> unless you tick the danger checkbox; it never auto-logs-in. So the TUI is **not** part of
> the smoke driver (it's interactive and would touch the account on `s`/`a`).

**Verify headlessly** (no TTY, no account): the TUI is covered by Pilot tests —
`.venv/bin/python -m pytest tests/unit/test_tui.py -q` drives mount/nav/filter/actions with
mocked seams and asserts no browser is built without an explicit confirm.

## Run (human path)

The actual job-search commands (`search`, `match`, `apply`, `batch`) need a seeded
LinkedIn session and touch the live account — out of scope for a smoke run. Seed a
session once as a human (`job-applicator login`, headed) or import cookies
(`job-applicator import-cookies`), then e.g. `job-applicator match --resume r.pdf`.
Easy Apply is dry-run by default; it fills the form and previews the generated cover
letter, but real submission requires `apply --submit`.

## Test

```bash
.venv/bin/python -m pytest -m unit -q      # fast green gate (no browser/GPU/vLLM)
```

`pytest -m live` (35 tests) needs vLLM + GPU; `pytest -m integration` (9) is
browser-wiring only. Run pytest from the repo root (it needs the repo CWD).

## Gotchas

- **`~/.local/bin/job-applicator` may use system Python 3.10** → `ImportError:
  cannot import name 'StrEnum' from 'enum'`. Always use `.venv/bin/job-applicator`
  (the driver hardcodes this).
- **`config-init` defaults to `./config.toml`** — that would overwrite the repo's
  real, credentialed (gitignored) `config.toml`. Always pass `-o <path>`; the driver
  runs from a temp dir and writes `sample-config.toml`.
- **litellm prints noise on SUCCESS** — a red "Give Feedback / Get Help" banner, and
  occasionally a transient `litellm.BadRequestError` that the validation-retry
  recovers from. The reliable success signal is **exit 0 + "Generated Cover Letter"**,
  never the presence of "error" in the output.
- **JSON → stdout, logs/Rich → stderr.** `ats-check --json | jq .` is clean; the
  `[time] INFO …` lines you see are on stderr.
- **`playwright install` warns "BEWARE: your OS is not officially supported …
  downloading fallback build for ubuntu22.04-x64"** — harmless; the fallback works.
- **`pip install -e` uninstalls then reinstalls `job-applicator-0.5.0`** (exit 0) —
  normal for an editable reinstall, not an error.

## Troubleshooting

- **`ImportError: cannot import name 'StrEnum'`** → you invoked the Python-3.10
  shim. Use `.venv/bin/job-applicator` (or activate the 3.12 venv).
- **LIVE checks SKIP, or `doctor` shows LLM `✗ unreachable`** → no LLM at
  `localhost:8000`. Start one (`bash scripts/serve-vllm.sh`) or set `[llm] api_base`
  to a reachable endpoint. CORE checks still pass without it.
- **vLLM is reachable with `curl`, but Python reports socket permission denied** → the sandbox is
  blocking Python socket creation. Re-run the CLI/QA command outside the sandbox. If vLLM leaves
  too little VRAM for CUDA embeddings, pin `JOB_APPLICATOR_EMBEDDING_DEVICE=cpu` for the isolated
  QA run.
- **`ats-check` on a PDF returns empty/garbled text** → install `poppler-utils`
  (`pdftotext`). The driver uses a `.docx` (python-docx, a core dep) to sidestep this.
