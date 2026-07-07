---
name: qa-sanity
description: Deterministic end-to-end QA / critical-paths sanity check for the job-applicator CLI. Drives every SAFE critical path (doctor, ats-check, match, generate-cover-letter, tailor, batch + crash-recovery) through the real CLI in FULL ISOLATION from real user state, and emits a PASS/FAIL/XFAIL report. Use at the END OF EVERY implementation arc, before opening/merging a PR, after any change to the CLI / commands / workflows / documents / batch / state, or whenever asked to sanity-check, regression-check, smoke-test, QA, or "make sure nothing broke" in job-applicator. Prefer this over ad-hoc manual CLI testing — it is isolated and account-safe by construction.
---

A single deterministic harness, **`.agents/skills/qa-sanity/qa.py`**, that exercises the
job-applicator CLI the way a user would and grades the result. It is the automated form of
a manual end-to-end QA pass: run it at the end of an arc or as a general sanity check.
Paths below are relative to the repo root.

> **Two guarantees that make this safe to run anytime:**
> 1. **Isolation.** Every CLI call runs with `HOME` redirected to a throwaway temp dir, so
>    the tool's real state — `~/.job-applicator/` (dedupe DB, batch progress, cookies, the
>    authenticated browser-profile) — is **never touched**. (Read-only caches — the
>    embedding model, the Playwright browser — are shared so the run stays fast.) The
>    harness aborts if isolation isn't in effect and fingerprints the real DB before/after
>    as proof. *This exists because a non-isolated manual QA run once deleted that DB —
>    don't undo the isolation.*
> 2. **Account safety.** It NEVER runs `login` / `import-cookies` / `search` / `apply` /
>    `check-session` (they touch the real LinkedIn account or launch a real browser+session).
>    Those are probed for `--help` only. Cover letters use inline `--description` (no scraping).

## Prerequisites

The repo set up per `.agents/skills/run-job-applicator/SKILL.md` (venv + deps). The **LIVE**
tier additionally needs vLLM at `http://localhost:8000/v1` and the embedding model cached
(both already true on the dev box); LIVE auto-SKIPs if vLLM is down. If self-hosting, use
`scripts/serve-vllm.sh` (vLLM 0.23.x CUDA 13.0 wheel; defaults to the project's own binary
with `GPU_MEM=0.65`, `MAX_MODEL_LEN=8192`, and `ENFORCE_EAGER=1` for the validated 12 GB
CUDA-embedding coexistence profile).

## Run (agent path)

```bash
.venv/bin/python .agents/skills/qa-sanity/qa.py            # CORE + LIVE (LIVE skips if vLLM down)
.venv/bin/python .agents/skills/qa-sanity/qa.py --core     # offline only (fast, no GPU/LLM)
.venv/bin/python .agents/skills/qa-sanity/qa.py --live     # live only
```

It prints a markdown report to stdout and writes a copy to `/tmp/job-applicator-qa-report.md`.
**Exit 0** = no regressions; **exit 1** = a real regression (a FAIL) or an isolation breach.
A full run is slow (LLM paths are GPU-serial, and a currently-hanging check spends its
timeout); `--core` is seconds.

### Reading the report

| status | meaning | action |
|---|---|---|
| **PASS** | expected-pass check passed | none |
| **FAIL** | expected-pass check FAILED → **regression** | investigate; exit code is 1 |
| **XFAIL** | a known open bug, still failing (expected) | not a regression; see KNOWN_FAIL in `qa.py` |
| **XPASS** | a known bug now PASSES → **it's fixed!** | remove its name from `KNOWN_FAIL` in `qa.py` so it becomes a guarded PASS |
| **WARN** | non-gating signal (e.g. voice-tells elevated) | judgment call |
| **SKIP** | tier/condition unavailable (vLLM down, partial state not observed) | informational |

This is a **living gate**: known bugs are asserted at their *correct* behavior and tagged
`XFAIL`. When you fix one, the harness flips it to `XPASS` and tells you to promote it. New
breakage in a previously-green check surfaces as `FAIL`.

## What it checks

CORE (offline): `--version`, `--help`, unknown-command exit code, `--log-file` requires
`--verbose`, `config-init` (good + bad path), `ats-check` (valid docx, `--json` validity,
bad-input clean-error), `--json --verbose` JSON validity, and that the account-touching
commands expose `--help` (without running them).

LIVE (vLLM): `doctor`, `match --jobs-file` (ranking + JSON + no-scrape), `generate-cover-letter`
(inline; voice-tells as a metric), `tailor --yes` (non-interactive + abort path), `batch`
(multi-job + malformed-input + crash-recovery via deterministic DB-state simulation).

## Known issues (the regression baseline)

**XFAIL: none.** The QA-arc backlog is fully cleared — `KNOWN_FAIL` in `qa.py` is empty, so
every check is a guarded PASS. A new `XFAIL` here means a freshly-triaged bug whose check asserts
the *correct* behavior (XFAIL until fixed → XPASS → promote).

**WARN: none open.** Every design item the manual QA surfaced is resolved + guarded as a PASS
check: `ats-check --strict`, the non-fatal unwritable-`--log-file` warning, and `--json` on every
data-producing command (`doctor`, `generate-cover-letter`, `tailor`, `match`, `batch`, `ats-check`,
`search`, `apply` — only the interactive/status-only `login`/`import-cookies`/`config-init`/
`check-session` omit it).

## Extending it

Add a check by calling `record(name, tier, ok_bool, detail)` (asserting *correct* behavior)
inside `core_checks`/`live_checks`. Assert STABLE signals (exit code, no-traceback, valid
JSON, artifact existence, DB rows) — never exact LLM text, which varies run to run. Keep
account-touching commands to `--help` probes only.

## Gotchas

- **Isolated run uses CONFIG DEFAULTS** (no real `config.toml` is loaded). Reproducible, but
  it won't catch config-specific issues — those need a separate, non-isolated check.
- **Live LLM checks pin `JOB_APPLICATOR_LLM_TEMPERATURE=0.2`** to keep the gate repeatable while
  preserving product defaults for normal CLI use.
- **Don't "fix" a red XFAIL by changing the assertion to match the bug.** The check asserts
  what *should* happen on purpose; that's what makes XPASS a fix signal.
- A check that drives a command expecting it to be **non-interactive** spends its full timeout
  if that command regresses to blocking on input — a timeout (exit 124) is a FAIL, not a hang
  in the harness itself. (This is how the `tailor --yes` hang was caught before it was fixed.)
