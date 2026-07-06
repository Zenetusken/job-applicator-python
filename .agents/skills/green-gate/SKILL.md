---
name: green-gate
description: >-
  Run the job-applicator GREEN GATE — the canonical pre-commit / pre-PR quality gate
  (ruff check · ruff format --check · mypy strict on src/ · pytest -m unit) as ONE fail-fast
  command with a clear per-stage PASS/FAIL. Use this whenever the user asks to "run the gate",
  "check it's green", "is the build green", "run the checks", "run lint + types + tests",
  "verify nothing broke", or before committing / pushing / opening a PR in job-applicator-python
  — even if they don't say the word "gate". Prefer this over running the four checks ad-hoc so the
  gate is ALWAYS the same canonical scope; it is the fast green-light for a PR (the heavier
  qa-sanity / integration / live tiers are separate).
---

# Green Gate

The single command that decides whether a change is shippable in **job-applicator-python**:

**ruff (lint) → ruff format --check → mypy (strict, `src/`) → pytest -m unit**, in that order,
**fail-fast**. Run it before every commit / push / PR, and after any code change to confirm
nothing broke. It mirrors the gate the project documents in `AGENTS.md` / `CLAUDE.md`.

## Run it

```bash
bash scripts/green_gate.sh
```

The agent wrapper at `.agents/skills/green-gate/scripts/gate.sh` delegates to this canonical
project script. The script finds the repo root itself (works from any working directory) and uses
the project `.venv`. **Exit 0 = GREEN** (all stages passed). Non-zero = a stage failed; the output
names the stage and, for the auto-fixable ones, the exact fix command.

## Interpreting the result + what to do on failure

Report the outcome plainly — which stage failed and the relevant error lines — don't bury it
under other output. Then:

- **ruff check** (lint) — most diagnostics auto-fix with `.venv/bin/ruff check --fix src/ tests/`,
  but read what changed first; don't blind-`--fix` a real bug into silence.
- **ruff format --check** — apply formatting with `.venv/bin/ruff format src/ tests/`, then re-run.
- **mypy** (strict) — a real type error; fix it at the source. Don't silence it with a blanket
  `# type: ignore` — prefer precise annotations or typed stubs (the project runs mypy strict on
  purpose, and `tests/` are intentionally checked by ruff only, not mypy).
- **pytest -m unit** — a failing/broken test; read the failure and fix whichever is wrong (code or
  test), then re-run. A timeout (exit 124) on a `--yes` / non-interactive path usually means a
  command regressed to blocking on input, not a slow test.

**Re-run the whole gate after any fix** — a fix for one stage can trip a later one (e.g. an
auto-format changes a line mypy then flags).

## Why one command, one scope

The gate's value is that it is **always the same canonical scope** — `src/ tests/` for
ruff + format, `src/` for mypy (strict), `-m unit` for pytest (the fast suite: no browser, GPU,
or vLLM). Running the four checks ad-hoc drifts — wrong paths, forgetting `format --check`, or
running the whole suite when you meant `-m unit` — so results stop being comparable run-to-run.
The bundled script pins the scope so every run means the same thing.

## Scope — what this is and isn't

- This is the **fast** gate and the always-run baseline: the green-light for a PR.
- It does **not** run the `-m integration` / `-m live` tiers, or the end-to-end **`qa-sanity`**
  harness — those are heavier and situational (use the `qa-sanity` skill at an arc's end or when
  asked to sanity-check the whole CLI). Green gate = every change; qa-sanity = deep check.
- Matcher/scoring changes have a **private-data companion gate**: run
  `.venv/bin/python scripts/check_matcher_gate_required.py --base <base>` to detect whether the
  companion gate is required, then run `.venv/bin/python scripts/eval_matching.py --required`
  after edits to `embeddings/matching.py`, skill
  extraction/normalization/grounding, score weights, thresholds, or `[matching] target_roles`
  behavior. This is not part of the universal green gate because it depends on the user's private
  `~/.job-applicator/matching-eval/gold-set.csv` and live funnel DB; `--required` exits non-zero
  when those inputs are missing or incomplete, so a matcher-sensitive change cannot be certified by
  absence of evidence.
- If the `.venv` is missing the script exits **2** with a pointer — set it up via the
  `run-job-applicator` skill first, then re-run.
