# Manual QA — End-to-End Pass #2 (2026-06-22)

Second full manual sanity pass driving the **live CLI** as a user, after the QA arc
(PRs #38–48). Precursor artifact for hardening new findings into the `/qa-sanity`
living gate (XFAIL → fix → XPASS).

## Method
- **Isolated**: `HOME`→temp (`/tmp/qahome`, isolates `~/.job-applicator`), CWD→temp
  (`/tmp/qawork`, no real `config.toml`), shared read-only model caches (`HF_HOME`,
  `PLAYWRIGHT_BROWSERS_PATH`). **Isolation proven**: real `~/.job-applicator` SHA
  `16552113…` / 1508 files identical before & after.
- **Account-safe** (verified from code, not assumed): `apply`/`check-session` launch a
  browser before any guard → `--help` only; `batch --query` scrapes → always
  `--jobs-file`; `match` is fully offline (file/demo). `search`/`login`/`import-cookies`
  excluded (browser/credentials).
- **Adversarial**: happy + unhappy + multiple edge cases per command. Fixtures:
  good/minimal/empty/binary/bad-pdf/no-ext/unsupported/huge(355KB)/unicode résumés;
  valid/empty/malformed/missing-fields/not-a-list/huge(120) jobs-files.
- vLLM up (Qwen3.5-4B-AWQ); 633 unit green on `main` (c4b216b).

## Findings

| ID | Sev | Command(s) | Finding | qa-sanity |
|----|-----|-----------|---------|-----------|
| **B6** | **HIGH** (match) | `match` bug; `batch` polish | `match --jobs-file <bad>` → **raw traceback** (4 variants); `batch` clean-but-verbose (see below) | **GAP** (tests valid file only) |
| **B7** | MED-UX | `match` | No `--jobs-file` → silently matches 2 hardcoded **demo jobs** as real results | **GAP** |
| **B2** | LOW | docx/txt/md | `--ocr-mode <invalid>` validated for PDF/image but silently ignored for docx/txt/md (inconsistent) | GAP |
| **B8** | LOW | `match` | `--top-k 0/-5`, `--min-score 2.0` (out of range) silently accepted | GAP |
| **B3** | LOW | `ats-check` | Empty/no-text résumé → scored 0.14 "Not Compatible" + suggestions, no "empty" flag | GAP |

**Triage — settled bug vs fix-direction design call.** Only **B6** is an unambiguous bug
(raw tracebacks violate a CLAUDE.md convention) → harden as XFAIL now. **B7 / B2 / B3 / B8**
each encode a fix DIRECTION that's defensible either way (`match` demo-jobs intentional-vs-error;
reject-vs-default a typo'd `--ocr-mode`; flag-empty-vs-score-it; clamp-vs-accept out-of-range) —
these need the user's intent BEFORE test-ifying, else an XFAIL bakes in an unchosen direction.
| **B4** | LOW | `config-init` | `-o <directory>` → "config.toml already exists. Skipping." + exit 0 (misleading) | n/a |
| **U1** | LOW | apply/gcl/tailor/batch | `--style-guide` has 3 different help texts (same flag) | n/a |
| **U2** | LOW | search/apply/match/batch | "max count" flag inconsistent: `--max -n` / `--limit -n` / `--top-k -k` | n/a |
| **U3** | TRIV | `doctor` | "Browser" line wraps the chromium path onto an unindented line | n/a |
| **U4** | TRIV | `batch` | "Tailoring 1 **jobs**..." pluralization | n/a |
| **U5** | obs | `batch` | `--resume-run` on a COMPLETED run → "No incomplete run; starting new" (silent re-run) vs "already complete" | n/a |

### B6 (HIGH for `match`) — jobs-file errors aren't typed → match re-raises a raw traceback
The convention is typed errors, never bare tracebacks (CLAUDE.md). Both commands wrap the
body in `except JobApplicatorError → clean ⚠` + `except Exception → record + RAISE` (the
re-raise → Typer pretty-traceback is the intended path for UNEXPECTED errors).
- `match --jobs-file <bad>` — load is OUTSIDE any inner try (cli.py:985–991), so
  `JSONDecodeError`/`ValidationError`/`TypeError`/`FileNotFoundError` (none a
  `JobApplicatorError`) fall to `except Exception` → **re-raised → raw traceback** for all
  four bad inputs. REAL BUG.
- `batch --jobs-file <bad>` — has an INNER try/except (cli.py:1242–1250) catching them as
  "Error reading jobs file: <msg>" → clean, but the pydantic `ValidationError` str is
  VERBOSE ("3 validation errors for JobListing …"). Caught, NOT a traceback — polish only.
  (My first read mis-tagged this as a leak; re-measured — it's caught + prefixed.)
- **Fix**: a shared `_load_jobs_file()` raising a clean typed `DocumentError` (concise
  message) for missing / malformed / not-a-list / invalid-listing, used by BOTH commands —
  match's raw traceback → caught by its existing `except JobApplicatorError`; batch's
  verbose pydantic str → a concise message. Mirrors the résumé loader's PR #40 wrapping.

### B2 (LOW — cosmetic) — `--ocr-mode` validation is inconsistent (NOT absent)
The loader DOES validate: `_load_pdf`/`_load_image` raise a clean
`DocumentError("Invalid ocr_mode '<x>'. Valid modes: auto, on, off")` (resume.py:96) — so
`--ocr-mode bogus` on a PDF/image is rejected cleanly. But docx/txt/md skip
`_load_pdf`/`_load_image`, so a bogus value is silently accepted there (verified on
good.docx: exit 0, score 1.00) — harmless (ocr_mode is a no-op for those formats) but
inconsistent. Optional fix: validate once in `load()` / make `--ocr-mode` a typer `Enum`
so a typo is rejected uniformly. (Earlier "could mis-route the PDF path" was wrong — the
PDF path is guarded.)

### B7 (MED-UX) — `match` silent demo jobs
`match --resume r.docx` (no `--jobs-file`) → "Loaded 2 jobs" + TechCorp/StartupXYZ at
82% (hardcoded demo, cli.py:996–1013). A user who forgets `--jobs-file` gets fake
results presented as real. `batch` already does the right thing ("Provide --jobs-file or
--query."). Make `match` consistent (error) or clearly label demo data.

## Confirmed good (regression — no defect)
- Required-flags → clean typer exit 2; bad/binary/bad-pdf/unsupported/missing/no-ext
  résumé → clean wrapped `⚠` errors (PR #40); `no --resume` → "Resume path required."
- `batch` missing-file/no-source/empty → clean messages, **no scrape fallback**.
- `--log-file` requires `--verbose`; unwritable → warn + exit 0 (PR #46);
  `ats-check --strict` gates exit (PR #46); bogus flag/subcommand → clean typer errors.
- **`--json` stdout-purity holds** (doctor/match/gcl/tailor/ats-check pipe clean to `jq`)
  — PRs #46/#47/#48 work confirmed live.
- Interactive `tailor` loop (accept / invalid-choice reprompt / quit) + `--yes` + `--json`;
  `gcl --style-guide` applies style; `batch` multi-job + ATS-before/after summary.
- **Stress**: 355 KB résumé, 120-job jobs-file, unicode — all handled, no crash/timeout.

## Retracted (measure-don't-assert)
- **B1 (no `--resume` → "not found: /path/to/your/resume.pdf")** was a TEST ARTIFACT, not
  an app bug: the `config-init` battery wrote `config.toml` into the CWD the wrapper
  `cd`s to, so later commands read its placeholder `resume_path`. After cleanup all
  commands correctly say "Resume path required." (Lesson: **config-init's default write
  pollutes same-CWD commands** — verify the qa.py harness writes config-init via `-o` to a
  temp path, never the CWD default.)

## Recommended `/qa-sanity` additions (harden these)
1. **B6 — harden NOW** (settled bug, was-GAP): `match`/`batch --jobs-file
   <malformed|not-a-list|missing-fields|missing>` → assert exit≠0 AND **no traceback**
   (`has_traceback(cp)` false). XFAIL until B6 fixed.
2. **B7 / B2 / B3 / B8 — confirm intent first** (fix-direction design calls, see Triage):
   each becomes a check ONLY after the user picks the direction (e.g. `match` no-jobs-file →
   error vs labelled-demo; `--ocr-mode` reject vs default; empty-résumé flag vs score;
   out-of-range numeric clamp vs accept). Don't XFAIL an unchosen direction.
3. qa.py config-init isolation: **VERIFIED SAFE** — it writes via `-o $WORK/c.toml` (not the
   default `./config.toml`) and runs every command with `cwd=$WORK`, so no stray `config.toml`
   contaminates later checks (the B1 trap was MY wrapper's default-write, not the harness). No
   change needed.
4. Add a stale-CV fixture so the `tailor` date-audit gate's fire-branch is covered LIVE
   (currently unit-only).
