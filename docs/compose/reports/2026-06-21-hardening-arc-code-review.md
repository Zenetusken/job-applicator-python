# Code-Review Audit — Hardening Arc (`738ef26..1e68dfe`)

**Scope:** the un-PR'd hardening arc on top of merged PR #23 — ~1,286 LoC across
`state.py` (new), `batch_state.py` (new), `utils/llm.py`, `cli.py`, `documents/resume.py`,
`documents/cover_letter.py`, `diagnostics.py`, `models.py`, `scrapers/*`.
**Pinned at SHA `1e68dfe`** (HEAD moved 5× during review; pinned to avoid drift). Line numbers
are as-of `1e68dfe`. **C1–C3 re-verified still live at HEAD `90dae9e`** (two newer commits —
skill-normalization + LinkedIn dry-run validation — don't touch the buggy paths; at HEAD
`start_run` is cli.py:1541). Green gate: 518 unit tests passed at the pinned SHA.

Method: 7 subsystem review subagents + self-verification of every High/Critical against the
live code. Findings split **CONFIRMED** (verified against code) vs **SUSPECTED**.

---

## VALIDATION UPDATE — Cluster 1 fixes verified end-to-end

All four Cluster 1 fixes were exercised through live UI / execution-path tests with vLLM up:

- **H2 ✅ VALIDATED** — Seeded a batch run with one job in `TAILORED` state and status `running`;
  `job-applicator batch --resume-run` re-processed the job and generated its cover letter instead of
  skipping it. `list_completed_jobs` now excludes `TAILORED`.
- **H1 ✅ VALIDATED** — Created a blank PDF and loaded it with `ocr_mode='auto'`;
  `ResumeLoader.load()` raised `DocumentError: insufficient extractable text` instead of returning an
  empty `ResumeData`.
- **H3 ✅ VALIDATED** — `strip_thinking_process(None)` returns `''` without raising; normal text still
  passes through.
- **M4 ✅ VALIDATED** — Tripped a `CircuitBreaker` and called the `@async_retry(..., exclude=(CircuitOpenError,))`
  wrapper; `CircuitOpenError` surfaced in ~0.00s with no backoff retry.

Full regression gates after updating tests to match the new H1 behavior:
`pytest -m unit` 553 passed, `pytest -m live` 21 passed; `ruff`, `ruff format --check`, `mypy` all green.

## STATUS UPDATE — fixes landed mid-review

The user fixed the three high-severity findings in the working tree **during** the review
(re-verified against the current tree):
- **C1 ✅ FIXED** — `resuming` flag now guards `start_run`; `start_run` gained `reset: bool=True`.
- **C2 ✅ FIXED** — `ApplicationResult.timestamp` now `default_factory=lambda: datetime.now(UTC)`
  (models.py:206); stored + bound are both tz-aware UTC → the TEXT comparison is now sound.
  (Residual nit: `count_today` docstring still says "local midnight"; code uses UTC.)
- **C3 ✅ FIXED** — `state.record(ar)` now gated by `if submit:` (cli.py:895); dry-runs no longer
  pollute the cap. `count_today` docstring updated to "real applications submitted".
- **Still live:** C4 (MEDIUM, global breaker), C5 (LOW, unclosed connections), C6 (LOW, empty
  `statuses` → `IN ()`).

The findings below are kept as the original audit record (line numbers as-of `1e68dfe`).

## ADDENDUM — fan-out findings (cross-verified at HEAD `132d339`)

The 7-reviewer fan-out reported back after the interim summary and surfaced **4 significant bugs the
self-review missed**, each re-verified against the live code. Most urgent: **H2 is newly LIVE because
the C1 fix unmasked it.**

**H2 · HIGH (now LIVE post-C1-fix) · `batch_state.py:246-251` + `cli.py:1555`** — `list_completed_jobs`
counts **TAILORED as completed** (`{COMPLETED, COVER_LETTER, TAILORED}`), but TAILORED is *intermediate*:
cli records TAILORED (cli.py:1668) then COMPLETED only after cover-letter gen (cli.py:1716/1734). That set
drives `pending_matches` (skips the whole job), so a job that crashed *after tailoring, before its cover
letter* is treated as done on resume → **its cover letter is silently never generated**. Masked by the old
C1 wipe; **the C1 fix unmasks it.** Bonus: `BatchJobStatus.COVER_LETTER` is recorded nowhere — dead enum.
*Fix: skip-set = `{COMPLETED}` (+ SKIPPED); keep TAILORED out so half-done jobs resume the cover-letter step.* (rev-batch)

**H1 · HIGH · `resume.py:104-110` (`_load_pdf`, auto mode)** — when every text extractor fails
(`fallback_result is None`) AND OCR returns "" *without raising* (verified `ocr.py:76,81` return "" with no
content guard), the `fallback_result is None` short-circuit returns the empty OCR result regardless → a
fully-unreadable PDF yields a **silent empty `ResumeData`** instead of raising; downstream match/tailor then
runs on nothing. `off` mode guards this (OCR_THRESHOLD); `auto` doesn't. *Fix: require
`len(raw_text.strip()) >= OCR_THRESHOLD` before returning, else raise.* (rev-resume)

**H3 · MEDIUM · `cover_letter.py:198,323` (+`style_analyzer.py:215`)** — the litellm *fallback* path does
`strip_thinking_process(response.choices[0].message.content)`, but `message.content` is `Optional[str]`
(None on an empty completion) and `strip_thinking_process` (utils/llm.py:86) does `"…" in text` → TypeError
on None, caught + re-wrapped as a misleading *"LLM call failed: argument of type 'NoneType'…"*. Instructor
path is safe (Pydantic str). *Fix: guard empty content, or make `strip_thinking_process` accept `str | None`.* (rev-docs)

**M4 · MEDIUM · `utils/llm.py:255` × `cover_letter.py:135,204`** — the breaker raises a plain `LLMError`
when open, but `generate`/`_generate_raw` are `@async_retry(exceptions=(LLMError,))`, so the retry layers
**catch the circuit-open error and retry against the already-open breaker**, burning nested jittered backoff
(up to 30s) before surfacing "circuit open." Fail-fast defeated. *Fix: distinct `CircuitOpenError(LLMError)`
excluded from the retry layers.* (rev-llm — sharpens M2)

**Also surfaced + verified (Medium/Low):** confidence uses **substring** section matching (`"skills" in "no
skills listed"`) + a vocabulary drifted from `_extract_skills_section` (resume.py:228 vs 381) — skews
consensus; **phone regex** false-positives on digit/year runs (resume.py:222,257) — pollutes `phone` +
confidence; `ValidatedOutput` re-rolls the **identical prompt** without feeding back the error
(utils/llm.py:286); **temperature** passed on the litellm fallback but not the instructor path — config
ignored on the primary path; auto-`run_id` hash includes `top_k/min_score/cover_letter` but `find_existing_run`
matches without them (cli.py:1496 vs batch_state.py:169); `_run_pymupdf` O(n²) string concat; breaker
thresholds hardcoded (not `AppSettings`); no half-open probe (thundering herd on cooldown expiry); `refine`
skips the breaker+validation `generate` uses. (rev-resume / rev-llm / rev-docs / rev-batch)

Reviewers also **confirmed CLEAN** (so they need no re-check): doctor blocking logic (`DoctorReport.ok` gates
only on reachable+HTTP-200), `llm_call_error` timeout-vs-connection classification, the breaker state machine
(monotonic clock, rolling window) apart from half-open, consensus tie-break/empty-candidates, SQL-injection
safety in both stores, and the `IS ?` null-safe matching.

## CONFIRMED findings (self-verified)

### C1 — CRITICAL · Bug · `cli.py:1519` + `batch_state.py:150`
**`--resume-run` wipes the progress it is supposed to restore — the crash-recovery feature is non-functional.**
The batch command computes/looks up `effective_run_id` (cli.py:1502–1517 `if resume_run:` →
`find_existing_run` → sets the existing id), then **unconditionally** calls
`batch_state.start_run(run_id=effective_run_id, …)` at cli.py:1519. `start_run` ends with
`conn.execute("DELETE FROM batch_jobs WHERE run_id = ?", (run_id,))` (batch_state.py:150) —
deleting every recorded job for that run. So on resume: find the run → **delete its jobs** →
`list_completed_jobs` returns `[]` → `pending_matches == all matches` → everything re-processed.
The `start_run` comment even says *"unless the caller is explicitly resuming"* — the caller does
not honor it.
**Fix:** only `start_run` when NOT resuming an existing run:
```python
resuming = False
if resume_run:
    existing = batch_state.find_existing_run(...)
    if existing:
        effective_run_id = existing; resuming = True; ...
if not resuming:
    batch_state.start_run(...)
```
or add a `reset: bool = True` param to `start_run` and pass `reset=False` on resume.

### C2 — HIGH · Bug · `state.py:139` (`count_today`) + `models.py:190`
**Daily-application cap is unreliable — naive-local timestamps compared against a UTC-aware bound as TEXT.**
`ApplicationResult.timestamp = Field(default_factory=datetime.now)` (models.py:190) is **tz-naive
local**. `record` stores `result.timestamp.isoformat()` → e.g. `2026-06-21T10:30:00` (no offset).
`count_today` builds `today = datetime.now(UTC).replace(hour=0,…)` → `2026-06-21T00:00:00+00:00`
(UTC, with offset) and does `WHERE applied_at >= ?`. SQLite compares these as **TEXT
(lexicographic)**, mixing formats (offset vs none) and timezones (local vs UTC). The window is
both semantically wrong (UTC midnight ≠ the user's local day) and lexicographically fragile
(`+00:00` vs `.123456` vs bare — ordering flips on format, not time). The docstring claims
"local midnight." Safety-relevant: this cap throttles applications to the user's **real** account.
**Fix:** store tz-aware UTC consistently (make `timestamp` default `datetime.now(UTC)`), compute
the bound the same way, and prefer comparing on a normalized column (or store epoch seconds).

### C3 — MEDIUM · Bug · `cli.py:889` + `state.py:137`
**Dry-run (the default mode) pollutes the state DB and consumes the daily cap.**
`state.record(ar)` (cli.py:889) runs for every job, **not gated by `submit`** (it sits in the
`with console.status(...)` block, after the `if submit:` cap re-check). In dry-run,
`applicator.apply(submit=False)` returns `status=SKIPPED` (base.py:49), so a row is written.
`count_today` (state.py:137) has **no status filter** → those SKIPPED rows count toward the cap.
Result: testing with the *default* dry-run accumulates "applications" and burns the real daily cap
without ever submitting.
**Fix:** gate `state.record(ar)` on `submit`, and/or filter `count_today` to
`status = 'submitted'` (the cap should count real submissions, not dry runs).

### C4 — MEDIUM · Design · `documents/cover_letter.py:26` + `utils/llm.py:198`
**Module-global mutable circuit-breaker violates the "no global mutable state" convention.**
`_CIRCUIT_BREAKER = CircuitBreaker(name="cover-letter")` is a module-level instance whose
`_failures`/`_open_until` mutate across calls. CLAUDE.md/AGENTS.md mandate "No global mutable
state. Pass via config/context objects." It also makes the breaker un-isolatable in tests and
couples unrelated in-process operations. (Thread-safety of the unlocked `_failures` list is
**low-risk** given the asyncio single-thread model — the sync `_record_*`/`_is_open` methods don't
await — but would matter if ever driven from threads.)
**Fix:** own the breaker on an injected service/context object, or document it as a deliberate
process-singleton exception.

### C5 — LOW · Bug · `state.py:60` + `batch_state.py:92` (`_connect`)
**SQLite connections are committed but never closed.** `with self._connect() as conn:` enters a
transaction context that commits/rolls-back on exit but does **not** close the connection. In
CPython refcounting reclaims it at method return, so it's not a true FD leak today — but it relies
on GC semantics and is the documented anti-pattern.
**Fix:** `with contextlib.closing(self._connect()) as conn, conn:` (close + transaction), or a
shared connection with explicit lifecycle.

### C6 — LOW · Bug · `state.py:113-129` (`has_applied`)
**Empty `statuses` set produces invalid SQL.** If a caller passes `statuses=set()`,
`status_placeholders=""` → `... AND status IN () LIMIT 1`, a syntax error in SQLite. The
None-default path is safe; an explicit empty set crashes.
**Fix:** short-circuit `if not statuses: return False` (or treat empty as "any status").

### C7 — LOW · Pattern · `state.py` / `batch_state.py` (testing)
**The confirmed bugs (C1, C2) live in cli.py orchestration, which has no test.**
`test_batch_state.py` / `test_state.py` test the store methods in isolation (all pass), so the
green gate stays green while the *integration* (resume flow, daily-cap enforcement) is untested —
exactly where C1/C2 hide.
**Fix:** add integration tests: a resume run that asserts completed jobs are skipped (not wiped);
a daily-cap test that records across a day boundary / mixed tz.

---

## Subsystem findings (direct review)

> Method note: a 7-subagent fan-out was launched but the named background agents did not report
> back, so coverage of `resume.py`, `cover_letter.py`, `diagnostics.py`, `models.py`, scrapers was
> completed by direct review. Green gate on the fixed tree: **547 passed**.
> **Live findings below re-verified current as of HEAD `132d339`** — the three live-finding files
> (`cover_letter.py`, `resume.py`, `utils/llm.py`) are unchanged since the `1e68dfe` audit base.

### STILL LIVE — Medium

**M1 (=C4) · Design · `cover_letter.py:26,202`** — `_CIRCUIT_BREAKER` module-global mutable state;
violates "no global mutable state." Low runtime risk (asyncio); testability/coupling cost.

**M2 · Design/Pattern · `cover_letter.py:135,204,235` + `utils/llm.py:198`** — **redundant retry
layers poison the shared global breaker.** `@async_retry(max_attempts=2)` decorates BOTH `generate`
and `_generate_raw`, layered with `ValidatedOutput(max_retries=1)` + instructor `max_retries=1` + a
litellm fallback. The breaker caps *actual* HTTP calls at ~3 (after `failure_threshold=3` it opens
and the rest fast-fail), so the cost is not call volume — it's that **one job's amplified transient
failures trip the process-global `_CIRCUIT_BREAKER`, after which every subsequent in-process
cover-letter fast-fails for the full `recovery_timeout` (30s)**: a batch-wide cascade from a single
blip (compounds M1). *Fix: retry at one layer; scope/parameterize the breaker so a transient on one
job doesn't open it for the whole batch.*

**M3 · Bug · `resume.py:195-207` (`_is_password_protected`)** — relies on `fitz.open` *raising*
for protected PDFs, but PyMuPDF opens an encrypted PDF without raising (sets `doc.needs_pass`). A
password-protected PDF slips through to a misleading "insufficient extractable text" error instead
of the clear password message. *Fix: check `doc.needs_pass` / `doc.is_encrypted`.*

### STILL LIVE — Low

- **L1 · `state.py:138`** — `count_today` docstring says "local midnight"; code uses UTC (C2 residual).
- **L2 · `cover_letter.py:246-325` (`refine`)** — duplicates the instructor→litellm fallback block
  from `_generate_raw` and skips the circuit breaker + `ValidatedOutput` that `generate` uses
  (inconsistent hardening + DRY).
- **L3 · `cover_letter.py:178,306`** — broad `except Exception` around instructor→fallback can mask
  non-LLM errors.
- **L4 · `diagnostics.py:277-284`** — `run_diagnostics` awaits the LLM probe then the browser check
  serially; `asyncio.gather` the two independent async checks to cut doctor latency (~5s + launch).
- **L5 · `diagnostics.py:55`** — module-load playwright import side-effect.
- **L6 · `diagnostics.py:266-268`** — `check_config` calls `ensure_output_dir()` (creates a dir),
  against the module's "side-effect-free" docstring.
- **L7 · `resume.py:88-89`** — duplicated comment line (copy-paste).
- **L8 · `resume.py:374`** — redundant `import re` (already module-level).
- **L9 · `resume.py:294-303`** — `elif len>1` and `else` branches are identical (dead branch).
- **L10 · `resume.py:209-234`** — `_compute_confidence` rewards raw length, so a verbose-but-garbled
  extraction can outscore a clean shorter one in the consensus pick (heuristic limitation).
- **L11 · `resume.py:249,279,328`** — `lines` rebound 3× in `parse_text` (clarity).
- **L12 · `scrapers/linkedin.py` `check_session`** — catches only `NavigationError`; a non-wrapped
  transient (e.g. a raw Playwright timeout) would propagate, against the "capture in details" contract
  (SUSPECTED).
- **L13 · `cover_letter.py:398` `generate_from_template`** — returns an LLM *prompt*, not a letter,
  despite name/docstring; **pre-existing + test-only caller** (no production use) → cosmetic.

### Clean (reviewed, no material findings)
`models.py` new check-models (`extra="forbid"`, no mutable defaults, enum `.value` round-trips through
SQLite TEXT), scraper `check_session` implementations, `diagnostics.py` cache/token probes,
`resume.py` consensus selection + confidence math (no div-by-zero, sensible tie-break).
