# Code-Review Audit — Hardening Arc (`738ef26..1e68dfe`)

**Scope:** the un-PR'd hardening arc on top of merged PR #23 — ~1,286 LoC across
`state.py` (new), `batch_state.py` (new), `utils/llm.py`, `cli.py`, `documents/resume.py`,
`documents/cover_letter.py`, `diagnostics.py`, `models.py`, `scrapers/*`.
**Pinned at SHA `1e68dfe`** (HEAD moved 3× during review; pinned to avoid drift).
Green gate: 518 unit tests passed at the prior SHA.

Method: 7 subsystem review subagents + self-verification of every High/Critical against the
live code. Findings split **CONFIRMED** (verified against code) vs **SUSPECTED**.

---

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

## SUSPECTED / subsystem findings (subagent coverage)

_Pending merge of the 7 review subagents (resume.py consensus, cover_letter, diagnostics,
models, scrapers). To be appended + verified._
