# LLM Skill Extraction Critical-Path CLI Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or execute inline.

**Goal:** Verify every critical path that touches the new LLM-driven skill extraction (and related cover-letter/tailor UX) through live CLI calls only, then fix any inconsistencies.

**Scope:**
- `job-applicator match` — LLM skill extraction triggered when `requirements` is empty.
- `job-applicator batch` — multi-job matching + tailoring, skill extraction on each job.
- `job-applicator tailor` — matcher invoked when `--min-score > 0`, skill extraction on description-only jobs.
- `job-applicator generate-cover-letter` — LLM path used by match/batch/apply dry-run.
- `job-applicator apply --validate` — dry-run application path (cover-letter preview).
- `job-applicator doctor`, `status`, `tui` — health/status UI sanity.

**Fix criteria:** Any crash, unclear error, misleading output, inconsistent flag behavior, or missing information observed from the user's perspective is in scope.

---

## Task 1: Prepare fixtures

**Files:**
- Create: `/tmp/resume_smoke.txt`
- Create: `/tmp/jobs_empty_reqs.json`
- Create: `/tmp/jobs_explicit_reqs.json`
- Create: `/tmp/jobs_mixed.json`

- [ ] **Step 1: Write a synthetic plaintext résumé**

```text
Andrei Petrov
Software Engineer

Skills: Python, FastAPI, PostgreSQL, Docker, Kubernetes, React, JavaScript, AWS, Terraform, Git, CI/CD.
Experience: Built backend services with Python and FastAPI, deployed on Kubernetes, used PostgreSQL and Docker.
Education: B.S. Computer Science.
```

- [ ] **Step 2: Write jobs with empty `requirements` to force LLM extraction**

```json
[
  {
    "board": "linkedin",
    "title": "Backend Engineer",
    "company": "Smoke",
    "url": "http://example.com/1",
    "description": "We need a react native engineer with python experience. Knowledge of docker and kubernetes is required.",
    "location": "Remote"
  }
]
```

- [ ] **Step 3: Write jobs with explicit `requirements` to bypass LLM extraction**

Same as Step 2 but add `"requirements": ["Python", "Docker", "Kubernetes", "React Native"]`.

- [ ] **Step 4: Write a multi-job mixed fixture**

3 jobs: one with empty requirements, one with explicit requirements, one with a vague description that should still extract valid skills.

---

## Task 2: Validate `match`

- [ ] **Step 1: Empty requirements path triggers extraction**

Run:
```bash
job-applicator match --resume /tmp/resume_smoke.txt --jobs-file /tmp/jobs_empty_reqs.json --top-k 1
```

Expected: reaches Submit, no crash, outputs a match table.

- [ ] **Step 2: Explicit requirements bypass extraction**

Run:
```bash
job-applicator match --resume /tmp/resume_smoke.txt --jobs-file /tmp/jobs_explicit_reqs.json --top-k 1 --json
```

Expected: JSON output with matched skills, no LLM extraction log line.

- [ ] **Step 3: Error UX when resume missing**

Run:
```bash
job-applicator match --jobs-file /tmp/jobs_empty_reqs.json --top-k 1
```

Expected: clear red error, exit code 1.

---

## Task 3: Validate `batch`

- [ ] **Step 1: Mixed jobs batch run**

Run:
```bash
job-applicator batch --resume /tmp/resume_smoke.txt --jobs-file /tmp/jobs_mixed.json --top-k 3 --no-cover-letter --json
```

Expected: JSON summary, all jobs processed, no unhandled exceptions.

- [ ] **Step 2: Resume-run idempotency**

Run the same command with `--resume-run`.

Expected: skips already completed jobs.

---

## Task 4: Validate `tailor`

- [ ] **Step 1: Description-only job triggers skill extraction**

Run:
```bash
job-applicator tailor --resume /tmp/resume_smoke.txt --job-title "Backend Engineer" --company "Smoke" --description "we need a react native engineer with python experience" --min-score 0.1 --yes --json
```

Expected: JSON tailored résumé, match score printed, no crash.

- [ ] **Step 2: Explicit requirements bypass extraction**

Run with `--requirements "Python,Docker"`.

Expected: tailored résumé, no extraction.

---

## Task 5: Validate `generate-cover-letter`

- [ ] **Step 1: Basic CLI path**

Run:
```bash
job-applicator generate-cover-letter --resume /tmp/resume_smoke.txt --job-title "Backend Engineer" --company "Smoke" --description "python and docker required" --json
```

Expected: JSON cover letter, sign-off verified.

---

## Task 6: Validate `apply`, `doctor`, `status`, `tui`

- [ ] **Step 1: `doctor`**

Run:
```bash
job-applicator doctor
```

Expected: clean health report.

- [ ] **Step 2: `status`**

After `match`/`batch`, run:
```bash
job-applicator status
```

Expected: shows saved jobs/matches.

- [ ] **Step 3: `apply` dry-run validation**

Run:
```bash
job-applicator apply --from <id> --validate --no-cover-letter
```

Expected: if no stored job, clear message; otherwise dry-run reaches Submit.

---

## Task 7: Fix inconsistencies

- [ ] Document each bug/UX issue with the exact command.
- [ ] Fix in source, add/update tests.
- [ ] Re-run the failing CLI command to confirm.

---

## Task 8: Final verification

- [ ] `pytest -m unit -q`
- [ ] `pytest -m integration -q`
- [ ] `ruff check src/ tests/`
- [ ] `ruff format --check src/ tests/`
- [ ] `mypy src/`
