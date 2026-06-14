# ATS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate ATS checking into tailor, match, and apply workflows as a pre-flight guard and post-tailor verification.

**Architecture:** Add helper functions in `cli.py` that reuse `ATSChecker` to run checks before and after operations. Pre-flight shows soft warnings; post-tailor shows score comparison.

**Tech Stack:** Python, Typer CLI, existing `ATSChecker` class

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/job_applicator/cli.py` | Add `_run_ats_preflight()` and `_run_ats_post_tailor()` helpers |
| `tests/unit/test_ats_checker.py` | Add tests for pre-flight and post-tailor helpers |

---

### Task 1: Add `_run_ats_preflight()` helper

**Covers:** [S3]

**Files:**
- Modify: `src/job_applicator/cli.py`
- Test: `tests/unit/test_ats_checker.py`

- [ ] **Step 1: Write the failing test**

```python
class TestATSPreflight:
    def test_preflight_warns_when_incompatible(self, capsys: object) -> None:
        from job_applicator.cli import _run_ats_preflight
        from job_applicator.models import ResumeData

        resume = ResumeData(
            raw_text="Bob\nbob@email.com\nstuff",
            name="Bob",
            email="bob@email.com",
            phone="",
        )
        _run_ats_preflight(resume)
        # Should not raise, just print warning

    def test_preflight_silent_when_compatible(self, capsys: object) -> None:
        from job_applicator.cli import _run_ats_preflight
        from job_applicator.models import ResumeData

        resume = ResumeData(
            raw_text="John Doe\njohn@example.com\n555-123-4567\nSummary\nExperienced developer.\nExperience\nSenior Dev at Corp (2020-Present)\n- Built stuff\nEducation\nBS CS (2016-2020)\nSkills\nPython, Java",
            name="John Doe",
            email="john@example.com",
            phone="555-123-4567",
            skills=["Python", "Java"],
        )
        _run_ats_preflight(resume)
        # Should not print anything
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_ats_checker.py::TestATSPreflight -v`
Expected: FAIL with "cannot import name '_run_ats_preflight'"

- [ ] **Step 3: Write minimal implementation**

Add to `cli.py` after `_resolve_ocr_mode()`:

```python
def _run_ats_preflight(resume: ResumeData) -> None:
    """Run ATS compatibility check and warn if issues found."""
    from job_applicator.documents.ats_checker import ATSChecker

    checker = ATSChecker()
    result = checker.check(resume)

    if result.is_compatible:
        return

    console.print(f"\n[yellow]⚠ ATS Compatibility: {result.score:.0%} (Not Compatible)[/yellow]")
    for warning in result.warnings[:3]:
        console.print(f"  [yellow]![/yellow] {warning}")
    console.print(
        f"  [dim]Tip: Run 'job-applicator ats-check --resume <path>' for full report[/dim]"
    )
    console.print()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_ats_checker.py::TestATSPreflight -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/job_applicator/cli.py tests/unit/test_ats_checker.py
git commit -m "feat: Add ATS pre-flight guard helper"
```

---

### Task 2: Add `_run_ats_post_tailor()` helper

**Covers:** [S4]

**Files:**
- Modify: `src/job_applicator/cli.py`
- Test: `tests/unit/test_ats_checker.py`

- [ ] **Step 1: Write the failing test**

```python
class TestATSPostTailor:
    def test_post_tailor_shows_improvement(self) -> None:
        from job_applicator.cli import _run_ats_post_tailor

        original_text = "Bob\nbob@email.com\nstuff"
        tailored_text = (
            "Bob\nbob@email.com\n555-123-4567\n"
            "Summary\nExperienced developer.\n"
            "Experience\nSenior Dev at Corp (2020-Present)\n"
            "Education\nBS CS (2016-2020)\n"
            "Skills\nPython, Java"
        )
        # Should not raise
        _run_ats_post_tailor(original_text, tailored_text)

    def test_post_tailor_detects_regression(self) -> None:
        from job_applicator.cli import _run_ats_post_tailor

        original_text = (
            "John\njohn@example.com\n555-123-4567\n"
            "Experience\nSenior Dev (2020-Present)\n"
            "Education\nBS CS (2016-2020)\n"
            "Skills\nPython"
        )
        tailored_text = "John\njohn@example.com\nTailored summary without sections."
        # Should not raise, just print warning
        _run_ats_post_tailor(original_text, tailored_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_ats_checker.py::TestATSPostTailor -v`
Expected: FAIL with "cannot import name '_run_ats_post_tailor'"

- [ ] **Step 3: Write minimal implementation**

Add to `cli.py` after `_run_ats_preflight()`:

```python
def _run_ats_post_tailor(original_text: str, tailored_text: str) -> None:
    """Compare ATS compatibility before and after tailoring."""
    from job_applicator.documents.ats_checker import ATSChecker
    from job_applicator.models import ResumeData

    checker = ATSChecker()

    original = ResumeData(raw_text=original_text)
    tailored = ResumeData(raw_text=tailored_text)

    original_result = checker.check(original)
    tailored_result = checker.check(tailored)

    before = original_result.score
    after = tailored_result.score

    if after >= before:
        console.print(
            f"\n[green]ATS Compatibility (before → after): "
            f"{before:.0%} → {after:.0%} ✓[/green]"
        )
        if after >= 0.6:
            console.print("  [green]✓ All checks passing after tailoring[/green]")
    else:
        console.print(
            f"\n[yellow]⚠ ATS Compatibility (before → after): "
            f"{before:.0%} → {after:.0%}[/yellow]"
        )
        # Find new issues
        original_checks = {c["name"]: c["passed"] for c in original_result.checks}
        for check in tailored_result.checks:
            if not check["passed"] and original_checks.get(check["name"], False):
                console.print(
                    f"  [yellow]![/yellow] New issue: {check['details']}"
                )
    console.print()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_ats_checker.py::TestATSPostTailor -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/job_applicator/cli.py tests/unit/test_ats_checker.py
git commit -m "feat: Add ATS post-tailor verification helper"
```

---

### Task 3: Integrate pre-flight guard into `tailor` command

**Covers:** [S3]

**Files:**
- Modify: `src/job_applicator/cli.py` (around line 1147)

- [ ] **Step 1: Add pre-flight call after resume loading**

In the `tailor` command's `_run()` async function, after `resume_data = loader.load(...)` and the console print, add:

```python
_run_ats_preflight(resume_data)
```

- [ ] **Step 2: Test manually**

Run: `.venv/bin/job-applicator tailor --resume /tmp/bad_resume.txt --job-title "Developer" --company "Corp"`
Expected: ATS warning shown before tailoring continues

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat: Integrate ATS pre-flight into tailor command"
```

---

### Task 4: Integrate pre-flight guard into `match` command

**Covers:** [S3]

**Files:**
- Modify: `src/job_applicator/cli.py` (around line 464)

- [ ] **Step 1: Add pre-flight call after resume loading**

In the `match` command's `_run()` async function, after `resume_data = loader.load(...)` and the console print, add:

```python
_run_ats_preflight(resume_data)
```

- [ ] **Step 2: Test manually**

Run: `.venv/bin/job-applicator match --resume /tmp/bad_resume.txt`
Expected: ATS warning shown before matching continues

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat: Integrate ATS pre-flight into match command"
```

---

### Task 5: Integrate pre-flight guard into `apply` command

**Covers:** [S3]

**Files:**
- Modify: `src/job_applicator/cli.py` (around line 246)

- [ ] **Step 1: Add pre-flight call after resume loading**

In the `apply` command's `_run()` async function, after `resume_data = loader.load(...)`, add:

```python
_run_ats_preflight(resume_data)
```

- [ ] **Step 2: Test manually**

Run: `.venv/bin/job-applicator apply --resume /tmp/bad_resume.txt --query "developer"`
Expected: ATS warning shown before applying continues

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat: Integrate ATS pre-flight into apply command"
```

---

### Task 6: Integrate post-tailor verification into `tailor` command

**Covers:** [S4]

**Files:**
- Modify: `src/job_applicator/cli.py` (after tailoring completes)

- [ ] **Step 1: Add post-tailor call after tailoring**

In the `tailor` command, after the tailoring result is created and before the user interaction loop, add:

```python
_run_ats_post_tailor(resume_data.raw_text, result.tailored_text)
```

- [ ] **Step 2: Test manually**

Run: `.venv/bin/job-applicator tailor --resume "/media/drei/KINGSTON/Andrei School/Other/Jobhunt/Andrei_Petrov_Resume.pdf" --job-title "IT Support" --company "TechCorp"`
Expected: ATS comparison shown after tailoring

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/job_applicator/cli.py
git commit -m "feat: Integrate ATS post-tailor verification"
```

---

### Task 7: Final verification and documentation update

**Covers:** [S3, S4, S7]

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Run full verification**

Run:
```bash
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/mypy src/job_applicator/ --ignore-missing-imports
.venv/bin/python -m pytest tests/unit/ -v
```
Expected: All pass

- [ ] **Step 2: Update AGENTS.md**

Add to the ATS Compatibility Checking section:
```
**Integrated checks:** ATS compatibility is automatically checked before `tailor`, `match`, and `apply` commands. Warnings shown if score < 60%. Post-tailor verification shows before/after comparison.
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: Update AGENTS.md with ATS integration info"
```
