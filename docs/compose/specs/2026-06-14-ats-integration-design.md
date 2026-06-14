# ATS Integration Design Spec

## [S1] Problem

The ATS checker is a standalone command (`ats-check`), but users don't know their resume has ATS issues until they explicitly check. Issues discovered after tailoring or applying waste LLM calls and time.

## [S2] Solution Overview

Integrate ATS checking into the existing `tailor`, `match`, and `apply` workflows:
1. **Pre-flight guard**: Soft warnings before operations proceed
2. **Post-tailor verification**: Score comparison and regression detection after tailoring

## [S3] Pre-flight ATS Guard

**Commands affected:** `tailor`, `match`, `apply`

**Behavior:**
- Run `ATSChecker.check(resume_data)` after loading resume
- If `is_compatible` is False (score < 60%): show yellow warning panel
- Show top 3 warnings (most impactful)
- Print tip to run full `ats-check` command
- **Continue execution** (soft guard, no blocking)

**Output format:**
```
⚠ ATS Compatibility: 43% (Not Compatible)
  ! Missing 'Experience' section — ATS expects standard headers
  ! Resume text too short (41 chars)
  ! No phone number found
  Tip: Run 'job-applicator ats-check --resume resume.pdf' for full report
```

**No output when score >= 60%** (silent pass).

## [S4] Post-tailor ATS Verification

**Commands affected:** `tailor` (after each tailoring attempt)

**Behavior:**
1. Before tailoring: record `original_score` from pre-flight check
2. After tailoring: create synthetic `ResumeData` from `tailored.tailored_text`
3. Run `ATSChecker.check()` on tailored text
4. Show comparison panel with:
   - Before/after score with arrow
   - Check mark if improved or stable
   - Warning if score regressed
   - List of new issues introduced by tailoring

**Output when improved/stable:**
```
ATS Compatibility (before → after): 86% → 100% ✓
  ✓ All checks passing after tailoring
```

**Output when regressed:**
```
⚠ ATS Compatibility (before → after): 86% → 71%
  ! New issue: 'Skills' section header removed during tailoring
  ! New issue: Text length reduced below minimum
```

## [S5] Implementation Details

- `ATSChecker` is reused (no duplication)
- Pre-flight check runs on already-loaded `ResumeData` (no extra file I/O)
- Post-tailor check parses `tailored_text` string directly (no temp files)
- Score comparison is side-effect only — doesn't block or modify the tailoring result
- The `_run_ats_check()` helper function extracts `ResumeData` from tailored text by creating a temporary instance

## [S6] Files to Modify

| File | Change |
|------|--------|
| `cli.py` | Add `_run_ats_preflight()` helper, call in `tailor`, `match`, `apply` |
| `cli.py` | Add `_run_ats_post_tailor()` helper, call after tailoring in `tailor` |
| `tests/unit/test_ats_checker.py` | Add tests for pre-flight and post-tailor helpers |

## [S7] Testing Strategy

- Unit tests for `_run_ats_preflight()` — verifies warning printed when score < 60%
- Unit tests for `_run_ats_post_tailor()` — verifies score comparison and regression detection
- Integration test: tailor command with real resume shows ATS comparison
- Edge case: tailored text that removes sections triggers regression warning
