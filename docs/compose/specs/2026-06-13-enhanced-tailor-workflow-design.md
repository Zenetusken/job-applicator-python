# Enhanced Resume Tailor Workflow Design

> [!NOTE]
> This document may not reflect the current implementation.
> See the final report for up-to-date state:
> [Final Report](../reports/tier1-tier2-gap-fixes-and-batch-mode.md)

## [S1] Problem

The current accept/retry workflow in `cli.py:tailor` and `scripts/tailor_cgi.py` works but lacks:
- Visibility into what changed between original and tailored resume
- Ability to review and revert to previous attempts
- Fine-grained control over specific resume sections
- Tone adaptation based on job posting type
- Production-grade error handling and test coverage

## [S2] Solution Overview

Add four capabilities to the existing A/R/I/Q loop:
1. **Diff View** — color-coded diff after each attempt
2. **Version History** — browse and select from all attempts
3. **Section-Level Editing** — target specific resume sections
4. **Auto Tone Detection** — adjust vocabulary based on job posting analysis

Plus production hardening: error handling, input validation, edge cases, unit tests.

## [S3] Diff View

After each tailoring attempt, show a unified diff between the original resume text and the tailored version.

**Implementation:**
- Use Python's `difflib.unified_diff` to compute diff lines
- Render with Rich: green for additions (`+`), red for removals (`-`), dim for context
- Show automatically after the preview panel (collapsed to first 30 lines, with "show more" option)
- Add `[D] Diff` option to action menu that shows full diff
- Store original text in the `TailorSession` for comparison

**Files changed:** `cli.py` (tailor command), `resume_tailor.py` (no changes needed — original text already available)

## [S4] Version History

Store all tailoring attempts in memory during the session. Let user browse, view, and select previous attempts.

**Implementation:**
- Create `TailorSession` dataclass holding `list[TailoredResume]` and current attempt index
- Add `[V] History` option to action menu
- History view: Rich table with columns (Attempt #, Timestamp, Match Score, User Instructions, Preview snippet)
- User can select an attempt by number to view full preview or accept it
- On accept, save the selected version (not necessarily the latest)

**New model:**
```python
@dataclass
class TailorSession:
    attempts: list[TailoredResume]
    current_index: int
    original_resume: ResumeData
    job: JobListing
```

**Files changed:** `cli.py`, `models.py` (add TailorSession), `resume_tailor.py` (no changes)

## [S5] Section-Level Editing

Parse the tailored resume into logical sections and let user target specific ones for refinement.

**Implementation:**
- Parse tailored text into sections by detecting headers (ALL CAPS lines, lines ending with `:`, common section names: Summary, Experience, Skills, Education, Certifications, Projects)
- Add `[S] Section` option to action menu
- Show numbered list of detected sections with line counts
- User picks a section, then gives instructions scoped to that section
- Refine prompt is constructed with only that section's content, plus instruction to keep other sections unchanged
- If section parsing fails (unusual format), fall back to full-resume refinement with a warning

**Section detection regex patterns:**
- `^(SUMMARY|EXPERIENCE|SKILLS|EDUCATION|CERTIFICATIONS|PROJECTS|OBJECTIVE|PROFILE|WORK EXPERIENCE|EMPLOYMENT|QUALIFICATIONS|ACHIEVEMENTS)\s*$` (case-insensitive)
- Lines that are ALL CAPS and < 50 chars
- Lines matching `^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:$`

**Files changed:** `cli.py`, new helper in `resume_tailor.py` (`parse_sections()`)

## [S6] Auto Tone Detection

Analyze the job posting to detect tone and inject appropriate vocabulary into the tailoring prompt.

**Implementation:**
- Create `ToneDetector` class with keyword-based analysis
- Four tone profiles:
  - **Corporate**: keywords like "compliance", "governance", "stakeholder", "enterprise", "SLA", "KPI"
  - **Startup**: keywords like "fast-paced", "wear many hats", "agile", "scrappy", "self-starter", "equity"
  - **Technical**: keywords like "architecture", "system design", "scalability", "CI/CD", "microservices", specific tech stack
  - **Creative**: keywords like "brand", "storytelling", "design thinking", "user experience", "visual", "content"
- Analyze job title + description + requirements
- Return primary tone + vocabulary suggestions
- Inject into tailoring prompt as a tone directive

**Tone prompt injection example:**
```
TONE: Corporate/Enterprise
- Use formal language: "leveraged", "orchestrated", "facilitated"
- Emphasize: compliance, process improvement, stakeholder management
- Avoid: casual language, slang, overly technical jargon
```

**New file:** `src/job_applicator/documents/tone_detector.py`

**Files changed:** `resume_tailor.py` (integrate tone into prompt), `cli.py` (display detected tone)

## [S7] Production Hardening

**Error handling:**
- Wrap all LLM calls in try/except with user-friendly messages
- On LLM timeout: "LLM is taking too long. [R] Retry or [Q] Quit?"
- On LLM error: "LLM returned an error: <message>. [R] Retry or [Q] Quit?"
- Max 10 retries with warning at attempt 8+

**Input validation:**
- User instructions: strip whitespace, reject empty on `[I]` choice (already done)
- Section selection: validate number is in range, re-prompt on invalid
- File paths: validate resume exists before starting

**Edge cases:**
- Empty resume text: abort with clear message
- Missing job description: warn but allow (tailoring will be generic)
- Very long resumes (>10 pages): truncate with warning
- LLM returns empty/whitespace: retry automatically

**Tests:**
- Unit test `TailorSession` (add attempt, get current, browse history)
- Unit test `parse_sections()` with various resume formats
- Unit test `ToneDetector` with each tone profile
- Unit test diff generation
- Integration test: mock LLM, run full accept/retry/input/quit cycle

## [S8] File Changes Summary

| File | Change |
|------|--------|
| `src/job_applicator/cli.py` | Enhance tailor command with diff, history, section editing, tone display |
| `src/job_applicator/documents/resume_tailor.py` | Add `parse_sections()`, integrate tone into prompt |
| `src/job_applicator/documents/tone_detector.py` | **NEW** — ToneDetector class |
| `src/job_applicator/models.py` | Add `TailorSession` model |
| `tests/unit/test_resume_tailor.py` | Add tests for parse_sections, tone integration |
| `tests/unit/test_tone_detector.py` | **NEW** — ToneDetector tests |
| `tests/unit/test_tailor_session.py` | **NEW** — TailorSession + workflow tests |
| `scripts/tailor_cgi.py` | Update to use enhanced workflow |

## [S9] Out of Scope

- Persistent version history across sessions (would need file/DB storage)
- Side-by-side terminal comparison (diff view is sufficient)
- LLM-based tone detection (keyword approach is fast, deterministic, no extra LLM call)
- Resume section editing via visual editor (terminal-only for now)
