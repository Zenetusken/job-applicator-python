# Post-Tailor Cover Letter Integration Design

## [S1] Problem

Cover letter generation is completely separate from resume tailoring. The `generate-cover-letter` command is standalone, and the `tailor` command doesn't produce cover letters. Users must run two separate commands with the same job info, with no shared tone or style consistency.

Additionally, there's no way to point to an existing cover letter as a style template (like the CV's `--style-guide`), so the LLM generates generic letters instead of mimicking the user's writing style.

## [S2] Solution

After the user accepts a tailored resume in the `tailor` command, optionally generate a matching cover letter. Reuse the same job data, resume, style guide, and tone profile. Show it in an accept/retry/input/diff/history workflow. The `generate-cover-letter` standalone command is preserved as-is (not deprecated) for users who want a one-off letter without the tailor flow.

## [S3] Flow

```
tailor resume → accept → save resume → "Generate cover letter? [Y/N]"
  → Y → generate cover letter (same tone, style, job, tailored resume text)
       → preview + diff + metadata
       → [A] Accept / [R] Retry / [I] Input / [D] Diff / [V] History / [Q] Skip
       → save cover letter alongside resume
       → update resume meta.json with cover_letter_path
  → N → done (resume only, no cover_letter_path in meta.json)
```

After resume acceptance and file save, the CLI prompts:
```
Generate a matching cover letter for Technical Support Specialist at CGI? [Y/N]:
```

If Y, enters a cover letter sub-loop. On accept, saves the cover letter and updates the resume's meta.json. On skip (Q), exits cleanly — resume is already saved, `cover_letter_path` is omitted from meta.json.

## [S4] Cover Letter Style Templating

Same as the CV workflow: the user can point to an existing cover letter as a style template via `--style-guide`. The style analyzer extracts writing patterns (tone, vocabulary, structure, key phrases) and injects them into the cover letter generation prompt.

The `CoverLetterGenerator._build_prompt()` already handles `StyleGuide` injection via `StyleAnalyzer.format_style_for_prompt()`. No changes needed to the style loading mechanism — the existing `--style-guide` flag works for cover letters too.

If no style guide is provided, the cover letter is generated with default professional tone.

## [S5] Cover Letter Prompt Enhancement

The existing `_build_prompt()` builds a basic prompt with job info, user profile, and skills. Enhance it to accept two new optional parameters:

**`generate()` signature change:**
```python
async def generate(
    self,
    job: JobListing,
    user: UserProfile,
    resume: ResumeData,
    style_guide: StyleGuide | None = None,
    tone_section: str = "",
    tailored_resume_text: str = "",
) -> str:
```

**`_build_prompt()` changes:**
- Accept `tone_section: str = ""` — injected after skills, before generation instruction
- Accept `tailored_resume_text: str = ""` — if provided, use this as the primary content source instead of `resume.raw_text`. The tailored resume contains optimized skills and experience, so `_build_prompt()` extracts skills/summary from it rather than the original resume.
- Add consistency directive: "The cover letter should complement, not repeat, the tailored resume. Reference specific achievements and skills without copying bullet points."

**Caller responsibility:** The CLI passes `tone_section=tone_detector.format_for_prompt(tone_profile)` and `tailored_resume_text=result.tailored_text` when generating post-tailor cover letters. Existing callers (standalone `generate-cover-letter`, `apply`) pass no `tone_section` or `tailored_resume_text`, preserving current behavior.

## [S6] File Organization

When both resume and cover letter are saved:

```
output/
├── tailored_CGI_TechSupport_20260613_120000.txt          # tailored resume
├── tailored_CGI_TechSupport_20260613_120000.meta.json    # resume metadata + cover_letter_path
├── cover_letter_CGI_TechSupport_20260613_120500.txt      # cover letter
└── cover_letter_CGI_TechSupport_20260613_120500.meta.json # cover letter metadata
```

**Resume meta.json:** The CLI writes the resume meta.json AFTER the cover letter flow completes (not before). If the user accepted a cover letter, `cover_letter_path` is included. If the user skipped, `cover_letter_path` is omitted. This avoids the problem of writing meta.json twice.

**Cover letter meta.json:** A new `CoverLetterResult` model is serialized. `output_path` is set after the cover letter file is written, before meta.json serialization (same pattern as `TailoredResume` in `cli.py:627`).

```python
class CoverLetterResult(BaseModel):
    job_title: str
    job_company: str
    job_url: str = ""
    cover_letter_text: str
    user_modifications: str = ""
    attempt: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    output_path: str = ""

    model_config = {"extra": "forbid"}
```

This is simpler than `TailoredResume` — no `match_score`, `matched_skills`, etc. since those are resume-specific concepts.

## [S7] Cover Letter Session

Create a new `CoverLetterSession` class (not reuse `TailorSession`) to avoid the model mismatch:

```python
class CoverLetterSession:
    """Tracks cover letter generation attempts."""

    def __init__(self, job_title: str, job_company: str) -> None:
        self.job_title = job_title
        self.job_company = job_company
        self.attempts: list[CoverLetterResult] = []
        self.current_index: int = -1

    def add_attempt(self, result: CoverLetterResult) -> None:
        self.attempts.append(result)
        self.current_index = len(self.attempts) - 1

    @property
    def current(self) -> CoverLetterResult:
        if not self.attempts or self.current_index < 0:
            raise IndexError("No attempts in session")
        return self.attempts[self.current_index]

    def select(self, index: int) -> None:
        if index < 0 or index >= len(self.attempts):
            raise IndexError(f"Index {index} out of range")
        self.current_index = index
```

**Diff behavior:** `_render_diff` compares each attempt to the first attempt's text (not empty string). After the first attempt, `session.attempts[0].cover_letter_text` serves as the baseline for diffs.

## [S8] Error Handling

Same pattern as resume tailoring:
- LLM failure on initial generation: catch, print error, offer `[R] Retry / [Q] Skip`
- LLM failure on refinement: catch, print error, offer `[R] Retry / [Q] Skip`
- Max retries: 10 (warning at 8)

## [S9] CLI Menu for Cover Letter Loop

The cover letter loop has these options (NO `[S] Section` — cover letters are 3-4 paragraphs without parseable section headers):

```
[A] Accept    Save this cover letter
[R] Retry     Regenerate with same instructions
[I] Input     Give custom instructions to refine
[D] Diff      Show changes from first attempt
[V] History   Browse previous attempts
[Q] Skip      Discard and exit (resume already saved)
```

## [S10] Changes Summary

| File | Change |
|------|--------|
| `src/job_applicator/cli.py` | Add post-tailor cover letter prompt + workflow loop after resume acceptance |
| `src/job_applicator/documents/cover_letter.py` | Add `tone_section` and `tailored_resume_text` params to `generate()` and `_build_prompt()` |
| `src/job_applicator/models.py` | Add `CoverLetterResult` model and `CoverLetterSession` class |
| `src/job_applicator/cli.py` | Defer resume meta.json write until after cover letter flow |
| `tests/unit/test_documents.py` | Test tone injection and tailored resume reference in cover letter prompt |
| `tests/unit/test_models.py` | Test `CoverLetterSession` (add, current, select, empty) |

## [S11] Out of Scope

- Deprecating `generate-cover-letter` standalone command (kept as-is)
- Bulk cover letter generation for multiple jobs (existing `apply` command handles this)
- Persistent cover letter session across CLI runs
- Side-by-side resume + cover letter preview
- Cover letter section editing (cover letters lack parseable sections)
