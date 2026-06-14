# Cover Letter Integration Implementation Plan

> [!NOTE]
> This document may not reflect the current implementation.
> See the final report for up-to-date state:
> [Final Report](../reports/tier1-tier2-gap-fixes-and-batch-mode.md)

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate cover letter generation as a post-tailor step in the resume workflow, with shared tone, full accept/retry/input/diff/history workflow, and co-located file output.

**Architecture:** Add `CoverLetterResult` model and `CoverLetterSession` to `models.py`. Extend `CoverLetterGenerator.generate()` with tone and tailored resume params. Modify the `tailor` CLI command's accept handler to offer cover letter generation after resume save. The cover letter sub-loop mirrors the resume loop but without `[S] Section` (cover letters lack parseable sections).

**Tech Stack:** Python 3.12+, Pydantic (models), Rich (CLI), litellm (LLM), difflib (diff)

---

### Task 1: CoverLetterResult Model + CoverLetterSession

**Covers:** [S6, S7]

**Files:**
- Modify: `src/job_applicator/models.py`
- Test: `tests/unit/test_models.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_models.py`:

```python
from job_applicator.models import CoverLetterResult, CoverLetterSession


class TestCoverLetterResult:
    def test_model_creation(self):
        result = CoverLetterResult(
            job_title="Developer",
            job_company="TechCo",
            cover_letter_text="Dear Hiring Manager...",
        )
        assert result.attempt == 1
        assert result.user_modifications == ""
        assert result.output_path == ""

    def test_model_serialization(self):
        result = CoverLetterResult(
            job_title="Dev",
            job_company="Co",
            cover_letter_text="Letter text",
        )
        data = result.model_dump()
        assert "cover_letter_text" in data
        assert "created_at" in data


class TestCoverLetterSession:
    def test_session_creation(self):
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        assert session.attempts == []
        assert session.current_index == -1

    def test_add_attempt(self):
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        result = CoverLetterResult(
            job_title="Dev",
            job_company="Co",
            cover_letter_text="Letter v1",
        )
        session.add_attempt(result)
        assert len(session.attempts) == 1
        assert session.current.cover_letter_text == "Letter v1"

    def test_current_empty_raises(self):
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        with pytest.raises(IndexError):
            _ = session.current

    def test_select_attempt(self):
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        for i in range(3):
            session.add_attempt(CoverLetterResult(
                job_title="Dev",
                job_company="Co",
                cover_letter_text=f"Version {i}",
                attempt=i + 1,
            ))
        session.select(1)
        assert session.current.cover_letter_text == "Version 1"
        assert session.current_index == 1

    def test_select_out_of_range(self):
        session = CoverLetterSession(job_title="Dev", job_company="Co")
        with pytest.raises(IndexError):
            session.select(99)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_models.py::TestCoverLetterResult tests/unit/test_models.py::TestCoverLetterSession -v`
Expected: FAIL — models not defined

- [ ] **Step 3: Implement models**

Add to `src/job_applicator/models.py` (after `TailoredResume` class):

```python
class CoverLetterResult(BaseModel):
    """A generated cover letter with metadata."""

    job_title: str
    job_company: str
    job_url: str = ""
    cover_letter_text: str
    user_modifications: str = ""
    attempt: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    output_path: str = ""

    model_config = {"extra": "forbid"}


class CoverLetterSession:
    """Tracks cover letter generation attempts."""

    def __init__(self, job_title: str, job_company: str) -> None:
        self.job_title = job_title
        self.job_company = job_company
        self.attempts: list[CoverLetterResult] = []
        self.current_index: int = -1

    def add_attempt(self, result: CoverLetterResult) -> None:
        """Add a new attempt and set it as current."""
        self.attempts.append(result)
        self.current_index = len(self.attempts) - 1

    @property
    def current(self) -> CoverLetterResult:
        """Get the currently selected attempt."""
        if not self.attempts or self.current_index < 0:
            raise IndexError("No attempts in session")
        return self.attempts[self.current_index]

    def select(self, index: int) -> None:
        """Select a previous attempt by index."""
        if index < 0 or index >= len(self.attempts):
            raise IndexError(f"Index {index} out of range (0-{len(self.attempts) - 1})")
        self.current_index = index
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_models.py::TestCoverLetterResult tests/unit/test_models.py::TestCoverLetterSession -v`
Expected: 7 passed

- [ ] **Step 5: Run full test suite**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short`
Expected: 80 passed (73 + 7)

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/models.py tests/unit/test_models.py
git commit -m "feat: add CoverLetterResult model and CoverLetterSession"
```

---

### Task 2: Extend CoverLetterGenerator with Tone + Tailored Resume

**Covers:** [S4, S5]

**Files:**
- Modify: `src/job_applicator/documents/cover_letter.py`
- Test: `tests/unit/test_documents.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_documents.py`:

```python
class TestCoverLetterWithTone:
    @pytest.mark.asyncio
    async def test_generate_includes_tone_section(self, llm_config):
        from job_applicator.documents.cover_letter import CoverLetterGenerator

        generator = CoverLetterGenerator(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Dear Hiring Manager,\n\nCover letter text."

        job = JobListing(
            title="Dev", company="Co", url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="123")
        resume = ResumeData(raw_text="Resume text", skills=["Python"])

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await generator.generate(
                job, user, resume,
                tone_section="TONE: Corporate\n- Power words: leveraged",
            )

        call_args = mock_call.call_args
        assert "TONE: Corporate" in str(call_args)

    @pytest.mark.asyncio
    async def test_generate_uses_tailored_resume_text(self, llm_config):
        from job_applicator.documents.cover_letter import CoverLetterGenerator

        generator = CoverLetterGenerator(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Dear Hiring Manager,\n\nCover letter."

        job = JobListing(
            title="Dev", company="Co", url="https://example.com",
            board=JobBoard.INDEED,
        )
        user = UserProfile(first_name="John", last_name="Doe", email="j@e.com", phone="123")
        resume = ResumeData(raw_text="Original resume", skills=["Python"])
        tailored = "Tailored resume with optimized Python experience"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await generator.generate(
                job, user, resume,
                tailored_resume_text=tailored,
            )

        call_args = mock_call.call_args
        prompt = str(call_args)
        assert "Tailored resume with optimized" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_documents.py::TestCoverLetterWithTone -v`
Expected: FAIL — `tone_section` param doesn't exist

- [ ] **Step 3: Extend generate() and _build_prompt()**

In `src/job_applicator/documents/cover_letter.py`:

Modify `generate()` signature (line 221) to add new params:

```python
    @async_retry(max_attempts=2, base_delay=1.0, exceptions=(LLMError,))
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

Update the `_build_prompt` call (line 236):

```python
        user_message = self._build_prompt(job, user, resume, style_guide, tone_section, tailored_resume_text)
```

Modify `_build_prompt()` signature (line 291) and body:

```python
    def _build_prompt(
        self,
        job: JobListing,
        user: UserProfile,
        resume: ResumeData,
        style_guide: StyleGuide | None = None,
        tone_section: str = "",
        tailored_resume_text: str = "",
    ) -> str:
        """Build the prompt for cover letter generation."""
        # Use tailored resume text if provided, otherwise use original
        resume_content = tailored_resume_text if tailored_resume_text else resume.raw_text
        skills_source = tailored_resume_text if tailored_resume_text else ""

        parts = [
            "Write a cover letter for the following position:",
            "",
            f"Job Title: {job.title}",
            f"Company: {job.company}",
            f"Location: {job.location}",
        ]

        if job.description:
            parts.extend(["", "Job Description:", job.description])

        parts.extend(
            [
                "",
                "Applicant Profile:",
                f"Name: {user.first_name} {user.last_name}",
                f"Email: {user.email}",
            ]
        )

        if resume_content:
            parts.extend(["", "Resume Content:", resume_content[:3000]])

        if resume.skills:
            parts.extend(["", f"Key Skills: {', '.join(resume.skills)}"])

        # Add tone guidance if provided
        if tone_section:
            parts.extend(["", tone_section])

        # Add style guide if provided
        if style_guide:
            from job_applicator.documents.style_analyzer import StyleAnalyzer

            analyzer = StyleAnalyzer(self._config)
            style_section = analyzer.format_style_for_prompt(style_guide)
            parts.extend(["", style_section])

        if tailored_resume_text:
            parts.extend([
                "",
                "IMPORTANT: The cover letter should complement, not repeat, "
                "the tailored resume. Reference specific achievements and skills "
                "without copying bullet points. Match the tone and emphasis of "
                "the resume.",
            ])

        parts.extend(["", "Generate a professional cover letter with key points highlighted."])

        return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/test_documents.py::TestCoverLetterWithTone -v`
Expected: 2 passed

- [ ] **Step 5: Run full test suite + lint**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short && ruff check src/job_applicator/documents/cover_letter.py`
Expected: 82 passed, lint clean

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/documents/cover_letter.py tests/unit/test_documents.py
git commit -m "feat: extend CoverLetterGenerator with tone_section and tailored_resume_text"
```

---

### Task 3: Post-Tailor Cover Letter Workflow in CLI

**Covers:** [S2, S3, S6, S8, S9]

**Files:**
- Modify: `src/job_applicator/cli.py`

- [ ] **Step 1: Defer resume meta.json write**

In the `[A] Accept` handler of the `tailor` command, change the flow so the resume text file is written immediately, but the meta.json is deferred. Replace the current accept block (around line 614-635) with:

```python
            if choice == "A":
                from datetime import datetime as dt

                output_dir = Path(settings.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

                safe_company = job.company.replace(" ", "_").replace("/", "_")
                safe_title = job.title.replace(" ", "_").replace("/", "_")
                timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
                output_path = output_dir / filename

                output_path.write_text(result.tailored_text, encoding="utf-8")
                result.output_path = str(output_path)

                console.print(f"\n[green]Tailored resume saved: {output_path}[/green]")
                console.print(f"[dim]Attempt #{attempt} | Score: {result.match_score:.0%}[/dim]")

                # Offer cover letter generation
                cover_letter_path = None
                cl_choice = (
                    console.input(
                        f"\n[bold cyan]Generate a matching cover letter "
                        f"for {job.title} at {job.company}? (Y/N): [/bold cyan]"
                    )
                    .strip()
                    .upper()
                )

                if cl_choice == "Y":
                    cover_letter_path = await _cover_letter_workflow(
                        console, settings, job, resume_data, style,
                        tone_profile, result.tailored_text,
                    )

                # Write resume meta.json (with or without cover_letter_path)
                if cover_letter_path:
                    result.cover_letter_path = str(cover_letter_path)
                meta_path = output_path.with_suffix(".meta.json")
                meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                console.print(f"[green]Metadata saved: {meta_path}[/green]")

                break
```

- [ ] **Step 2: Add the cover letter workflow function**

Add a new async function `_cover_letter_workflow()` before the `tailor` command definition. This function contains the full A/R/I/D/V/Q loop for cover letters:

```python
async def _cover_letter_workflow(
    console: Console,
    settings: AppSettings,
    job: "JobListing",
    resume_data: "ResumeData",
    style: "StyleGuide | None",
    tone_profile: "ToneProfile | None",
    tailored_resume_text: str,
) -> Path | None:
    """Generate and save a cover letter with accept/retry workflow.

    Returns the Path to the saved cover letter, or None if skipped.
    """
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.models import CoverLetterResult, CoverLetterSession

    generator = CoverLetterGenerator(settings.llm)
    tone_section = ""
    if tone_profile:
        from job_applicator.documents.tone_detector import ToneDetector
        tone_section = ToneDetector().format_for_prompt(tone_profile)

    session = CoverLetterSession(job_title=job.title, job_company=job.company)
    attempt = 0
    user_instructions = ""

    try:
        with console.status("Generating cover letter..."):
            letter = await generator.generate(
                job, _load_user_profile(settings), resume_data,
                style_guide=style,
                tone_section=tone_section,
                tailored_resume_text=tailored_resume_text,
            )
        result = CoverLetterResult(
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            cover_letter_text=letter,
            attempt=1,
        )
        session.add_attempt(result)
    except Exception as exc:
        console.print(f"[red]LLM error: {exc}[/red]")
        retry = console.input("[bold cyan][R] Retry or [Q] Skip? [/bold cyan]").strip().upper()
        if retry == "R":
            # Retry once
            try:
                with console.status("Generating cover letter..."):
                    letter = await generator.generate(
                        job, _load_user_profile(settings), resume_data,
                        style_guide=style,
                        tone_section=tone_section,
                        tailored_resume_text=tailored_resume_text,
                    )
                result = CoverLetterResult(
                    job_title=job.title, job_company=job.company,
                    job_url=str(job.url), cover_letter_text=letter, attempt=1,
                )
                session.add_attempt(result)
            except Exception:
                console.print("[red]Cover letter generation failed. Skipping.[/red]")
                return None
        else:
            return None

    while True:
        attempt += 1
        if attempt > 10:
            console.print("[red]Maximum retry limit (10) reached.[/red]")
            break
        if attempt >= 8:
            console.print("[yellow]Warning: approaching retry limit (10 max).[/yellow]")

        result = session.current
        console.print(f"\n[bold blue]--- Cover Letter Attempt #{attempt} ---[/bold blue]")

        console.print("\n[bold]Cover Letter Preview:[/bold]\n")
        console.print(Panel(result.cover_letter_text, title="Cover Letter", border_style="green"))

        # Diff from first attempt
        if len(session.attempts) > 1:
            from job_applicator.utils.diff import render_diff
            render_diff(console, session.attempts[0].cover_letter_text, result.cover_letter_text, max_lines=30)

        console.print("\n[bold]What would you like to do?[/bold]")
        action_table = Table(show_header=False, box=None)
        action_table.add_column("Option", style="cyan bold")
        action_table.add_column("Description")
        action_table.add_row("[A] Accept", "Save this cover letter")
        action_table.add_row("[R] Retry", "Regenerate with same instructions")
        action_table.add_row("[I] Input", "Give custom instructions to refine")
        action_table.add_row("[D] Diff", "Show full diff from first attempt")
        action_table.add_row("[V] History", "Browse previous attempts")
        action_table.add_row("[Q] Skip", "Discard cover letter (resume already saved)")
        console.print(action_table)

        choice = console.input("\n[bold cyan]Your choice (A/R/I/D/V/Q): [/bold cyan]").strip().upper()

        if choice == "A":
            from datetime import datetime as dt

            output_dir = Path(settings.output_dir)
            safe_company = job.company.replace(" ", "_").replace("/", "_")
            safe_title = job.title.replace(" ", "_").replace("/", "_")
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            cl_filename = f"cover_letter_{safe_company}_{safe_title}_{timestamp}.txt"
            cl_path = output_dir / cl_filename

            cl_path.write_text(result.cover_letter_text, encoding="utf-8")
            result.output_path = str(cl_path)

            cl_meta_path = cl_path.with_suffix(".meta.json")
            cl_meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

            console.print(f"\n[green]Cover letter saved: {cl_path}[/green]")
            return cl_path

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            user_instructions = ""
            try:
                with console.status("Generating cover letter..."):
                    letter = await generator.generate(
                        job, _load_user_profile(settings), resume_data,
                        style_guide=style, tone_section=tone_section,
                        tailored_resume_text=tailored_resume_text,
                    )
                new_result = CoverLetterResult(
                    job_title=job.title, job_company=job.company,
                    job_url=str(job.url), cover_letter_text=letter,
                    attempt=attempt + 1,
                )
                session.add_attempt(new_result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
                console.input("[bold cyan][R] Retry or [Q] Skip? [/bold cyan]")
            continue

        elif choice == "I":
            user_instructions = console.input(
                "\n[bold]Enter instructions (e.g., 'emphasize customer service'): [/bold]"
            ).strip()
            if not user_instructions:
                console.print("[yellow]No instructions provided, retrying.[/yellow]")
            try:
                with console.status("Refining cover letter..."):
                    refine_prompt = (
                        f"User wants changes to this cover letter.\n\n"
                        f"Job: {job.title} at {job.company}\n\n"
                        f"Current cover letter:\n{result.cover_letter_text}\n\n"
                        f"User feedback: {user_instructions}\n\n"
                        f"Return the complete updated cover letter."
                    )
                    from litellm import acompletion
                    model = f"openai/{settings.llm.model}" if settings.llm.api_base else settings.llm.model
                    response = await acompletion(
                        model=model, api_base=settings.llm.api_base, api_key=settings.llm.api_key,
                        messages=[{"role": "user", "content": refine_prompt}],
                        max_tokens=settings.llm.max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    from job_applicator.documents.cover_letter import strip_thinking_process
                    refined = strip_thinking_process(response.choices[0].message.content)
                new_result = CoverLetterResult(
                    job_title=job.title, job_company=job.company,
                    job_url=str(job.url), cover_letter_text=refined,
                    user_modifications=user_instructions, attempt=attempt + 1,
                )
                session.add_attempt(new_result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
            continue

        elif choice == "D":
            from job_applicator.utils.diff import render_diff
            if len(session.attempts) > 1:
                render_diff(console, session.attempts[0].cover_letter_text, result.cover_letter_text, max_lines=0)
            else:
                console.print("[yellow]Only one attempt so far.[/yellow]")
            continue

        elif choice == "V":
            if len(session.attempts) < 2:
                console.print("[yellow]No previous attempts yet.[/yellow]")
                continue
            hist_table = Table(title="Cover Letter History")
            hist_table.add_column("#", style="dim")
            hist_table.add_column("Attempt")
            hist_table.add_column("Preview", style="dim")
            for i, att in enumerate(session.attempts):
                preview = att.cover_letter_text[:60].replace("\n", " ")
                marker = "→" if i == session.current_index else " "
                hist_table.add_row(marker, str(att.attempt), preview + "...")
            console.print(hist_table)
            sel = console.input("\n[bold cyan]Select attempt # (or Enter to go back): [/bold cyan]").strip()
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(session.attempts):
                    session.select(idx)
                    console.print(f"[green]Switched to attempt #{session.current.attempt}[/green]")
                else:
                    console.print("[red]Invalid attempt number.[/red]")
            continue

        elif choice == "Q":
            console.print("[yellow]Cover letter skipped. Resume already saved.[/yellow]")
            return None

        else:
            console.print("[red]Invalid choice. Please enter A, R, I, D, V, or Q.[/red]")

    return None
```

- [ ] **Step 3: Add tone_profile to tailor command scope**

In the `tailor` command, the `tone_profile` variable is already created (line ~459). Ensure it's accessible in the accept handler scope. Read the current tailor command to verify `tone_profile` is in scope at the accept handler.

- [ ] **Step 4: Add cover_letter_path field to TailoredResume**

In `src/job_applicator/models.py`, add to `TailoredResume`:

```python
    cover_letter_path: str = Field(default="", description="Path to generated cover letter, if any")
```

- [ ] **Step 5: Run tests + lint**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short && ruff check src/job_applicator/cli.py src/job_applicator/models.py`
Expected: All tests pass, lint clean

- [ ] **Step 6: Commit**

```bash
git add src/job_applicator/cli.py src/job_applicator/models.py
git commit -m "feat: add post-tailor cover letter workflow with accept/retry/diff/history"
```

---

### Task 4: Update tailor_cgi.py Script

**Covers:** [S2, S3]

**Files:**
- Modify: `scripts/tailor_cgi.py`

- [ ] **Step 1: Add cover letter prompt to script**

In `scripts/tailor_cgi.py`, after the resume accept handler saves the resume file (around the `break` after saving), add the same cover letter prompt and workflow. Read the current script first to understand the structure.

- [ ] **Step 2: Commit**

```bash
git add scripts/tailor_cgi.py
git commit -m "feat: add cover letter workflow to tailor_cgi.py"
```

---

### Task 5: Tests and Documentation

**Covers:** [S8, S10]

**Files:**
- Modify: `tests/unit/test_resume_tailor.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add integration test for cover letter session workflow**

Add to `tests/unit/test_resume_tailor.py`:

```python
class TestCoverLetterWorkflow:
    def test_cover_letter_session_workflow(self):
        from job_applicator.models import CoverLetterResult, CoverLetterSession

        session = CoverLetterSession(job_title="Dev", job_company="Co")

        for i in range(3):
            session.add_attempt(CoverLetterResult(
                job_title="Dev",
                job_company="Co",
                cover_letter_text=f"Letter version {i + 1}",
                attempt=i + 1,
            ))

        assert len(session.attempts) == 3
        assert session.current.cover_letter_text == "Letter version 3"

        # Diff baseline should be first attempt
        assert session.attempts[0].cover_letter_text == "Letter version 1"

        session.select(0)
        assert session.current.cover_letter_text == "Letter version 1"
```

- [ ] **Step 2: Run full test suite + lint + format + typecheck**

Run: `cd /home/drei/project/job-applicator-python && .venv/bin/python -m pytest tests/unit/ -v --tb=short && ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: All tests pass, all checks clean

- [ ] **Step 3: Update README.md**

Add a section about the post-tailor cover letter workflow.

- [ ] **Step 4: Update AGENTS.md**

Add gotchas about:
- Cover letter sub-loop has no `[S] Section` option (cover letters lack parseable sections)
- `CoverLetterResult` is simpler than `TailoredResume` (no match_score, matched_skills, etc.)
- Resume meta.json write is deferred until after cover letter flow completes

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_resume_tailor.py README.md AGENTS.md
git commit -m "feat: add cover letter integration tests and docs"
```
