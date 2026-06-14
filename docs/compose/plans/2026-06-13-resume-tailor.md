# Resume Tailor Implementation Plan

> [!NOTE]
> This document may not reflect the current implementation.
> See the final report for up-to-date state:
> [Final Report](../reports/tier1-tier2-gap-fixes-and-batch-mode.md)

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an LLM-powered resume tailoring engine that rewrites a resume for a specific job, with an interactive preview/accept/retry/input loop and full metadata storage.

**Architecture:** New `ResumeTailor` class in `documents/resume_tailor.py` uses the existing LLM infrastructure (litellm) to rewrite resume content targeted at a specific job listing. A new CLI `tailor` command provides an interactive loop: generate → preview → accept/retry/input. Accepted resumes are saved to `output/` as `.txt` with a `TailoredResume` metadata model.

**Tech Stack:** litellm, Pydantic, Rich (CLI), existing `LLMConfig` + `StyleAnalyzer` + `JobMatcher`

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `src/job_applicator/models.py` | Modify | Add `TailoredResume` metadata model |
| `src/job_applicator/documents/resume_tailor.py` | Create | LLM-powered resume tailoring engine |
| `src/job_applicator/cli.py` | Modify | Add `tailor` command with interactive loop |
| `tests/unit/test_resume_tailor.py` | Create | Unit tests for tailor logic |
| `scripts/tailor_cgi.py` | Create | Standalone script for CGI job tailoring |

---

### Task 1: Add TailoredResume metadata model

**Covers:** Full metadata visibility in system

**Files:**
- Modify: `src/job_applicator/models.py`

- [ ] **Step 1: Add TailoredResume model**

Add after `ApplicationResult` (line 113):

```python
class TailoredResume(BaseModel):
    """A resume tailored for a specific job, with full metadata."""

    original_path: str = Field(description="Path to original resume")
    tailored_text: str = Field(description="Full tailored resume text")
    job_title: str
    job_company: str
    job_url: str = ""
    match_score: float = Field(description="Combined match score at tailoring time")
    semantic_score: float = Field(description="Semantic similarity score")
    skill_score: float = Field(description="Skill coverage score")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    changes_summary: str = Field(description="LLM-generated summary of changes made")
    user_modifications: str = Field(default="", description="User's custom input that was applied")
    attempt: int = Field(default=1, description="Which attempt this is (1 = first)")
    created_at: datetime = Field(default_factory=datetime.now)
    output_path: str = Field(default="", description="Path where tailored resume was saved")

    model_config = {"extra": "forbid"}
```

- [ ] **Step 2: Run lint + format + typecheck**

```bash
ruff check src/job_applicator/models.py
ruff format --check src/job_applicator/models.py
mypy src/job_applicator/models.py --ignore-missing-imports
```

Expected: All pass.

---

### Task 2: Create ResumeTailor engine

**Covers:** LLM-powered resume rewriting with change tracking

**Files:**
- Create: `src/job_applicator/documents/resume_tailor.py`

- [ ] **Step 1: Create the resume_tailor.py module**

```python
"""LLM-powered resume tailoring — rewrites resume content for a specific job."""

from __future__ import annotations

from job_applicator.config import LLMConfig
from job_applicator.exceptions import LLMError
from job_applicator.models import (
    JobListing,
    ResumeData,
    StyleGuide,
    TailoredResume,
)
from job_applicator.utils.logging import get_logger
from job_applicator.utils.retry import async_retry

logger = get_logger("documents.resume_tailor")

TAILOR_SYSTEM_PROMPT = """You are an expert resume writer. Your job is to tailor \
a resume to better match a specific job posting.

Rules:
- Rewrite the resume to emphasize relevant skills and experience for this job
- Keep ALL factual information accurate — do NOT invent experience, skills, or jobs
- Reorder sections to put most relevant content first
- Adjust language to mirror the job posting's terminology where truthful
- Strengthen bullet points with action verbs and metrics where possible
- Keep the same overall structure (name, contact, skills, experience, education)
- Return the FULL tailored resume text, not just changes
- Also provide a brief summary of what you changed and why"""

TAILOR_PROMPT_TEMPLATE = """Tailor this resume for the following job:

Job Title: {job_title}
Company: {job_company}
Location: {job_location}
Description: {job_description}
Requirements: {requirements}

Current Resume:
---
{resume_text}

Current Skills: {skills}

{user_instructions}

Return the complete tailored resume text."""

CHANGES_PROMPT_TEMPLATE = """Given the original and tailored resume below, \
provide a concise bullet-point summary of what changed and why.

Original (first 500 chars):
---
{original_preview}
---

Tailored (first 500 chars):
---
{tailored_preview}
---

Return 3-5 bullet points describing the key changes."""


class ResumeTailor:
    """Tailor resumes for specific job listings using LLM."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @async_retry(max_attempts=2, base_delay=1.0, exceptions=(LLMError,))
    async def tailor(
        self,
        resume: ResumeData,
        job: JobListing,
        user_instructions: str = "",
        style_guide: StyleGuide | None = None,
    ) -> TailoredResume:
        """Tailor a resume for a specific job.

        Args:
            resume: Original parsed resume
            job: Target job listing
            user_instructions: Optional user guidance for tailoring
            style_guide: Optional style guide to apply

        Returns:
            TailoredResume with full metadata
        """
        from job_applicator.embeddings.matching import JobMatcher
        from job_applicator.config import EmbeddingConfig

        # Compute current match scores
        matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5))
        match_result = matcher.match_resume_to_job(resume, job)

        logger.info(
            "Current match: %.0f%% (semantic=%.0f%%, skill=%.0f%%)",
            match_result.score * 100,
            0,  # Will be computed
            0,
        )

        # Build user instructions section
        instruction_section = ""
        if user_instructions:
            instruction_section = f"Additional instructions from user:\n{user_instructions}"
        else:
            instruction_section = "No additional instructions."

        # Build prompt
        prompt = TAILOR_PROMPT_TEMPLATE.format(
            job_title=job.title,
            job_company=job.company,
            job_location=job.location,
            job_description=job.description[:800],
            requirements=", ".join(job.requirements),
            resume_text=resume.raw_text[:2000],
            skills=", ".join(resume.skills),
            user_instructions=instruction_section,
        )

        # Add style guide if provided
        if style_guide:
            from job_applicator.documents.style_analyzer import StyleAnalyzer

            analyzer = StyleAnalyzer(self._config)
            style_section = analyzer.format_style_for_prompt(style_guide)
            prompt += f"\n\n{style_section}"

        # Call LLM
        tailored_text = await self._call_llm(prompt)

        # Generate changes summary
        changes = await self._summarize_changes(resume.raw_text, tailored_text)

        return TailoredResume(
            original_path="",
            tailored_text=tailored_text,
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            match_score=match_result.score,
            semantic_score=0.0,  # Will be computed by caller if needed
            skill_score=0.0,
            matched_skills=match_result.matched_skills,
            missing_skills=match_result.missing_skills,
            changes_summary=changes,
            user_modifications=user_instructions,
        )

    @async_retry(max_attempts=2, base_delay=1.0, exceptions=(LLMError,))
    async def refine(
        self,
        original_resume: ResumeData,
        current_tailored: TailoredResume,
        user_feedback: str,
        job: JobListing,
    ) -> TailoredResume:
        """Refine a tailored resume based on user feedback.

        Args:
            original_resume: The original resume
            current_tailored: The current tailored version
            user_feedback: User's feedback/instructions
            job: Target job listing

        Returns:
            New TailoredResume with refinements applied
        """
        prompt = f"""The user wants changes to this tailored resume.

Job: {job.title} at {job.company}
Requirements: {', '.join(job.requirements)}

Current tailored resume:
---
{current_tailored.tailored_text[:2000]}
---

User feedback:
{user_feedback}

Apply the user's feedback while keeping the resume tailored for the job.
Return the complete updated resume text."""

        refined_text = await self._call_llm(prompt)
        changes = await self._summarize_changes(
            current_tailored.tailored_text, refined_text
        )

        return TailoredResume(
            original_path=current_tailored.original_path,
            tailored_text=refined_text,
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            match_score=current_tailored.match_score,
            semantic_score=current_tailored.semantic_score,
            skill_score=current_tailored.skill_score,
            matched_skills=current_tailored.matched_skills,
            missing_skills=current_tailored.missing_skills,
            changes_summary=changes,
            user_modifications=user_feedback,
            attempt=current_tailored.attempt + 1,
        )

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response text."""
        try:
            from litellm import acompletion

            model = (
                f"openai/{self._config.model}"
                if self._config.api_base
                else self._config.model
            )

            response = await acompletion(
                model=model,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                messages=[
                    {"role": "system", "content": TAILOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._config.max_tokens * 2,
                temperature=self._config.temperature,
            )

            from job_applicator.documents.cover_letter import strip_thinking_process

            content = strip_thinking_process(response.choices[0].message.content)
            return content.strip()

        except Exception as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

    async def _summarize_changes(self, original: str, tailored: str) -> str:
        """Generate a summary of changes between original and tailored."""
        try:
            prompt = CHANGES_PROMPT_TEMPLATE.format(
                original_preview=original[:500],
                tailored_preview=tailored[:500],
            )
            return await self._call_llm(prompt)
        except Exception:
            return "Changes applied (summary generation failed)"
```

- [ ] **Step 2: Run lint + format**

```bash
ruff check src/job_applicator/documents/resume_tailor.py
ruff format --check src/job_applicator/documents/resume_tailor.py
```

Expected: All pass.

- [ ] **Step 3: Run typecheck**

```bash
mypy src/job_applicator/documents/resume_tailor.py --ignore-missing-imports
```

Expected: Pass.

---

### Task 3: Add CLI tailor command with interactive loop

**Covers:** Preview, accept/retry/input workflow

**Files:**
- Modify: `src/job_applicator/cli.py`

- [ ] **Step 1: Add tailor command after match command (around line 402)**

Insert after the `match` command's `asyncio.run(_run())`:

```python
@app.command()
def tailor(
    resume_path: str = typer.Option("", "--resume", help="Path to resume file."),
    job_title: str = typer.Option(..., "--job-title", "-t", help="Job title."),
    company: str = typer.Option(..., "--company", "-c", help="Company name."),
    job_description: str = typer.Option("", "--description", "-d", help="Job description."),
    job_url: str = typer.Option("", "--url", help="Job posting URL."),
    requirements: str = typer.Option("", "--requirements", "-r", help="Comma-separated requirements."),
    location: str = typer.Option("", "--location", "-l", help="Job location."),
    style_guide: str = typer.Option("", "--style-guide", help="Style examples."),
    headed: bool = typer.Option(False, "--headed", help="Run browser in headed mode."),
) -> None:
    """Tailor a resume for a specific job with interactive preview."""
    settings = _get_settings(headed)
    if resume_path:
        settings.resume_path = resume_path
    if style_guide:
        settings.style_guide_path = style_guide
    setup_logging(settings.log_level)

    async def _run() -> None:
        from pydantic import HttpUrl

        from job_applicator.documents.resume import ResumeLoader
        from job_applicator.documents.resume_tailor import ResumeTailor
        from job_applicator.models import JobBoard, JobListing

        if not settings.resume_path:
            console.print("[red]Resume path required. Use --resume.[/red]")
            raise typer.Exit(1)

        # Load resume
        loader = ResumeLoader()
        resume_data = loader.load(settings.resume_path)
        console.print(f"[green]Loaded resume: {resume_data.name}[/green]")

        # Build job listing
        req_list = [r.strip() for r in requirements.split(",") if r.strip()] if requirements else []
        url = HttpUrl(job_url) if job_url else HttpUrl("https://example.com/placeholder")

        job = JobListing(
            title=job_title,
            company=company,
            description=job_description,
            url=url,
            requirements=req_list,
            location=location,
            board=JobBoard.INDEED,
        )

        # Load style guide if provided
        style = None
        if settings.style_guide_path:
            from job_applicator.documents.cover_letter import CoverLetterGenerator

            generator = CoverLetterGenerator(settings.llm)
            with console.status("Analyzing writing style..."):
                style = await generator.load_style_guide(settings.style_guide_path)
            console.print(f"[green]Style loaded: {style.tone}[/green]")

        # Interactive tailoring loop
        tailor_engine = ResumeTailor(settings.llm)
        attempt = 0
        user_instructions = ""

        while True:
            attempt += 1
            console.print(f"\n[bold blue]--- Attempt #{attempt} ---[/bold blue]")

            with console.status("Tailoring resume..."):
                if attempt == 1:
                    result = await tailor_engine.tailor(
                        resume_data, job, user_instructions, style
                    )
                else:
                    result = await tailor_engine.refine(
                        resume_data, result, user_instructions, job
                    )

            result.attempt = attempt

            # Display preview
            console.print("\n[bold]📋 Tailored Resume Preview:[/bold]\n")
            console.print(Panel(result.tailored_text, title="Tailored Resume", border_style="cyan"))

            # Display metadata
            console.print("\n[bold]📊 Metadata:[/bold]")
            meta_table = Table(show_header=False, box=None)
            meta_table.add_column("Key", style="dim")
            meta_table.add_column("Value")
            meta_table.add_row("Job", f"{job.title} at {job.company}")
            meta_table.add_row("Match Score", f"{result.match_score:.0%}")
            meta_table.add_row("Matched Skills", ", ".join(result.matched_skills[:5]) or "—")
            meta_table.add_row("Missing Skills", ", ".join(result.missing_skills[:5]) or "—")
            meta_table.add_row("Attempt", str(attempt))
            if result.user_modifications:
                meta_table.add_row("User Input", result.user_modifications)
            console.print(meta_table)

            # Display changes
            console.print("\n[bold]📝 Changes Made:[/bold]")
            console.print(result.changes_summary)

            # Interactive prompt
            console.print("\n[bold]What would you like to do?[/bold]")
            action_table = Table(show_header=False, box=None)
            action_table.add_column("Option", style="cyan bold")
            action_table.add_column("Description")
            action_table.add_row("[A] Accept", "Save this version as final")
            action_table.add_row("[R] Retry", "Regenerate with same instructions")
            action_table.add_row("[I] Input", "Give custom instructions to refine")
            action_table.add_row("[Q] Quit", "Discard and exit")
            console.print(action_table)

            choice = console.input("\n[bold cyan]Your choice (A/R/I/Q): [/bold cyan]").strip().upper()

            if choice == "A":
                # Save the tailored resume
                from pathlib import Path
                from datetime import datetime

                output_dir = Path(settings.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                safe_company = job.company.replace(" ", "_").replace("/", "_")
                safe_title = job.title.replace(" ", "_").replace("/", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
                output_path = output_dir / filename

                # Write resume
                output_path.write_text(result.tailored_text, encoding="utf-8")
                result.output_path = str(output_path)

                # Write metadata
                meta_path = output_path.with_suffix(".meta.json")
                meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

                console.print(f"\n[green]✓ Tailored resume saved: {output_path}[/green]")
                console.print(f"[green]✓ Metadata saved: {meta_path}[/green]")
                console.print(f"[dim]Attempt #{attempt} | Score: {result.match_score:.0%}[/dim]")
                break

            elif choice == "R":
                console.print("[yellow]Regenerating...[/yellow]")
                user_instructions = ""
                continue

            elif choice == "I":
                user_instructions = console.input(
                    "\n[bold]Enter your instructions (e.g., 'emphasize customer service experience', "
                    "'add more detail about troubleshooting skills'): [/bold]"
                ).strip()
                if not user_instructions:
                    console.print("[yellow]No instructions provided, retrying with defaults.[/yellow]")
                continue

            elif choice == "Q":
                console.print("[yellow]Discarded. No changes saved.[/yellow]")
                break

            else:
                console.print("[red]Invalid choice. Please enter A, R, I, or Q.[/red]")

    asyncio.run(_run())
```

- [ ] **Step 2: Run lint + format**

```bash
ruff check src/job_applicator/cli.py
ruff format --check src/job_applicator/cli.py
```

Expected: All pass.

---

### Task 4: Create CGI tailoring script

**Covers:** Standalone script for the CGI job

**Files:**
- Create: `scripts/tailor_cgi.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Tailor resume for CGI Technical Support Specialist (best match from report)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

RESUME_PATH = (
    "/media/drei/KINGSTON/Andrei School/Other/Jobhunt/Andrei_Petrov_Resume.pdf"
)

CGI_JOB = {
    "title": "Technical Support Specialist",
    "company": "CGI",
    "url": "https://ca.indeed.com/viewjob?jk=1234567890abc",
    "description": (
        "Provide technical support to clients via phone, email and chat. "
        "Troubleshoot hardware and software issues. Use ticketing systems "
        "like ServiceNow. 3+ years experience required."
    ),
    "requirements": [
        "Technical Support",
        "Troubleshooting",
        "ServiceNow",
        "Windows",
        "Office 365",
    ],
    "location": "Montreal, QC",
}


async def main():
    console = Console()

    console.print(
        Panel.fit(
            "[bold]Resume Tailor — CGI Technical Support Specialist[/bold]",
            style="blue",
        )
    )

    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.models import JobBoard, JobListing

    # Load resume
    console.print("\n[bold]Loading resume...[/bold]")
    loader = ResumeLoader()
    resume = loader.load(RESUME_PATH)
    console.print(f"  Name: {resume.name}")
    console.print(f"  Skills: {', '.join(resume.skills[:6])}")

    # Build job listing
    from pydantic import HttpUrl

    job = JobListing(
        title=CGI_JOB["title"],
        company=CGI_JOB["company"],
        url=HttpUrl(CGI_JOB["url"]),
        description=CGI_JOB["description"],
        requirements=CGI_JOB["requirements"],
        location=CGI_JOB["location"],
        board=JobBoard.INDEED,
    )

    # Run tailor
    from job_applicator.config import LLMConfig

    config = LLMConfig()
    tailor_engine = ResumeTailor(config)

    attempt = 0
    user_instructions = ""
    result = None

    while True:
        attempt += 1
        console.print(f"\n[bold blue]=== Attempt #{attempt} ===[/bold blue]")

        with console.status("Tailoring resume with LLM..."):
            if attempt == 1:
                result = await tailor_engine.tailor(
                    resume, job, user_instructions
                )
            else:
                result = await tailor_engine.refine(
                    resume, result, user_instructions, job
                )

        result.attempt = attempt

        # Preview
        console.print("\n[bold]Tailored Resume:[/bold]\n")
        console.print(
            Panel(result.tailored_text, title="Preview", border_style="cyan")
        )

        # Metadata
        console.print("\n[bold]Metadata:[/bold]")
        meta = Table(show_header=False, box=None)
        meta.add_column("Key", style="dim")
        meta.add_column("Value")
        meta.add_row("Job", f"{job.title} at {job.company}")
        meta.add_row("Match Score", f"{result.match_score:.0%}")
        meta.add_row(
            "Matched Skills",
            ", ".join(result.matched_skills[:5]) or "—",
        )
        meta.add_row(
            "Missing Skills",
            ", ".join(result.missing_skills[:5]) or "—",
        )
        meta.add_row("Attempt", str(attempt))
        console.print(meta)

        # Changes
        console.print("\n[bold]Changes:[/bold]")
        console.print(result.changes_summary)

        # Interactive prompt
        console.print("\n[bold]Options:[/bold]")
        opts = Table(show_header=False, box=None)
        opts.add_column("Key", style="cyan bold")
        opts.add_column("Description")
        opts.add_row("[A] Accept", "Save this version")
        opts.add_row("[R] Retry", "Regenerate")
        opts.add_row("[I] Input", "Give custom instructions")
        opts.add_row("[Q] Quit", "Discard and exit")
        console.print(opts)

        choice = console.input(
            "\n[bold cyan]Choice (A/R/I/Q): [/bold cyan]"
        ).strip().upper()

        if choice == "A":
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"tailored_CGI_TechSupport_{ts}.txt"
            out = output_dir / fname
            out.write_text(result.tailored_text, encoding="utf-8")
            result.output_path = str(out)

            meta_path = out.with_suffix(".meta.json")
            meta_path.write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )

            console.print(f"\n[green]Saved: {out}[/green]")
            console.print(f"[green]Meta:  {meta_path}[/green]")
            break

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            user_instructions = ""

        elif choice == "I":
            user_instructions = console.input(
                "\n[bold]Instructions: [/bold]"
            ).strip()

        elif choice == "Q":
            console.print("[yellow]Discarded.[/yellow]")
            break

        else:
            console.print("[red]Invalid choice.[/red]")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
```

- [ ] **Step 2: Run lint + format**

```bash
ruff check scripts/tailor_cgi.py
ruff format --check scripts/tailor_cgi.py
```

Expected: All pass.

---

### Task 5: Add unit tests for ResumeTailor

**Covers:** Test coverage for tailoring logic

**Files:**
- Create: `tests/unit/test_resume_tailor.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for resume tailoring engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from job_applicator.documents.resume_tailor import (
    CHANGES_PROMPT_TEMPLATE,
    TAILOR_PROMPT_TEMPLATE,
    TAILOR_SYSTEM_PROMPT,
    ResumeTailor,
)
from job_applicator.models import JobBoard, JobListing, ResumeData, TailoredResume


@pytest.fixture
def sample_resume():
    return ResumeData(
        raw_text="ANDREI PETROV\nandre@example.com\nSkills\nWindows, Office 365, Troubleshooting",
        name="ANDREI PETROV",
        email="andre@example.com",
        skills=["Windows", "Office 365", "Troubleshooting"],
    )


@pytest.fixture
def sample_job():
    return JobListing(
        title="Technical Support Specialist",
        company="CGI",
        url="https://example.com/job",
        description="Provide technical support.",
        requirements=["Windows", "Office 365", "ServiceNow"],
        location="Montreal, QC",
        board=JobBoard.INDEED,
    )


@pytest.fixture
def llm_config():
    from job_applicator.config import LLMConfig

    return LLMConfig(
        api_base="http://localhost:8000/v1",
        model="test-model",
    )


class TestResumeTailor:
    def test_init(self, llm_config):
        tailor = ResumeTailor(llm_config)
        assert tailor._config == llm_config

    def test_prompt_template_formatting(self):
        prompt = TAILOR_PROMPT_TEMPLATE.format(
            job_title="Test Job",
            job_company="Test Co",
            job_location="Remote",
            job_description="Test desc",
            requirements="Skill1, Skill2",
            resume_text="Resume text",
            skills="Skill1, Skill2",
            user_instructions="No instructions.",
        )
        assert "Test Job" in prompt
        assert "Test Co" in prompt
        assert "Resume text" in prompt

    def test_changes_prompt_template(self):
        prompt = CHANGES_PROMPT_TEMPLATE.format(
            original_preview="Original text",
            tailored_preview="Tailored text",
        )
        assert "Original text" in prompt
        assert "Tailored text" in prompt

    @pytest.mark.asyncio
    async def test_tailor_returns_result(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "ANDREI PETROV\nandre@example.com\n"
            "Skills: Windows, Office 365, Troubleshooting, ServiceNow\n"
            "Experience: Technical Support..."
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await tailor.tailor(sample_resume, sample_job)

        assert isinstance(result, TailoredResume)
        assert result.job_title == "Technical Support Specialist"
        assert result.job_company == "CGI"
        assert result.attempt == 1
        assert len(result.tailored_text) > 0

    @pytest.mark.asyncio
    async def test_refine_increments_attempt(self, llm_config, sample_resume, sample_job):
        tailor = ResumeTailor(llm_config)

        initial = TailoredResume(
            original_path="",
            tailored_text="Initial tailored text",
            job_title="Technical Support Specialist",
            job_company="CGI",
            match_score=0.7,
            semantic_score=0.76,
            skill_score=0.6,
            matched_skills=["Windows"],
            missing_skills=["ServiceNow"],
            changes_summary="Initial changes",
            attempt=1,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Refined resume text"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await tailor.refine(sample_resume, initial, "Add more detail", sample_job)

        assert result.attempt == 2
        assert result.user_modifications == "Add more detail"


class TestTailoredResumeModel:
    def test_model_creation(self):
        resume = TailoredResume(
            original_path="/path/to/resume.pdf",
            tailored_text="Tailored content",
            job_title="Test Job",
            job_company="Test Co",
            match_score=0.75,
            semantic_score=0.8,
            skill_score=0.65,
            matched_skills=["Python"],
            missing_skills=["AWS"],
            changes_summary="Emphasized Python skills",
        )
        assert resume.attempt == 1
        assert resume.user_modifications == ""
        assert resume.output_path == ""

    def test_model_serialization(self):
        resume = TailoredResume(
            original_path="",
            tailored_text="text",
            job_title="Job",
            job_company="Co",
            match_score=0.5,
            semantic_score=0.5,
            skill_score=0.5,
            changes_summary="changes",
        )
        data = resume.model_dump()
        assert "tailored_text" in data
        assert "match_score" in data
        assert "created_at" in data
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/unit/test_resume_tailor.py -v
```

Expected: All pass.

- [ ] **Step 3: Run full test suite + lint**

```bash
pytest tests/unit/ -v
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/job_applicator/ --ignore-missing-imports
```

Expected: All pass.
