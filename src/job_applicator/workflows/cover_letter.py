"""Interactive cover-letter generation workflow — extracted from cli.py.

The accept/retry/refine loop plus its generate/refine/save helpers. Orchestration
only: it composes ``documents.cover_letter`` (the generator) with the shared LLM
runtime. The shared cli helpers ``_detect_tone`` / ``_load_user_profile`` are imported
lazily from cli (inside the functions) to avoid a cli ↔ workflow import cycle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from job_applicator.factories import _make_runtime
from job_applicator.utils.diff import render_diff

if TYPE_CHECKING:
    from rich.console import Console

    from job_applicator.config import AppSettings
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.models import (
        CoverLetterResult,
        CoverLetterSession,
        JobListing,
        ResumeData,
        StyleGuide,
    )
    from job_applicator.utils.llm import LLMRuntime


async def _generate_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    resume_data: ResumeData,
    style: StyleGuide | None,
    tone_section: str,
    tailored_resume_text: str,
    session: CoverLetterSession,
    attempt: int = 1,
    *,
    runtime: LLMRuntime,
) -> CoverLetterResult | None:
    """Generate a cover letter via LLM. Returns None on failure.

    ``runtime`` is REQUIRED so the caller always shares one breaker — a per-call
    default would reset the breaker across the interactive retry loop.
    """
    from job_applicator.cli import _load_user_profile
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.models import CoverLetterResult

    generator = CoverLetterGenerator(settings.llm, runtime=runtime)
    try:
        with console.status("Generating cover letter..."):
            letter = await generator.generate(
                job,
                _load_user_profile(settings),
                resume_data,
                style_guide=style,
                tone_section=tone_section,
                tailored_resume_text=tailored_resume_text,
            )
        result = CoverLetterResult(
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            cover_letter_text=letter,
            attempt=attempt,
            prompt_version="1.0",
        )
        session.add_attempt(result)
        return result
    except Exception as exc:
        console.print(f"[red]LLM error: {escape(str(exc))}[/red]")
        return None


async def _save_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    result: CoverLetterResult,
) -> Path:
    """Save cover letter to disk and return the path."""
    from datetime import datetime as dt

    output_dir = await asyncio.to_thread(settings.ensure_output_dir)
    safe_company = job.company.replace(" ", "_").replace("/", "_")
    safe_title = job.title.replace(" ", "_").replace("/", "_")
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    cl_filename = f"cover_letter_{safe_company}_{safe_title}_{timestamp}.txt"
    cl_path = output_dir / cl_filename
    await asyncio.to_thread(cl_path.write_text, result.cover_letter_text, encoding="utf-8")
    result.output_path = str(cl_path)
    cl_meta_path = cl_path.with_suffix(".meta.json")
    await asyncio.to_thread(
        cl_meta_path.write_text, result.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(f"\n[green]Cover letter saved: {cl_path}[/green]")
    return cl_path


async def _refine_cover_letter(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    result: CoverLetterResult,
    user_instructions: str,
    session: CoverLetterSession,
    attempt: int,
    resume_data: ResumeData | None = None,
    style: StyleGuide | None = None,
    tone_section: str = "",
    *,
    runtime: LLMRuntime,
) -> bool:
    """Refine a cover letter via LLM (shared ``runtime`` REQUIRED; see _generate_cover_letter)."""
    from job_applicator.documents.cover_letter import CoverLetterGenerator
    from job_applicator.models import CoverLetterResult as CLResult
    from job_applicator.models import ResumeData

    try:
        generator = CoverLetterGenerator(settings.llm, runtime=runtime)
        with console.status("Refining cover letter..."):
            refined = await generator.refine(
                job=job,
                resume=resume_data or ResumeData(raw_text=""),
                current_text=result.cover_letter_text,
                user_feedback=user_instructions,
                style_guide=style,
                tone_section=tone_section,
            )
        new_result = CLResult(
            job_title=job.title,
            job_company=job.company,
            job_url=str(job.url),
            cover_letter_text=refined,
            user_modifications=user_instructions,
            attempt=attempt + 1,
        )
        session.add_attempt(new_result)
        return True
    except Exception as exc:
        console.print(f"[red]LLM error: {escape(str(exc))}[/red]")
        return False


async def _cover_letter_workflow(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    resume_data: ResumeData,
    style: StyleGuide | None,
    tone_profile: ToneProfile | None,
    tailored_resume_text: str,
) -> Path | None:
    """Generate and save a cover letter with accept/retry workflow.

    Returns the Path to the saved cover letter, or None if skipped.
    """
    from job_applicator.cli import _detect_tone
    from job_applicator.documents.tone_detector import ToneDetector
    from job_applicator.models import CoverLetterSession

    tone_section = ""
    if tone_profile is None:
        tone_profile = _detect_tone(job)

    tone_section = ToneDetector().format_for_prompt(tone_profile)

    session = CoverLetterSession(job_title=job.title, job_company=job.company)
    # One breaker shared across every attempt of this interactive loop (a fresh
    # runtime per call would reset the failure counter and never trip the breaker).
    runtime = _make_runtime(settings)
    attempt = 0

    result = await _generate_cover_letter(
        console,
        settings,
        job,
        resume_data,
        style,
        tone_section,
        tailored_resume_text,
        session,
        runtime=runtime,
    )
    if result is None:
        retry = console.input("[bold cyan][R] Retry or [Q] Skip? [/bold cyan]").strip().upper()
        if retry == "R":
            result = await _generate_cover_letter(
                console,
                settings,
                job,
                resume_data,
                style,
                tone_section,
                tailored_resume_text,
                session,
                runtime=runtime,
            )
            if result is None:
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

        if len(session.attempts) > 1:
            render_diff(
                console,
                session.attempts[0].cover_letter_text,
                result.cover_letter_text,
                max_lines=30,
            )

        console.print("\n[bold]What would you like to do?[/bold]")
        action_table = Table(show_header=False, box=None)
        action_table.add_column("Option", style="cyan bold")
        action_table.add_column("Description")
        action_table.add_row("[A] Accept", "Save this cover letter")
        action_table.add_row("[R] Retry", "Regenerate")
        action_table.add_row("[I] Input", "Give custom instructions")
        action_table.add_row("[D] Diff", "Show full diff from first attempt")
        action_table.add_row("[V] History", "Browse previous attempts")
        action_table.add_row("[Q] Skip", "Discard (resume already saved)")
        console.print(action_table)

        choice = (
            console.input("\n[bold cyan]Your choice (A/R/I/D/V/Q): [/bold cyan]").strip().upper()
        )

        if choice == "A":
            return await _save_cover_letter(console, settings, job, result)

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            new_result = await _generate_cover_letter(
                console,
                settings,
                job,
                resume_data,
                style,
                tone_section,
                tailored_resume_text,
                session,
                attempt=attempt + 1,
                runtime=runtime,
            )
            if new_result is None:
                console.print("[red]Generation failed. Please try again.[/red]")
            continue

        elif choice == "I":
            user_instructions = console.input(
                "\n[bold]Instructions (e.g., 'emphasize customer service'): [/bold]"
            ).strip()
            if not user_instructions:
                console.print("[yellow]No instructions provided.[/yellow]")
                continue
            refined_ok = await _refine_cover_letter(
                console,
                settings,
                job,
                result,
                user_instructions,
                session,
                attempt,
                resume_data,
                style,
                tone_section,
                runtime=runtime,
            )
            if not refined_ok:
                console.print("[red]Refinement failed. Please try again.[/red]")
                continue
            result = session.current
            continue

        elif choice == "D":
            if len(session.attempts) > 1:
                render_diff(
                    console,
                    session.attempts[0].cover_letter_text,
                    result.cover_letter_text,
                    max_lines=0,
                )
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
            sel = console.input(
                "\n[bold cyan]Select attempt # (or Enter back): [/bold cyan]"
            ).strip()
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
