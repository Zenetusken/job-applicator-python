"""Interactive résumé-tailoring workflow — the accept/retry/refine/section-edit loop
extracted from the `tailor` command.

Orchestration only: it mutates the ``TailorSession`` and writes artifacts on accept.
The shared cli helper ``_llm_with_retry`` is imported lazily (inside the function) to
avoid a cli <-> workflow import cycle.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table

from job_applicator.utils.diff import render_diff
from job_applicator.workflows.cover_letter import _cover_letter_workflow

if TYPE_CHECKING:
    from rich.console import Console

    from job_applicator.config import AppSettings
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.models import (
        JobListing,
        ResumeData,
        StyleGuide,
        TailoredResume,
        TailorSession,
    )
    from job_applicator.utils.verbose import VerboseReporter


async def _tailor_workflow(
    console: Console,
    settings: AppSettings,
    job: JobListing,
    resume_data: ResumeData,
    style: StyleGuide | None,
    tone_profile: ToneProfile | None,
    tailor_engine: ResumeTailor,
    session: TailorSession,
    result: TailoredResume,
    reporter: VerboseReporter | None,
    yes: bool = False,
) -> None:
    """Run the interactive tailor loop until the user accepts ([A]) or quits ([Q]).

    ``yes`` (the ``--yes`` flag) makes it non-interactive: auto-accept the first tailored
    version ([A]) and skip the cover-letter offer (whose own workflow is interactive) —
    so the command never blocks on input in CI / non-tty.
    """
    from job_applicator.cli import _llm_with_retry

    attempt = 0
    user_instructions = ""

    while True:
        attempt += 1
        if attempt > 10:
            console.print("[red]Maximum retry limit (10) reached.[/red]")
            break
        if attempt >= 8:
            console.print("[yellow]Warning: approaching retry limit (10 max).[/yellow]")

        console.print(f"\n[bold blue]--- Attempt #{attempt} ---[/bold blue]")

        console.print("\n[bold]Tailored Resume Preview:[/bold]\n")
        console.print(
            Panel(
                result.tailored_text,
                title="Tailored Resume",
                border_style="cyan",
            )
        )
        render_diff(console, session.original_text, result.tailored_text, max_lines=30)

        console.print("\n[bold]Metadata:[/bold]")
        meta_table = Table(show_header=False, box=None)
        meta_table.add_column("Key", style="dim")
        meta_table.add_column("Value")
        meta_table.add_row("Job", f"{job.title} at {job.company}")
        meta_table.add_row("Match Score", f"{result.match_score:.0%}")
        meta_table.add_row(
            "Matched Skills",
            ", ".join(result.matched_skills[:5]) or "—",
        )
        meta_table.add_row(
            "Missing Skills",
            ", ".join(result.missing_skills[:5]) or "—",
        )
        meta_table.add_row("Attempt", str(attempt))
        if result.user_modifications:
            meta_table.add_row("User Input", result.user_modifications)
        console.print(meta_table)

        console.print("\n[bold]Changes Made:[/bold]")
        console.print(result.changes_summary)

        console.print("\n[bold]What would you like to do?[/bold]")
        action_table = Table(show_header=False, box=None)
        action_table.add_column("Option", style="cyan bold")
        action_table.add_column("Description")
        action_table.add_row("[A] Accept", "Save this version as final")
        action_table.add_row("[R] Retry", "Regenerate with same instructions")
        action_table.add_row("[I] Input", "Give custom instructions to refine")
        action_table.add_row("[D] Diff", "Show changes from original resume")
        action_table.add_row("[V] History", "Browse previous attempts")
        action_table.add_row("[S] Section", "Edit a specific section")
        action_table.add_row("[Q] Quit", "Discard and exit")
        console.print(action_table)

        if yes:
            console.print("\n[dim]--yes: accepting this version automatically.[/dim]")
            choice = "A"
        else:
            choice = (
                console.input("\n[bold cyan]Your choice (A/R/I/D/V/S/Q): [/bold cyan]")
                .strip()
                .upper()
            )

        if choice == "A":
            from datetime import datetime as dt

            output_dir = await asyncio.to_thread(settings.ensure_output_dir)

            safe_company = job.company.replace(" ", "_").replace("/", "_")
            safe_title = job.title.replace(" ", "_").replace("/", "_")
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tailored_{safe_company}_{safe_title}_{timestamp}.txt"
            output_path = output_dir / filename

            await asyncio.to_thread(output_path.write_text, result.tailored_text, encoding="utf-8")
            result.output_path = str(output_path)

            if reporter:
                reporter.record_io(files_written=[str(output_path)])

            console.print(f"\n[green]Tailored resume saved: {output_path}[/green]")
            console.print(f"[dim]Attempt #{attempt} | Score: {result.match_score:.0%}[/dim]")

            # Offer cover letter generation
            cover_letter_path = None
            if yes:
                # Non-interactive: skip the offer — _cover_letter_workflow is itself
                # interactive. Use `generate-cover-letter` / `batch --cover-letter` for a
                # non-interactive cover letter.
                cl_choice = "N"
            else:
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
                    console,
                    settings,
                    job,
                    resume_data,
                    style,
                    tone_profile,
                    result.tailored_text,
                )

            # Write resume meta.json (with or without cover_letter_path)
            if cover_letter_path:
                result.cover_letter_path = str(cover_letter_path)
            meta_path = output_path.with_suffix(".meta.json")
            await asyncio.to_thread(
                meta_path.write_text, result.model_dump_json(indent=2), encoding="utf-8"
            )
            console.print(f"[green]Metadata saved: {meta_path}[/green]")

            break

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            user_instructions = ""
            refined: TailoredResume | None = await _llm_with_retry(
                console,
                partial(
                    tailor_engine.refine,
                    resume_data,
                    result,
                    "",
                    job,
                    tone_profile=tone_profile,
                    style_guide=style,
                ),
                "Tailoring resume...",
            )
            if refined is None:
                break
            result = refined
            result.attempt = attempt
            session.add_attempt(result)
            continue

        elif choice == "I":
            user_instructions = console.input(
                "\n[bold]Enter your instructions (e.g., 'emphasize "
                "customer service', 'add troubleshooting detail'): "
                "[/bold]"
            ).strip()
            if not user_instructions:
                console.print("[yellow]No instructions provided, retrying.[/yellow]")
            refined = await _llm_with_retry(
                console,
                partial(
                    tailor_engine.refine,
                    resume_data,
                    result,
                    user_instructions,
                    job,
                    tone_profile=tone_profile,
                    style_guide=style,
                ),
                "Tailoring resume...",
            )
            if refined is None:
                break
            result = refined
            result.attempt = attempt
            session.add_attempt(result)
            continue

        elif choice == "D":
            render_diff(console, session.original_text, result.tailored_text, max_lines=0)
            continue

        elif choice == "V":
            if len(session.attempts) < 2:
                console.print("[yellow]No previous attempts yet.[/yellow]")
                continue
            hist_table = Table(title="Version History")
            hist_table.add_column("#", style="dim")
            hist_table.add_column("Attempt")
            hist_table.add_column("Score", style="cyan")
            hist_table.add_column("Instructions")
            hist_table.add_column("Preview", style="dim")
            for i, att in enumerate(session.attempts):
                preview = att.tailored_text[:60].replace("\n", " ")
                marker = "\u2192" if i == session.current_index else " "
                hist_table.add_row(
                    marker,
                    str(att.attempt),
                    f"{att.match_score:.0%}",
                    att.user_modifications or "\u2014",
                    preview + "...",
                )
            console.print(hist_table)
            sel = console.input(
                "\n[bold cyan]Select attempt # to view (or Enter to go back): [/bold cyan]"
            ).strip()
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(session.attempts):
                    session.select(idx)
                    result = session.current
                    console.print(f"[green]Switched to attempt #{result.attempt}[/green]")
                else:
                    console.print("[red]Invalid attempt number.[/red]")
            continue

        elif choice == "S":
            from job_applicator.documents.resume_tailor import parse_sections

            sections = parse_sections(result.tailored_text)
            if len(sections) <= 1 and sections[0].name == "Full Document":
                console.print(
                    "[yellow]Could not detect sections. "
                    "Use [I] for full-resume instructions.[/yellow]"
                )
                continue

            console.print("\n[bold]Sections:[/bold]")
            sec_table = Table(show_header=False, box=None)
            sec_table.add_column("#", style="cyan")
            sec_table.add_column("Section", style="bold")
            sec_table.add_column("Lines", style="dim")
            for i, sec in enumerate(sections, 1):
                line_count = sec.text.count("\n") + 1
                sec_table.add_row(str(i), sec.name, f"{line_count} lines")
            console.print(sec_table)

            sec_choice = console.input(
                "\n[bold cyan]Section # to edit (or Enter to go back): [/bold cyan]"
            ).strip()
            if not sec_choice.isdigit():
                continue
            sec_idx = int(sec_choice) - 1
            if sec_idx < 0 or sec_idx >= len(sections):
                console.print("[red]Invalid section number.[/red]")
                continue

            target_section = sections[sec_idx]
            console.print(f"\n[dim]Editing: {target_section.name}[/dim]")
            console.print(f"[dim]{target_section.text[:200]}...[/dim]\n")

            sec_instructions = console.input("[bold]Instructions for this section: [/bold]").strip()
            if not sec_instructions:
                console.print("[yellow]No instructions provided.[/yellow]")
                continue

            user_instructions = (
                f"ONLY modify the {target_section.name} section. "
                f"Keep all other sections unchanged.\n\n"
                f"Current {target_section.name} content:\n{target_section.text}\n\n"
                f"User instructions for this section: {sec_instructions}"
            )
            refined = await _llm_with_retry(
                console,
                partial(
                    tailor_engine.refine,
                    resume_data,
                    result,
                    user_instructions,
                    job,
                    tone_profile=tone_profile,
                    style_guide=style,
                ),
                "Refining section...",
            )
            if refined is None:
                break
            result = refined
            result.attempt = attempt
            session.add_attempt(result)
            continue

        elif choice == "Q":
            console.print("[yellow]Discarded. No changes saved.[/yellow]")
            break

        else:
            console.print("[red]Invalid choice. Please enter A, R, I, D, V, S, or Q.[/red]")
