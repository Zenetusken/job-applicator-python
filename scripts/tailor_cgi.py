#!/usr/bin/env python3
"""Tailor resume for CGI Technical Support Specialist (best match from report)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from job_applicator.models import UserProfile
from job_applicator.utils.diff import render_diff as _render_diff

RESUME_PATH = "/media/drei/KINGSTON/Andrei School/Other/Jobhunt/Andrei_Petrov_Resume.pdf"

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


async def _cover_letter_workflow(
    console: Console,
    config: object,
    job: object,
    resume: object,
    tone_profile: object,
    tailored_resume_text: str,
) -> Path | None:
    from job_applicator.documents.cover_letter import CoverLetterGenerator, strip_thinking_process
    from job_applicator.models import CoverLetterResult, CoverLetterSession

    generator = CoverLetterGenerator(config)
    tone_section = ""
    if tone_profile:
        from job_applicator.documents.tone_detector import ToneDetector

        tone_section = ToneDetector().format_for_prompt(tone_profile)

    name_parts = (resume.name or "User").split()
    user = UserProfile(
        first_name=name_parts[0],
        last_name=name_parts[-1] if len(name_parts) > 1 else "",
        email=resume.email or "",
        phone=resume.phone or "",
        resume_path=RESUME_PATH,
    )

    session = CoverLetterSession(job_title=job.title, job_company=job.company)
    attempt = 0

    try:
        with console.status("Generating cover letter..."):
            letter = await generator.generate(
                job,
                user,
                resume,
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
            try:
                with console.status("Generating cover letter..."):
                    letter = await generator.generate(
                        job,
                        user,
                        resume,
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

        if len(session.attempts) > 1:
            _render_diff(
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
            from datetime import datetime as dt

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
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
            try:
                with console.status("Generating cover letter..."):
                    letter = await generator.generate(
                        job,
                        user,
                        resume,
                        tone_section=tone_section,
                        tailored_resume_text=tailored_resume_text,
                    )
                new_result = CoverLetterResult(
                    job_title=job.title,
                    job_company=job.company,
                    job_url=str(job.url),
                    cover_letter_text=letter,
                    attempt=attempt + 1,
                )
                session.add_attempt(new_result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
            continue

        elif choice == "I":
            user_instructions = console.input(
                "\n[bold]Instructions (e.g., 'emphasize customer service'): [/bold]"
            ).strip()
            if not user_instructions:
                console.print("[yellow]No instructions provided.[/yellow]")
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

                    from job_applicator.utils.llm import litellm_model

                    model = litellm_model(config)
                    response = await acompletion(
                        model=model,
                        api_base=config.api_base,
                        api_key=config.api_key,
                        messages=[{"role": "user", "content": refine_prompt}],
                        max_tokens=config.max_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    refined = strip_thinking_process(response.choices[0].message.content)
                new_result = CoverLetterResult(
                    job_title=job.title,
                    job_company=job.company,
                    job_url=str(job.url),
                    cover_letter_text=refined,
                    user_modifications=user_instructions,
                    attempt=attempt + 1,
                )
                session.add_attempt(new_result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
            continue

        elif choice == "D":
            if len(session.attempts) > 1:
                _render_diff(
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
                marker = "\u2192" if i == session.current_index else " "
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


async def main() -> bool:
    console = Console()

    console.print(
        Panel.fit(
            "[bold]Resume Tailor — CGI Technical Support Specialist[/bold]",
            style="blue",
        )
    )

    from pydantic import HttpUrl

    from job_applicator.documents.resume import ResumeLoader
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.models import JobBoard, JobListing, TailorSession

    console.print("\n[bold]Loading resume...[/bold]")
    loader = ResumeLoader()
    resume = loader.load(RESUME_PATH)
    console.print(f"  Name: {resume.name}")
    console.print(f"  Skills: {', '.join(resume.skills[:6])}")

    job = JobListing(
        title=CGI_JOB["title"],
        company=CGI_JOB["company"],
        url=HttpUrl(CGI_JOB["url"]),
        description=CGI_JOB["description"],
        requirements=CGI_JOB["requirements"],
        location=CGI_JOB["location"],
        board=JobBoard.INDEED,
    )

    from job_applicator.config import LLMConfig
    from job_applicator.documents.tone_detector import ToneDetector

    config = LLMConfig()
    tailor_engine = ResumeTailor(config)

    tone_detector = ToneDetector()
    tone_profile = tone_detector.detect(
        title=job.title,
        description=job.description,
        requirements=job.requirements,
    )
    console.print(
        f"[dim]Detected tone: {tone_profile.primary} "
        f"(confidence: {tone_profile.confidence:.0%})[/dim]"
    )

    session = TailorSession(
        original_text=resume.raw_text,
        job_title=job.title,
        job_company=job.company,
    )

    # Pre-ingestion date audit
    from job_applicator.documents.resume_tailor import ResumeDateValidator

    validator = ResumeDateValidator()
    audit = validator.audit(resume)

    console.print("\n[bold]CV Date Audit[/bold]")
    audit_table = Table(title="Date Analysis", show_lines=True)
    audit_table.add_column("Section", style="dim")
    audit_table.add_column("Entry", style="bold")
    audit_table.add_column("Start")
    audit_table.add_column("End")
    for entry in audit.entries:
        audit_table.add_row(
            entry.section,
            entry.label,
            entry.start,
            entry.end,
        )
    console.print(audit_table)

    console.print(f"\n[dim]Date range: {audit.earliest_date} → {audit.latest_date}[/dim]")

    if audit.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in audit.warnings:
            console.print(f"  [yellow]- {w}[/yellow]")

    if audit.staleness_issues:
        console.print("\n[bold red]Staleness:[/bold red]")
        for s in audit.staleness_issues:
            console.print(f"  [red]- {s}[/red]")

    if audit.ordering_issues:
        console.print("\n[bold red]Ordering:[/bold red]")
        for o in audit.ordering_issues:
            console.print(f"  [red]- {o}[/red]")

    if audit.is_stale or audit.ordering_issues:
        console.print(
            "\n[bold yellow]This CV may be outdated or have ordering "
            "issues. Please verify your CV is up to date.[/bold yellow]"
        )
        confirm = console.input("\n[bold cyan]Proceed anyway? (y/n): [/bold cyan]").strip().lower()
        if confirm != "y":
            console.print("[yellow]Aborted.[/yellow]")
            return False
    else:
        console.print("[green]Dates look coherent and current.[/green]")

    attempt = 0
    user_instructions = ""
    result = None

    try:
        with console.status("Tailoring resume with LLM..."):
            result = await tailor_engine.tailor(resume, job, user_instructions)
        session.add_attempt(result)
    except Exception as exc:
        console.print(f"[red]LLM error: {exc}[/red]")
        console.print("[yellow]Could not generate tailored resume.[/yellow]")
        return False

    while True:
        attempt += 1
        if attempt > 10:
            console.print("[red]Maximum retry limit (10) reached.[/red]")
            break
        if attempt >= 8:
            console.print("[yellow]Warning: approaching retry limit (10 max).[/yellow]")

        console.print(f"\n[bold blue]=== Attempt #{attempt} ===[/bold blue]")

        console.print("\n[bold]Tailored Resume:[/bold]\n")
        console.print(Panel(result.tailored_text, title="Preview", border_style="cyan"))
        _render_diff(console, session.original_text, result.tailored_text, max_lines=30)

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
        if result.user_modifications:
            meta.add_row("User Input", result.user_modifications)
        console.print(meta)

        console.print("\n[bold]Changes:[/bold]")
        console.print(result.changes_summary)

        console.print("\n[bold]Options:[/bold]")
        opts = Table(show_header=False, box=None)
        opts.add_column("Key", style="cyan bold")
        opts.add_column("Description")
        opts.add_row("[A] Accept", "Save this version")
        opts.add_row("[R] Retry", "Regenerate")
        opts.add_row("[I] Input", "Give custom instructions")
        opts.add_row("[D] Diff", "Show full diff from original")
        opts.add_row("[V] History", "Browse previous attempts")
        opts.add_row("[S] Section", "Edit a specific section")
        opts.add_row("[Q] Quit", "Discard and exit")
        console.print(opts)

        choice = console.input("\n[bold cyan]Choice (A/R/I/D/V/S/Q): [/bold cyan]").strip().upper()

        if choice == "A":
            from datetime import datetime

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"tailored_CGI_TechSupport_{ts}.txt"
            out = output_dir / fname
            out.write_text(result.tailored_text, encoding="utf-8")
            result.output_path = str(out)

            meta_path = out.with_suffix(".meta.json")
            meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

            console.print(f"\n[green]Saved: {out}[/green]")
            console.print(f"[dim]Attempt #{attempt} | Score: {result.match_score:.0%}[/dim]")

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
                    console,
                    config,
                    job,
                    resume,
                    tone_profile,
                    result.tailored_text,
                )

            if cover_letter_path:
                result.cover_letter_path = str(cover_letter_path)
            meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[green]Meta:  {meta_path}[/green]")
            break

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            user_instructions = ""
            try:
                with console.status("Tailoring resume with LLM..."):
                    result = await tailor_engine.refine(resume, result, "", job)
                result.attempt = attempt
                session.add_attempt(result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
                retry_choice = (
                    console.input("[bold cyan][R] Retry or [Q] Quit? [/bold cyan]").strip().upper()
                )
                if retry_choice == "Q":
                    break
                continue
            continue

        elif choice == "I":
            user_instructions = console.input("\n[bold]Instructions: [/bold]").strip()
            try:
                with console.status("Tailoring resume with LLM..."):
                    result = await tailor_engine.refine(resume, result, user_instructions, job)
                result.attempt = attempt
                session.add_attempt(result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
                retry_choice = (
                    console.input("[bold cyan][R] Retry or [Q] Quit? [/bold cyan]").strip().upper()
                )
                if retry_choice == "Q":
                    break
                continue
            continue

        elif choice == "D":
            _render_diff(console, session.original_text, result.tailored_text, max_lines=0)
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
            try:
                with console.status("Refining section..."):
                    result = await tailor_engine.refine(resume, result, user_instructions, job)
                result.attempt = attempt
                session.add_attempt(result)
            except Exception as exc:
                console.print(f"[red]LLM error: {exc}[/red]")
                retry_choice = (
                    console.input("[bold cyan][R] Retry or [Q] Quit? [/bold cyan]").strip().upper()
                )
                if retry_choice == "Q":
                    break
                continue
            continue

        elif choice == "Q":
            console.print("[yellow]Discarded.[/yellow]")
            break

        else:
            console.print("[red]Invalid choice.[/red]")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
