#!/usr/bin/env python3
"""Tailor resume for CGI Technical Support Specialist (best match from report)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


async def main():
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
    from job_applicator.models import JobBoard, JobListing

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

    config = LLMConfig()
    tailor_engine = ResumeTailor(config)

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
            str(entry.get("section", "")),
            str(entry.get("label", "")),
            str(entry.get("start", "")),
            str(entry.get("end", "")),
        )
    console.print(audit_table)

    console.print(
        f"\n[dim]Date range: {audit.earliest_date} → {audit.latest_date}[/dim]"
    )

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
        confirm = console.input(
            "\n[bold cyan]Proceed anyway? (y/n): [/bold cyan]"
        ).strip().lower()
        if confirm != "y":
            console.print("[yellow]Aborted.[/yellow]")
            return False
    else:
        console.print("[green]Dates look coherent and current.[/green]")

    attempt = 0
    user_instructions = ""
    result = None

    while True:
        attempt += 1
        console.print(f"\n[bold blue]=== Attempt #{attempt} ===[/bold blue]")

        with console.status("Tailoring resume with LLM..."):
            if attempt == 1:
                result = await tailor_engine.tailor(resume, job, user_instructions)
            else:
                result = await tailor_engine.refine(resume, result, user_instructions, job)

        result.attempt = attempt

        console.print("\n[bold]Tailored Resume:[/bold]\n")
        console.print(Panel(result.tailored_text, title="Preview", border_style="cyan"))

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

        console.print("\n[bold]Changes:[/bold]")
        console.print(result.changes_summary)

        console.print("\n[bold]Options:[/bold]")
        opts = Table(show_header=False, box=None)
        opts.add_column("Key", style="cyan bold")
        opts.add_column("Description")
        opts.add_row("[A] Accept", "Save this version")
        opts.add_row("[R] Retry", "Regenerate")
        opts.add_row("[I] Input", "Give custom instructions")
        opts.add_row("[Q] Quit", "Discard and exit")
        console.print(opts)

        choice = console.input("\n[bold cyan]Choice (A/R/I/Q): [/bold cyan]").strip().upper()

        if choice == "A":
            from datetime import datetime

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)  # noqa: ASYNC240

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"tailored_CGI_TechSupport_{ts}.txt"
            out = output_dir / fname
            out.write_text(result.tailored_text, encoding="utf-8")
            result.output_path = str(out)

            meta_path = out.with_suffix(".meta.json")
            meta_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

            console.print(f"\n[green]Saved: {out}[/green]")
            console.print(f"[green]Meta:  {meta_path}[/green]")
            break

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            user_instructions = ""

        elif choice == "I":
            user_instructions = console.input("\n[bold]Instructions: [/bold]").strip()

        elif choice == "Q":
            console.print("[yellow]Discarded.[/yellow]")
            break

        else:
            console.print("[red]Invalid choice.[/red]")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
