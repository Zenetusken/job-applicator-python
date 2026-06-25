"""Per-job apply workflow — the apply loop + results display extracted from the `apply`
command.

Applies each matched job (dry-run by default), honoring the daily cap and the local
already-applied state, then renders the per-job results. ``typer.Exit`` propagates the
--validate failure exactly as the original command body did.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import TYPE_CHECKING

import typer
from rich.markup import escape
from rich.table import Table

from job_applicator.state import ApplicationState

if TYPE_CHECKING:
    from rich.console import Console

    from job_applicator.applicators.base import BaseApplicator
    from job_applicator.config import AppSettings
    from job_applicator.models import JobListing
    from job_applicator.utils.verbose import VerboseReporter


async def _apply_to_jobs(
    jobs: list[JobListing],
    applicator: BaseApplicator,
    cover_letters: dict[str, str],
    settings: AppSettings,
    site: str,
    limit: int,
    *,
    submit: bool,
    validate: bool,
    as_json: bool,
    console: Console,
    reporter: VerboseReporter | None,
    cover_letter_pdf_paths: dict[str, str] | None = None,
) -> None:
    """Apply to each job (dry-run unless ``submit``) and render results."""
    from job_applicator.models import ApplicationResult, ApplicationStatus

    state = ApplicationState()
    daily_cap = settings.target.max_applications_per_day if submit else 0

    if submit:
        today_count = state.count_today(board=site)
        if today_count >= daily_cap:
            console.print(
                f"[yellow]Daily application cap reached ({today_count}/{daily_cap}). "
                "Skipping apply loop.[/yellow]"
            )
            return

    app_results: list[ApplicationResult] = []
    for job in jobs[:limit]:
        job_url = str(job.url)
        if submit and state.has_applied(
            job_url,
            statuses={ApplicationStatus.SUBMITTED, ApplicationStatus.ALREADY_APPLIED},
        ):
            console.print(f"[dim]Skipping {job.title} at {job.company} — already applied.[/dim]")
            app_results.append(
                ApplicationResult(
                    job=job,
                    status=ApplicationStatus.ALREADY_APPLIED,
                    notes=(
                        "Skipped by local state store (previous submitted/already-applied record)."
                    ),
                )
            )
            continue

        if submit:
            today_count = state.count_today(board=site)
            if today_count >= daily_cap:
                console.print(
                    f"[yellow]Daily application cap reached ({today_count}/{daily_cap}). "
                    "Stopping.[/yellow]"
                )
                break

        with console.status(f"Applying to {job.title} at {job.company}..."):
            job_letter = cover_letters.get(job_url)
            ar: ApplicationResult = await applicator.apply(job, job_letter, submit=submit)
            app_results.append(ar)
            if submit:
                state.record(ar)

    if reporter and app_results:
        written: list[str] = []
        if cover_letter_pdf_paths:
            written.extend(path for path in cover_letter_pdf_paths.values() if path)
        reporter.record_io(files_written=written)

    # Display results
    validation_failed = any(
        r.dry_run is not None and not r.dry_run.reached_submit for r in app_results
    )

    if as_json:
        import json

        output = [
            {
                "job": r.job.title,
                "company": r.job.company,
                "status": r.status.value,
                "error": r.error_message,
                "notes": r.notes,
                "cover_letter": r.cover_letter,
                "cover_letter_pdf_path": cover_letter_pdf_paths.get(str(r.job.url))
                if cover_letter_pdf_paths
                else None,
                "dry_run": r.dry_run.model_dump() if r.dry_run else None,
            }
            for r in app_results
        ]
        sys.stdout.write(json.dumps(output, indent=2) + "\n")
    else:
        table = Table(title="Application Results")
        table.add_column("Job", style="cyan")
        table.add_column("Company", style="green")
        table.add_column("Status")
        table.add_column("Notes")

        for r in app_results:
            status_style = {
                "submitted": "green",
                "failed": "red",
                "skipped": "yellow",
                "already_applied": "magenta",
                "pending": "blue",
            }.get(r.status.value, "white")
            note_parts: list[str] = []
            if r.dry_run:
                reached = "✓" if r.dry_run.reached_submit else "✗"
                note_parts.append(f"[submit {reached}]")
            if r.cover_letter:
                note_parts.append(f"[cover letter: {len(r.cover_letter)} chars]")
            job_url = str(r.job.url)
            if cover_letter_pdf_paths and job_url in cover_letter_pdf_paths:
                note_parts.append("[PDF cover letter]")
            if r.error_message:
                note_parts.append(r.error_message)
            elif r.notes:
                note_parts.append(r.notes)
            notes = escape(" ".join(note_parts))
            table.add_row(
                r.job.title,
                r.job.company,
                f"[{status_style}]{r.status.value}[/{status_style}]",
                notes,
            )

        console.print(table)
        # Count every status (incl. already_applied) so the summary
        # never silently under-reports outcomes.
        counts = Counter(r.status.value for r in app_results)
        summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        console.print(f"\n{summary}")

    if validate and validation_failed:
        console.print(
            "[red]Validation failed: one or more dry runs did not reach the Submit step.[/red]"
        )
        raise typer.Exit(1)
