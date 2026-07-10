"""Interactive résumé-tailoring workflow — the accept/retry/refine/section-edit loop
extracted from the `tailor` command.

Orchestration only: it mutates the ``TailorSession`` and writes artifacts on accept.
The shared cli helper ``_llm_with_retry`` is imported lazily (inside the function) to
avoid a cli <-> workflow import cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table

from job_applicator.documents.artifacts import (
    strip_markdown_bold,
    write_tailored,
    write_tailored_pdf,
)
from job_applicator.documents.job_category import detect_job_category
from job_applicator.exceptions import TailorIntegrityError
from job_applicator.models import Format
from job_applicator.utils.diff import render_diff
from job_applicator.utils.logging import get_logger
from job_applicator.workflows.cover_letter import _cover_letter_workflow

logger = get_logger("workflows.tailor")


@dataclass(frozen=True)
class PostTailorIntegrity:
    """Structured version of the post-tailor integrity display."""

    base_ats_compatible: bool
    tailored_ats_compatible: bool
    missing_contact: tuple[str, ...] = ()

    @property
    def blocks_auto_accept(self) -> bool:
        return (self.base_ats_compatible and not self.tailored_ats_compatible) or bool(
            self.missing_contact
        )


def _write_text_file(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


if TYPE_CHECKING:
    from rich.console import Console

    from job_applicator.config import AppSettings
    from job_applicator.documents.resume_tailor import ResumeTailor
    from job_applicator.documents.tone_detector import ToneProfile
    from job_applicator.models import (
        GroundingReport,
        JobListing,
        ResumeData,
        StyleGuide,
        TailoredResume,
        TailorSession,
    )
    from job_applicator.utils.verbose import VerboseReporter


def _print_grounding_report(console: Console, report: GroundingReport | None) -> None:
    """Surface the honesty check of the tailored text vs the base résumé (spec §6).

    Flags for human review — NEVER auto-removes a claim (this is the document of record; the user
    is its ground truth and the verifier has a measured precision residual). ``None`` means the
    verifier was unavailable (fail-safe) — surfaced as such, never as 'verified clean'.
    """
    if report is None:
        console.print("\n[dim]Grounding check not run for this version.[/dim]")
        return
    if report.clean:
        console.print("\n[green]✓ Grounding: every claim is supported by your résumé.[/green]")
        return
    lines = [
        f"• {c.claim}  [dim]({c.note or 'not in your résumé'})[/dim]" for c in report.unsupported
    ]
    lines += [f"• {gap}  [dim](could not verify — check it)[/dim]" for gap in report.coverage_gaps]
    n = len(report.unsupported) + len(report.coverage_gaps)
    console.print(
        Panel(
            "\n".join(lines),
            title=f"⚠  {n} claim(s) to REVIEW — not auto-removed; verify vs your résumé or fix V1",
            border_style="yellow",
        )
    )


def _compute_post_tailor_integrity(
    original_text: str, tailored_text: str
) -> tuple[PostTailorIntegrity, float, float] | None:
    """Compute cheap post-tailor integrity signals without printing."""
    try:
        from job_applicator.documents.ats_checker import ATSChecker
        from job_applicator.documents.resume import ResumeLoader

        loader = ResumeLoader()
        base = loader.parse_text(original_text)
        tailored = loader.parse_text(tailored_text)
        before_result = ATSChecker().check(base)
        after_result = ATSChecker().check(tailored)
        before = before_result.score
        after = after_result.score

        # Contact green-check: only verify fields the base actually exposes. Match leniently so a
        # faithful REFORMAT isn't flagged as a drop — compare the email without a parser-captured
        # trailing dot, and the phone by its national (last-10-digit) number so a +1 country-code
        # difference doesn't cry wolf. Errs toward NOT flagging (advisory; a false alarm erodes it).
        low = tailored_text.lower()
        tail_digits = re.sub(r"\D", "", tailored_text)
        missing: list[str] = []
        base_email = base.email.strip().rstrip(".").lower()
        if base_email and base_email not in low:
            missing.append("email")
        base_phone = re.sub(r"\D", "", base.phone)[-10:]  # national number, country-code-agnostic
        if len(base_phone) == 10 and base_phone not in tail_digits:
            missing.append("phone")
        return (
            PostTailorIntegrity(
                base_ats_compatible=before_result.is_compatible,
                tailored_ats_compatible=after_result.is_compatible,
                missing_contact=tuple(missing),
            ),
            before,
            after,
        )
    except Exception as exc:
        # Advisory surface only — a parse/ATS hiccup (or an unparseable draft) must never abort the
        # tailor flow; skip the line with a debug note rather than crashing the review loop.
        logger.debug("Post-tailor integrity surface skipped: %s", exc)
        return None


def _print_post_tailor_integrity(
    console: Console, original_text: str, tailored_text: str
) -> PostTailorIntegrity | None:
    """Surface two cheap post-tailor integrity signals per version.

    Both are SURFACED for review, never gates for interactive review — consistent with the grounding
    report. Non-interactive auto-accept uses the returned status to fail closed when a previously
    ATS-compatible base résumé becomes incompatible or contact details disappear.
    """
    computed = _compute_post_tailor_integrity(original_text, tailored_text)
    if computed is None:
        return None
    integrity, before, after = computed
    mark = "green" if after >= before else "yellow"
    console.print(f"\n[bold]Tailored ATS:[/bold] [{mark}]{before:.0%} → {after:.0%}[/{mark}]")
    if integrity.missing_contact:
        console.print(
            f"[yellow]⚠ Contact: your {', '.join(integrity.missing_contact)} from the base résumé "
            "is missing from the tailored output — verify before sending.[/yellow]"
        )
    else:
        console.print("[green]✓ Contact preserved in the tailored output.[/green]")
    return integrity


def grounding_failure_summary(report: GroundingReport) -> str:
    """Summarize residual grounding issues with enough text to fix the prompt/output."""
    issue_count = len(report.unsupported) + len(report.coverage_gaps)
    message = f"{issue_count} unsupported or unchecked claim(s)"
    unsupported = "; ".join(check.claim for check in report.unsupported[:3])
    if unsupported:
        message += f"; unsupported: {unsupported}"
    gaps = "; ".join(report.coverage_gaps[:3])
    if gaps:
        message += f"; unchecked: {gaps}"
    return message


def assert_tailored_auto_saveable(result: TailoredResume, original_text: str) -> None:
    """Fail closed before an automated tailored-CV save."""
    grounding_report = result.grounding_report
    if grounding_report is None:
        raise TailorIntegrityError(
            "Automated save refused this tailored résumé because grounding verification did not "
            "complete. Review manually, or retry after fixing the verifier."
        )
    if not grounding_report.clean:
        raise TailorIntegrityError(
            "Automated save refused this tailored résumé because grounding verification found "
            f"{grounding_failure_summary(grounding_report)}. Review manually, or adjust the "
            "tailoring prompt."
        )
    computed = _compute_post_tailor_integrity(original_text, result.tailored_text)
    if computed is None:
        return
    integrity, _before, _after = computed
    if integrity.blocks_auto_accept:
        reasons: list[str] = []
        if integrity.base_ats_compatible and not integrity.tailored_ats_compatible:
            reasons.append("tailored résumé became ATS-incompatible")
        if integrity.missing_contact:
            reasons.append(f"missing contact: {', '.join(integrity.missing_contact)}")
        raise TailorIntegrityError(
            "Automated save refused this tailored résumé because "
            + "; ".join(reasons)
            + ". Review manually, or adjust the tailoring prompt."
        )


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
    *,
    output_format: Format = Format.TXT,
    resume_template: str = "modern",
    cover_letter_template: str = "modern",
    category: str | None = None,
) -> TailoredResume | None:
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
                strip_markdown_bold(result.tailored_text),
                title="Tailored Resume",
                border_style="cyan",
            )
        )
        render_diff(
            console, session.original_text, strip_markdown_bold(result.tailored_text), max_lines=30
        )

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
        console.print(strip_markdown_bold(result.changes_summary))

        # Every version carries its own source-overlay integrity report. Refinement regenerates the
        # summary from the original source and reruns the same digest/citation checks.
        _print_grounding_report(console, result.grounding_report)
        _print_post_tailor_integrity(console, session.original_text, result.tailored_text)

        console.print("\n[bold]What would you like to do?[/bold]")
        action_table = Table(show_header=False, box=None)
        action_table.add_column("Option", style="cyan bold")
        action_table.add_column("Description")
        action_table.add_row("[A] Accept", "Save this version as final")
        action_table.add_row("[R] Retry", "Regenerate the summary")
        action_table.add_row("[I] Input", "Give summary focus or voice instructions")
        action_table.add_row("[D] Diff", "Show changes from original resume")
        action_table.add_row("[V] History", "Browse previous attempts")
        action_table.add_row("[S] Summary", "Refine the generated summary")
        action_table.add_row("[Q] Quit", "Discard and exit")
        console.print(action_table)

        if yes:
            try:
                assert_tailored_auto_saveable(result, session.original_text)
            except TailorIntegrityError as exc:
                msg = str(exc).replace("Automated save refused", "--yes refused to auto-accept")
                raise TailorIntegrityError(msg) from exc
            console.print("\n[dim]--yes: accepting this version automatically.[/dim]")
            choice = "A"
        else:
            choice = (
                console.input("\n[bold cyan]Your choice (A/R/I/D/V/S/Q): [/bold cyan]")
                .strip()
                .upper()
            )

        if choice == "A":
            output_dir = settings.ensure_output_dir()
            when = datetime.now()
            effective_category = category or detect_job_category(job)

            if output_format == Format.TXT:
                resume_path, meta_path = write_tailored(output_dir, result, when=when)
                result.output_path = resume_path
                files_written = [resume_path]
            elif output_format == Format.PDF:
                pdf_path = await write_tailored_pdf(
                    output_dir,
                    result,
                    settings,
                    template=resume_template,
                    category=effective_category,
                    when=when,
                )
                result.output_path = str(pdf_path)
                result.pdf_path = str(pdf_path)
                files_written = [str(pdf_path)]
            else:  # both
                resume_path, meta_path = write_tailored(output_dir, result, when=when)
                pdf_path = await write_tailored_pdf(
                    output_dir,
                    result,
                    settings,
                    template=resume_template,
                    category=effective_category,
                    when=when,
                    write_meta=False,
                )
                result.output_path = resume_path
                result.pdf_path = str(pdf_path)
                # Update the text sidecar to reference the PDF.
                _write_text_file(meta_path, result.model_dump_json(indent=2))
                files_written = [resume_path, str(pdf_path)]

            if reporter:
                reporter.record_io(files_written=files_written)

            console.print(f"\n[green]Tailored resume saved: {result.output_path}[/green]")
            console.print(f"[dim]Attempt #{attempt} | Score: {result.match_score:.0%}[/dim]")

            # Offer cover letter generation
            cover_letter_path: Path | None = None
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
                    output_format=output_format,
                    template=cover_letter_template,
                    category=category,
                )

            # Write resume meta.json (with or without cover_letter_path)
            if cover_letter_path:
                result.cover_letter_path = str(cover_letter_path)
            final_meta_path = Path(result.output_path).with_suffix(".meta.json")
            _write_text_file(final_meta_path, result.model_dump_json(indent=2))
            console.print(f"[green]Metadata saved: {final_meta_path}[/green]")

            return result

        elif choice == "R":
            console.print("[yellow]Regenerating...[/yellow]")
            user_instructions = ""
            refined: TailoredResume | None = await _llm_with_retry(
                console,
                partial(
                    tailor_engine.refine_verified,
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
                "\n[bold]Enter summary instructions (e.g., 'emphasize "
                "customer service evidence', 'use a more concise voice'): "
                "[/bold]"
            ).strip()
            if not user_instructions:
                console.print("[yellow]No instructions provided, retrying.[/yellow]")
            refined = await _llm_with_retry(
                console,
                partial(
                    tailor_engine.refine_verified,
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
            # Strip markdown bold the same way the inline preview diff does (L122) — the raw
            # tailored text carries `**header**` markers the PDF formatter consumes, not the human.
            render_diff(
                console,
                session.original_text,
                strip_markdown_bold(result.tailored_text),
                max_lines=0,
            )
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
                # Strip bold before slicing so the 60-char window holds content, not `**` markers.
                preview = strip_markdown_bold(att.tailored_text)[:60].replace("\n", " ")
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
            from job_applicator.documents.resume import section_header
            from job_applicator.documents.resume_tailor import parse_sections

            sections = [
                section
                for section in parse_sections(result.tailored_text)
                if section_header(section.name) == "Summary"
            ]
            if len(sections) != 1:
                console.print(
                    "[yellow]Could not identify exactly one summary section. "
                    "The source résumé body remains read-only.[/yellow]"
                )
                continue

            target_section = sections[0]
            console.print(f"\n[dim]Editing generated summary: {target_section.name}[/dim]")
            # Strip bold before slicing — the section body keeps raw `**` from parse_sections.
            console.print(f"[dim]{strip_markdown_bold(target_section.text)[:200]}...[/dim]\n")

            sec_instructions = console.input("[bold]Instructions for this section: [/bold]").strip()
            if not sec_instructions:
                console.print("[yellow]No instructions provided.[/yellow]")
                continue

            user_instructions = sec_instructions
            refined = await _llm_with_retry(
                console,
                partial(
                    tailor_engine.refine_verified,
                    resume_data,
                    result,
                    user_instructions,
                    job,
                    tone_profile=tone_profile,
                    style_guide=style,
                ),
                "Refining summary...",
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

    return None
