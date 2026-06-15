"""Structured observability reporter for CLI commands."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from job_applicator.models import (
    ATSReport,
    IOReport,
    LLMReport,
    MatchReport,
    ResumeParsingReport,
    TailoringReport,
    VerboseReport,
)


class VerboseReporter:
    """Collect and render structured observability reports."""

    def __init__(self, command: str, args: dict[str, Any], config: dict[str, Any]) -> None:
        self._started_at = datetime.now()
        self.report = VerboseReport(
            command=command,
            args=args,
            config=config,
        )

    def record_resume(
        self,
        *,
        source: str,
        ocr_mode: str = "auto",
        text_length: int = 0,
        parsed_name: str = "",
        parsed_email: str = "",
        parsed_phone: str = "",
        parsed_skills: list[str] | None = None,
        parsed_summary_preview: str = "",
        warnings: list[str] | None = None,
    ) -> None:
        self.report.resume = ResumeParsingReport(
            source=source,
            ocr_mode=ocr_mode,
            text_length=text_length,
            parsed_name=parsed_name,
            parsed_email=parsed_email,
            parsed_phone=parsed_phone,
            parsed_skills=parsed_skills or [],
            parsed_summary_preview=parsed_summary_preview,
            warnings=warnings or [],
        )

    def record_ats(
        self,
        *,
        score: float,
        is_compatible: bool,
        checks: list[dict[str, Any]],
        warnings: list[str],
        suggestions: list[str],
    ) -> None:
        self.report.ats = ATSReport(
            score=score,
            is_compatible=is_compatible,
            checks=checks,
            warnings=warnings,
            suggestions=suggestions,
        )

    def record_match(
        self,
        *,
        embedding_model: str,
        device: str,
        load_time_ms: int,
        results: list[dict[str, Any]],
    ) -> None:
        self.report.match = MatchReport(
            embedding_model=embedding_model,
            device=device,
            load_time_ms=load_time_ms,
            job_count=len(results),
            results=results,
        )

    def record_llm_call(
        self,
        *,
        model: str,
        endpoint: str,
        prompt_tokens: int | None = None,
        response_tokens: int | None = None,
        temperature: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.report.llm is None:
            self.report.llm = LLMReport(model=model, endpoint=endpoint)
        self.report.llm.calls.append(
            {
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "temperature": temperature,
                "details": details or {},
            }
        )

    def record_tailoring(
        self,
        *,
        tone: str = "",
        tone_confidence: float = 0.0,
        pre_match_score: float | None = None,
        attempts: int = 0,
        ats_before: float = 0.0,
        ats_after: float = 0.0,
        hallucination_actions: list[str] | None = None,
        changes_summary: str = "",
    ) -> None:
        self.report.tailoring = TailoringReport(
            tone=tone,
            tone_confidence=tone_confidence,
            pre_match_score=pre_match_score,
            attempts=attempts,
            ats_before=ats_before,
            ats_after=ats_after,
            hallucination_actions=hallucination_actions or [],
            changes_summary=changes_summary,
        )

    def record_io(
        self,
        *,
        files_written: list[str] | None = None,
        files_read: list[str] | None = None,
        batch_summary_path: str | None = None,
    ) -> None:
        if self.report.io is None:
            self.report.io = IOReport()
        if files_written:
            self.report.io.files_written.extend(files_written)
        if files_read:
            self.report.io.files_read.extend(files_read)
        if batch_summary_path:
            self.report.io.batch_summary_path = batch_summary_path

    def record_error(self, message: str) -> None:
        self.report.errors.append(message)

    def _finalize(self) -> None:
        self.report.duration_ms = int((datetime.now() - self._started_at).total_seconds() * 1000)

    def render(self, console: Console | None, log_file: str | None = None) -> None:
        self._finalize()
        if console is not None:
            self._render_terminal(console)
        if log_file:
            try:
                Path(log_file).write_text(self.report.model_dump_json(indent=2), encoding="utf-8")
            except (IsADirectoryError, PermissionError, OSError) as exc:
                if console is not None:
                    console.print(f"[yellow]Warning: Could not write verbose log: {exc}[/yellow]")

    def _render_terminal(self, console: Console) -> None:
        table = Table(title="Observability Report")
        table.add_column("Section", style="cyan")
        table.add_column("Value")

        table.add_row("Command", self.report.command)
        table.add_row("Duration", f"{self.report.duration_ms} ms")

        if self.report.resume:
            r = self.report.resume
            table.add_row(
                "Resume",
                f"{r.source} | text={r.text_length} | skills={len(r.parsed_skills)}",
            )

        if self.report.ats:
            a = self.report.ats
            status = "PASS" if a.is_compatible else "FAIL"
            table.add_row("ATS", f"{status} ({a.score:.0%})")

        if self.report.match:
            m = self.report.match
            table.add_row("Match", f"{m.job_count} jobs | model={m.embedding_model}")

        if self.report.tailoring:
            t = self.report.tailoring
            table.add_row("Tailoring", f"{t.attempts} attempt(s) | tone={t.tone}")

        if self.report.io:
            io = self.report.io
            table.add_row("I/O", f"written={len(io.files_written)} read={len(io.files_read)}")

        if self.report.errors:
            table.add_row("Errors", "; ".join(self.report.errors))

        console.print(Panel(table, title="Verbose Report", expand=False))
