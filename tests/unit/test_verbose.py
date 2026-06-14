from pathlib import Path

from job_applicator.models import (
    ResumeParsingReport,
    VerboseReport,
)
from job_applicator.utils.verbose import VerboseReporter


def test_resume_parsing_report_defaults() -> None:
    r = ResumeParsingReport(source="resume.pdf")
    assert r.source == "resume.pdf"


def test_verbose_report_serializes() -> None:
    v = VerboseReport(command="ats-check", args={"resume": "r.pdf"})
    data = v.model_dump()
    assert data["command"] == "ats-check"
    assert data["args"]["resume"] == "r.pdf"


def test_reporter_collects_resume_info() -> None:
    reporter = VerboseReporter(command="ats-check", args={"resume": "r.pdf"}, config={})
    reporter.record_resume(
        source="r.pdf",
        ocr_mode="auto",
        text_length=1234,
        parsed_name="John",
        parsed_email="j@example.com",
        parsed_phone="555-1234",
        parsed_skills=["Python"],
        parsed_summary_preview="Summary...",
    )
    report = reporter.report
    assert report.resume is not None
    assert report.resume.parsed_name == "John"


def test_reporter_writes_log_file(tmp_path: Path) -> None:
    reporter = VerboseReporter(command="ats-check", args={}, config={})
    reporter.record_ats(score=1.0, is_compatible=True, checks=[], warnings=[], suggestions=[])
    log_path = tmp_path / "out.json"
    reporter.render(console=None, log_file=str(log_path))
    assert log_path.exists()


def test_reporter_collects_errors() -> None:
    reporter = VerboseReporter(command="ats-check", args={}, config={})
    reporter.record_error("something failed")
    assert reporter.report.errors == ["something failed"]
