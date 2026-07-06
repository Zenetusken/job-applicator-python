"""Tests for local quality-gate helper scripts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import check_matcher_gate_required as matcher_gate
from scripts import eval_document_quality, eval_matching


def test_matcher_gate_detects_sensitive_paths() -> None:
    sensitive = matcher_gate.matcher_sensitive(
        [
            "README.md",
            "src/job_applicator/embeddings/matching.py",
            "src/job_applicator/skills/normalization.py",
        ]
    )

    assert sensitive == [
        "src/job_applicator/embeddings/matching.py",
        "src/job_applicator/skills/normalization.py",
    ]


def test_matcher_gate_ignores_unrelated_paths() -> None:
    assert matcher_gate.matcher_sensitive(["README.md", "src/job_applicator/cli.py"]) == []


def test_matcher_gate_includes_untracked_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        if cmd[:3] == ["git", "diff", "--name-only"]:
            return SimpleNamespace(returncode=0, stdout="README.md\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout="src/job_applicator/skills/new_taxonomy.py\n",
            stderr="",
        )

    monkeypatch.setattr(matcher_gate.subprocess, "run", fake_run)

    paths = matcher_gate.changed_paths("HEAD")

    assert "README.md" in paths
    assert "src/job_applicator/skills/new_taxonomy.py" in paths


def test_eval_matching_missing_gold_set_is_skip_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.csv"
    monkeypatch.setenv("GOLD_SET_CSV", str(missing))

    assert eval_matching._run(required=False) == 0
    assert "not certified" in capsys.readouterr().out


def test_eval_matching_missing_gold_set_fails_when_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.csv"
    monkeypatch.setenv("GOLD_SET_CSV", str(missing))

    assert eval_matching._run(required=True) == 2
    assert "not certified" in capsys.readouterr().out


def test_document_quality_accepts_complete_packet() -> None:
    cover = (
        "Dear Hiring Manager,\n\n"
        "I bring practical Python and incident-response experience from support work, where I "
        "triaged alerts, documented root causes, and coordinated follow-up with technical teams. "
        "That background maps well to the role's need for reliable investigation habits and clear "
        "communication under pressure.\n\n"
        "In recent projects I used Python, SQL, and Linux tooling to automate checks, review logs, "
        "and make recurring operational work easier to track. I focus on concrete evidence, "
        "careful "
        "handoffs, and improvements that reduce avoidable repeat issues.\n\n"
        "I would welcome the chance to discuss how that mix of support discipline, automation, and "
        "security-minded analysis can help your team move investigations forward with less "
        "noise.\n\n"
        "Sincerely,\n"
        "John Doe"
    )
    resume = (
        "John Doe\njohn@example.com\n514-555-0199\n\n"
        "Experience\nSecurity Support Analyst, Acme, 2021 - Present\n"
        "Used Python, SQL, Linux, SIEM, and incident response workflows to triage alerts, document "
        "evidence, and coordinate escalations across support and infrastructure teams.\n"
        "Improved recurring operational checks through scripts and clear runbooks. Reviewed "
        "authentication failures, endpoint alerts, and network symptoms before escalating. "
        "Maintained concise notes so recurring issues could be compared across shifts.\n\n"
        "Technical Support Analyst, Beta, 2019 - 2021\n"
        "Resolved workstation, account, and access issues for users while documenting repeatable "
        "fixes. Partnered with infrastructure staff on monitoring alerts and service-impacting "
        "incidents. Built small automation helpers to reduce manual follow-up.\n\n"
        "Education\nCertificate in Cybersecurity, 2024\n\n"
        "Skills\nPython, SQL, Linux, SIEM, incident response, log analysis, documentation, "
        "support, "
        "networking, automation, troubleshooting, monitoring, escalation, communication."
    )

    cover_report = eval_document_quality.assess_cover_letter(cover, applicant_name="John Doe")
    resume_report = eval_document_quality.assess_resume(
        resume, keywords=["Python", "SIEM", "incident response", "Linux"]
    )

    assert cover_report.passed
    assert resume_report.passed


def test_document_quality_rejects_bad_packet() -> None:
    cover = "- TODO: write letter\n\nRegards,\nYour Name"
    resume = "Jane\nNo sections yet"

    cover_report = eval_document_quality.assess_cover_letter(cover, applicant_name="Jane Doe")
    resume_report = eval_document_quality.assess_resume(resume, keywords=["Python"])

    assert not cover_report.passed
    assert not resume_report.passed
    assert any("placeholder" in item for item in cover_report.failures)
    assert any("email" in item for item in resume_report.failures)
