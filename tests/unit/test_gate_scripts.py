"""Tests for local quality-gate helper scripts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import check_matcher_gate_required as matcher_gate
from scripts import eval_document_quality, eval_matching
from typer.testing import CliRunner


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


def _quality_resume() -> str:
    return (
        "John Doe\njohn@example.com\n514-555-0199\n\n"
        "Experience\nSecurity Support Analyst, Acme, 2021 - Present\n"
        "Used Python, SQL, Linux, SIEM, incident response, alert triage, IAM, and log analysis "
        "workflows to investigate authentication failures, document evidence, and coordinate "
        "escalations across support and infrastructure teams. Improved recurring monitoring "
        "checks through scripts, runbooks, and clear handoff notes that helped analysts compare "
        "patterns across shifts.\n\n"
        "Technical Support Analyst, Beta, 2019 - 2021\n"
        "Resolved workstation, account, network, and access issues for users while documenting "
        "repeatable fixes. Partnered with infrastructure staff on monitoring alerts and "
        "service-impacting incidents. Built small Python automation helpers to reduce manual "
        "follow-up and reviewed Linux logs before escalating ambiguous security symptoms.\n\n"
        "Education\nCertificate in Cybersecurity, 2024\n\n"
        "Skills\nPython, SQL, Linux, SIEM, incident response, IAM, log analysis, alert triage, "
        "documentation, support, networking, automation, troubleshooting, monitoring, escalation, "
        "communication."
    )


def _quality_cover_letter() -> str:
    return (
        "Dear Acme Security Team,\n\n"
        "I bring practical Python, Linux, SIEM, and incident response experience from support "
        "work where I triaged alerts, reviewed authentication failures, documented root causes, "
        "and coordinated follow-up with technical teams. That background maps directly to the "
        "Security Analyst role's need for reliable investigation habits and clear communication "
        "under pressure.\n\n"
        "In recent projects I used Python and SQL to automate checks, review Linux logs, and make "
        "recurring operational work easier to track. I also supported IAM and access issues, "
        "which gave me a careful evidence-first approach to alert triage and escalation decisions "
        "when symptoms were incomplete or noisy.\n\n"
        "I would welcome the chance to discuss how that mix of support discipline, automation, "
        "SIEM analysis, and incident response practice can help Acme move investigations forward "
        "with less friction while keeping handoffs accurate and useful.\n\n"
        "Sincerely,\n"
        "John Doe"
    )


def test_document_quality_missing_private_set_is_skip_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.jsonl"

    assert eval_document_quality._run_packet_set(packet_set=missing, required=False) == 0
    assert "not certified" in capsys.readouterr().out


def test_document_quality_missing_private_set_fails_when_required(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.jsonl"

    assert eval_document_quality._run_packet_set(packet_set=missing, required=True) == 2
    assert "not certified" in capsys.readouterr().out


def test_document_quality_private_packet_set_scores_complete_packets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    packet_set = tmp_path / "packet-set.jsonl"
    resume.write_text(_quality_resume(), encoding="utf-8")
    cover.write_text(_quality_cover_letter(), encoding="utf-8")
    packet_set.write_text(
        (
            '{"id":"acme-security","resume_path":"resume.txt","cover_letter_path":"cover.txt",'
            '"applicant_name":"John Doe","job_title":"Security Analyst","company":"Acme",'
            '"keywords":["Python","Linux","SIEM","incident response","IAM","alert triage"]}\n'
        ),
        encoding="utf-8",
    )

    assert eval_document_quality._run_packet_set(packet_set=packet_set, required=True) == 0
    output = capsys.readouterr().out

    assert "PASS document packet quality" in output
    assert "PASS acme-security" in output


def test_document_quality_private_packet_set_fails_weak_packets(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    packet_set = tmp_path / "packet-set.jsonl"
    resume.write_text("Jane\nNo sections yet", encoding="utf-8")
    cover.write_text("- TODO: write letter\n\nRegards,\nYour Name", encoding="utf-8")
    packet_set.write_text(
        (
            '{"id":"weak","resume_path":"resume.txt","cover_letter_path":"cover.txt",'
            '"applicant_name":"Jane Doe","keywords":["Python","SIEM"]}\n'
        ),
        encoding="utf-8",
    )

    reports = eval_document_quality.assess_packet_set(packet_set)

    assert eval_document_quality._run_packet_set(packet_set=packet_set, required=True) == 1
    assert reports[0].overall < 3
    assert any("usefulness score" in item for item in reports[0].failures)


def test_document_quality_private_packet_set_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    packet_set = tmp_path / "packet-set.jsonl"
    resume.write_text(_quality_resume(), encoding="utf-8")
    cover.write_text(_quality_cover_letter(), encoding="utf-8")
    packet_set.write_text(
        (
            '{"id":"acme-security","resume_path":"resume.txt","cover_letter_path":"cover.txt",'
            '"applicant_name":"John Doe","job_title":"Security Analyst","company":"Acme",'
            '"keywords":["Python","Linux","SIEM","incident response","IAM","alert triage"]}\n'
        ),
        encoding="utf-8",
    )

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set, required=True, json_output=True
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["passed"] is True
    assert payload["count"] == 1
    assert set(payload["dimension_means"]) == {
        "usefulness",
        "specificity",
        "coherence",
        "writing_quality",
        "formatting_polish",
    }
    assert payload["packets"][0]["packet_id"] == "acme-security"


def test_document_quality_private_packet_set_fails_incoherent_packet(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    packet_set = tmp_path / "packet-set.jsonl"
    resume.write_text(_quality_resume(), encoding="utf-8")
    cover.write_text(
        (
            "Dear Acme Security Team,\n\n"
            "I am applying for the Security Analyst role at Acme. My background is centered on "
            "guest relations, event coordination, vendor follow-up, and service planning, with a "
            "focus on calm communication and dependable handoffs.\n\n"
            "In previous work I helped teams stay organized, maintain schedules, and keep clients "
            "informed when priorities changed. That experience would help me contribute to a team "
            "that values patience, preparation, and clear written updates.\n\n"
            "I would welcome the opportunity to discuss how this service background can support "
            "your team.\n\n"
            "Sincerely,\nJohn Doe"
        ),
        encoding="utf-8",
    )
    packet_set.write_text(
        (
            '{"id":"incoherent","resume_path":"resume.txt","cover_letter_path":"cover.txt",'
            '"applicant_name":"John Doe","job_title":"Security Analyst","company":"Acme",'
            '"keywords":["Python","Linux","SIEM","incident response","IAM","alert triage"]}\n'
        ),
        encoding="utf-8",
    )

    report = eval_document_quality.assess_packet_set(packet_set)[0]

    assert report.dimensions["coherence"] < 3.0
    assert any("coherence score" in item for item in report.failures)


def test_document_quality_company_alias_counts_for_target_mention() -> None:
    assert (
        eval_document_quality._target_mention_score(
            "I am applying for the IT On-site Support Technician role at WSP in Montreal.",
            job_title="IT On-site Support Technician",
            company="WSP in Canada",
        )
        == 1.0
    )


def test_document_quality_cli_single_artifact_json(tmp_path: Path) -> None:
    import json

    from job_applicator import cli

    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    resume.write_text(_quality_resume(), encoding="utf-8")
    cover.write_text(_quality_cover_letter(), encoding="utf-8")

    result = CliRunner().invoke(
        cli.app,
        [
            "document-quality",
            "--resume",
            str(resume),
            "--cover-letter",
            str(cover),
            "--applicant-name",
            "John Doe",
            "--keyword",
            "Python",
            "--keyword",
            "SIEM",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert {report["kind"] for report in payload} == {"resume", "cover_letter"}
    assert all(report["passed"] for report in payload)


def test_document_quality_cli_missing_optional_packet_set_skips(tmp_path: Path) -> None:
    from job_applicator import cli

    missing = tmp_path / "missing.jsonl"

    result = CliRunner().invoke(cli.app, ["document-quality", "--packet-set", str(missing)])

    assert result.exit_code == 0
    assert "not certified" in result.output


def test_document_quality_cli_missing_required_packet_set_fails(tmp_path: Path) -> None:
    from job_applicator import cli

    missing = tmp_path / "missing.jsonl"

    result = CliRunner().invoke(
        cli.app, ["document-quality", "--packet-set", str(missing), "--required"]
    )

    assert result.exit_code == 2
    assert "not certified" in result.output


def test_document_quality_cli_private_packet_set_json_output(tmp_path: Path) -> None:
    import json

    from job_applicator import cli

    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    packet_set = tmp_path / "packet-set.jsonl"
    resume.write_text(_quality_resume(), encoding="utf-8")
    cover.write_text(_quality_cover_letter(), encoding="utf-8")
    packet_set.write_text(
        (
            '{"id":"acme-security","resume_path":"resume.txt","cover_letter_path":"cover.txt",'
            '"applicant_name":"John Doe","job_title":"Security Analyst","company":"Acme",'
            '"keywords":["Python","Linux","SIEM","incident response","IAM","alert triage"],'
            '"coherence_terms":["Python","SIEM","incident response"]}\n'
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "document-quality",
            "--packet-set",
            str(packet_set),
            "--required",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert payload["packets"][0]["dimensions"]["coherence"] >= 3.0


def test_document_quality_cli_private_packet_set_uses_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from job_applicator import cli

    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    packet_set = tmp_path / "packet-set.jsonl"
    resume.write_text(_quality_resume(), encoding="utf-8")
    cover.write_text(_quality_cover_letter(), encoding="utf-8")
    packet_set.write_text(
        (
            '{"id":"acme-security","resume_path":"resume.txt","cover_letter_path":"cover.txt",'
            '"applicant_name":"John Doe","job_title":"Security Analyst","company":"Acme",'
            '"keywords":["Python","Linux","SIEM","incident response","IAM","alert triage"]}\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCUMENT_QUALITY_SET", str(packet_set))

    result = CliRunner().invoke(
        cli.app,
        ["document-quality", "--private-packet-set", "--required", "--json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["packet_set"] == str(packet_set)
