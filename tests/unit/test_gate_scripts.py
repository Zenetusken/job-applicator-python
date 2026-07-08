"""Tests for local quality-gate helper scripts."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
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


def _write_quality_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    resume.write_text(_quality_resume(), encoding="utf-8")
    cover.write_text(_quality_cover_letter(), encoding="utf-8")
    return resume, cover


def _french_quality_cover_letter() -> str:
    return (
        "Bonjour à l'équipe de recrutement d'Acme.\n\n"
        "Je vous propose ma candidature pour le poste d'analyste sécurité. Mon parcours combine "
        "plus de dix ans en gestion des opérations, résolution de problèmes clients, triage et "
        "escalade, avec une formation récente en cybersécurité opérationnelle. Cette expérience "
        "me permet de documenter les incidents avec rigueur et de communiquer clairement sous "
        "pression.\n\n"
        "Dans mes fonctions précédentes, j'ai coordonné des demandes urgentes, traité des "
        "problèmes techniques de première ligne et assuré des suivis fiables entre les équipes. "
        "Ma formation couvre le SIEM, les opérations SOC, la réponse aux incidents, Linux et les "
        "réseaux, ce qui correspond directement aux besoins du rôle.\n\n"
        "Je serais heureux de discuter de la façon dont cette combinaison de discipline "
        "opérationnelle, de triage et de documentation peut aider Acme à améliorer ses enquêtes "
        "et ses transferts d'information.\n\n"
        "Cordialement,\n"
        "John Doe"
    )


def _write_french_quality_artifacts(
    tmp_path: Path,
    *,
    profile: str,
    education_note: str,
) -> tuple[Path, Path]:
    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    resume.write_text(
        (
            "John Doe\n"
            "Montréal, QC | 514-555-0100 | john@example.test\n\n"
            "Profil\n"
            f"{profile}\n\n"
            "Compétences\n"
            "SIEM, SOC monitoring, incident response, Linux, Python, TCP/IP, ticketing, "
            "escalation, Microsoft 365\n\n"
            "Expérience\n"
            "• Géré les opérations quotidiennes, le triage des demandes et les escalades avec "
            "des équipes internes et des clients.\n"
            "• Fourni un support technique niveau 1 par téléphone, chat et courriel pour des "
            "problèmes de connectivité, de signal et de site web.\n"
            "• Documenté les incidents, coordonné les suivis et maintenu une résolution en "
            "premier appel de 95 % lorsque les procédures le permettaient.\n\n"
            "Formation\n"
            "Certificat universitaire — Analyse et cybersécurité opérationnelle 2024 - Présent\n"
            "Northbridge Technical Institute\n"
            f"{education_note}\n\n"
            "Langues\n"
            "Français et anglais courants; espagnol.\n"
        ),
        encoding="utf-8",
    )
    cover.write_text(_french_quality_cover_letter(), encoding="utf-8")
    return resume, cover


def _quality_packet_record(
    *,
    packet_id: str,
    category: str = "support",
    language: str = "en",
    generated_at: str | None = None,
    min_dimension_score: float | None = None,
    min_overall_score: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "id": packet_id,
        "resume_path": "resume.txt",
        "cover_letter_path": "cover.txt",
        "applicant_name": "John Doe",
        "job_title": "Security Analyst",
        "company": "Acme",
        "keywords": ["Python", "Linux", "SIEM", "incident response", "IAM", "alert triage"],
        "coherence_terms": ["Python", "SIEM", "incident response"],
        "category": category,
        "language": language,
        "run_id": f"run-{packet_id}",
        "source_job_url": f"https://example.test/jobs/{packet_id}",
        "template": "modern",
        "format": "txt",
        "model": "test-model",
        "generator_version": "test-version",
    }
    if generated_at is not None:
        record["generated_at"] = generated_at
    if min_dimension_score is not None:
        record["min_dimension_score"] = min_dimension_score
    if min_overall_score is not None:
        record["min_overall_score"] = min_overall_score
    return record


def _write_packet_set(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    packet_set = tmp_path / "packet-set.jsonl"
    packet_set.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return packet_set


def _french_quality_packet_record(packet_id: str = "risk-fr") -> dict[str, object]:
    return {
        **_quality_packet_record(packet_id=packet_id, category="risk", language="fr"),
        "keywords": ["SIEM", "SOC", "incident", "triage", "escalade", "Linux"],
        "coherence_terms": ["SIEM", "SOC", "incident", "triage"],
    }


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
    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(tmp_path, [_quality_packet_record(packet_id="acme-security")])

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            min_cases=1,
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "Document packet certification: CERTIFIED" in output
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
    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(tmp_path, [_quality_packet_record(packet_id="acme-security")])

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            json_output=True,
            min_cases=1,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["passed"] is True
    assert payload["certified"] is True
    assert payload["mode"] == "packet_set"
    assert payload["required"] is True
    assert payload["thresholds"]["min_cases"] == 1
    assert payload["count"] == 1
    assert set(payload["dimension_means"]) == {
        "usefulness",
        "specificity",
        "coherence",
        "writing_quality",
        "formatting_polish",
    }
    assert payload["packets"][0]["packet_id"] == "acme-security"
    assert payload["packets"][0]["category"] == "support"
    assert payload["packets"][0]["language"] == "en"
    assert payload["packets"][0]["provenance"]["run_id"] == "run-acme-security"


def test_document_quality_required_packet_set_needs_three_passing_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(tmp_path, [_quality_packet_record(packet_id="single")])

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            json_output=True,
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["passed"] is True
    assert payload["certified"] is False
    assert payload["thresholds"]["min_cases"] == 3
    assert "passing packet count 1 is below required 3" in payload["certification_failures"]


def test_document_quality_optional_packet_set_reports_uncertified_without_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(tmp_path, [_quality_packet_record(packet_id="single")])

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=False,
            min_cases=3,
            json_output=True,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["passed"] is True
    assert payload["certified"] is False
    assert payload["thresholds"]["min_cases"] == 3


def test_document_quality_stale_packet_fails_required_certification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_quality_artifacts(tmp_path)
    old_generated_at = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    packet_set = _write_packet_set(
        tmp_path,
        [_quality_packet_record(packet_id="old", generated_at=old_generated_at)],
    )

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            min_cases=1,
            max_artifact_age_days=14,
            json_output=True,
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["certified"] is False
    assert payload["freshness"]["stale_packet_ids"] == ["old"]


def test_document_quality_future_generated_at_fails_required_certification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_quality_artifacts(tmp_path)
    future_generated_at = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    packet_set = _write_packet_set(
        tmp_path,
        [_quality_packet_record(packet_id="future", generated_at=future_generated_at)],
    )

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            min_cases=1,
            max_artifact_age_days=14,
            json_output=True,
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["certified"] is False
    assert "packet generated_at is in the future: future" in payload["certification_failures"]


def test_document_quality_artifact_mtime_freshness_uses_oldest_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resume, cover = _write_quality_artifacts(tmp_path)
    old_mtime = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    fresh_mtime = datetime.now(UTC).timestamp()
    os.utime(resume, (old_mtime, old_mtime))
    os.utime(cover, (fresh_mtime, fresh_mtime))
    packet_set = _write_packet_set(
        tmp_path,
        [_quality_packet_record(packet_id="mixed-mtime")],
    )

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            min_cases=1,
            max_artifact_age_days=14,
            json_output=True,
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["certified"] is False
    assert payload["freshness"]["stale_packet_ids"] == ["mixed-mtime"]
    assert payload["packets"][0]["generated_at_source"] == "artifact_mtime"


def test_document_quality_generated_at_overrides_artifact_mtime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resume, cover = _write_quality_artifacts(tmp_path)
    old_mtime = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    os.utime(resume, (old_mtime, old_mtime))
    os.utime(cover, (old_mtime, old_mtime))
    fresh_generated_at = datetime.now(UTC).isoformat()
    packet_set = _write_packet_set(
        tmp_path,
        [_quality_packet_record(packet_id="fresh", generated_at=fresh_generated_at)],
    )

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            min_cases=1,
            max_artifact_age_days=14,
            json_output=True,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["certified"] is True
    assert payload["freshness"]["stale_packet_ids"] == []
    assert payload["packets"][0]["generated_at_source"] == "generated_at"


def test_document_quality_required_language_accepts_alias_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_french_quality_artifacts(
        tmp_path,
        profile=(
            "Professionnel des opérations avec plus de 10 ans d'expérience en gestion des "
            "opérations, résolution de problèmes clients, triage et escalade. Apporte une "
            "expérience de support technique et une formation en cybersécurité et réseautique."
        ),
        education_note=(
            "Cours complétés : Intro to Cybersecurity, Attack & Defense Methods, Server Security "
            "et Networking & Security, incluant des laboratoires SIEM, opérations SOC et réponse "
            "aux incidents."
        ),
    )
    record = _french_quality_packet_record(packet_id="risk-french")
    record["language"] = "French"
    packet_set = _write_packet_set(tmp_path, [record])

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            min_cases=1,
            required_languages=["fr"],
            json_output=True,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["certified"] is True
    assert payload["coverage"]["present_languages"] == ["fr"]
    assert payload["coverage"]["missing_languages"] == []


def test_document_quality_missing_required_category_and_language_fails_certification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(tmp_path, [_quality_packet_record(packet_id="support-en")])

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            min_cases=1,
            required_categories=["risk"],
            required_languages=["fr"],
            json_output=True,
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["coverage"]["missing_categories"] == ["risk"]
    assert payload["coverage"]["missing_languages"] == ["fr"]


def test_document_quality_represented_category_and_language_coverage_certifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    support_en_dir = tmp_path / "support-en"
    risk_fr_dir = tmp_path / "risk-fr"
    support_fr_dir = tmp_path / "support-fr"
    support_en_dir.mkdir()
    risk_fr_dir.mkdir()
    support_fr_dir.mkdir()
    _write_quality_artifacts(support_en_dir)
    french_profile = (
        "Professionnel des opérations avec plus de 10 ans d'expérience en gestion des "
        "opérations, résolution de problèmes clients, triage et escalade. Apporte une "
        "expérience de support technique et une formation en cybersécurité et réseautique."
    )
    french_education = (
        "Cours complétés : Intro to Cybersecurity, Attack & Defense Methods, Server Security "
        "et Networking & Security, incluant des laboratoires SIEM, opérations SOC et réponse "
        "aux incidents."
    )
    _write_french_quality_artifacts(
        risk_fr_dir,
        profile=french_profile,
        education_note=french_education,
    )
    _write_french_quality_artifacts(
        support_fr_dir,
        profile=french_profile,
        education_note=french_education,
    )
    support_en = _quality_packet_record(
        packet_id="support-en",
        category="support",
        language="en",
    )
    support_en.update(
        {"resume_path": "support-en/resume.txt", "cover_letter_path": "support-en/cover.txt"}
    )
    risk_fr = _french_quality_packet_record(packet_id="risk-fr")
    risk_fr.update({"resume_path": "risk-fr/resume.txt", "cover_letter_path": "risk-fr/cover.txt"})
    support_fr = {**_french_quality_packet_record(packet_id="support-fr"), "category": "support"}
    support_fr.update(
        {"resume_path": "support-fr/resume.txt", "cover_letter_path": "support-fr/cover.txt"}
    )
    packet_set = _write_packet_set(
        tmp_path,
        [support_en, risk_fr, support_fr],
    )

    assert (
        eval_document_quality._run_packet_set(
            packet_set=packet_set,
            required=True,
            required_categories=["support", "risk"],
            required_languages=["en", "fr"],
            json_output=True,
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["certified"] is True
    assert payload["coverage"]["missing_categories"] == []
    assert payload["coverage"]["missing_languages"] == []
    assert payload["coverage"]["present_categories"] == ["risk", "support"]
    assert payload["coverage"]["present_languages"] == ["en", "fr"]


def test_document_quality_declared_french_packet_rejects_english_resume_prose(
    tmp_path: Path,
) -> None:
    _write_french_quality_artifacts(
        tmp_path,
        profile=(
            "Operations professional with 10+ years of operations management, high-stakes client "
            "problem-solving, triage, and escalation experience. Brings front-line technical "
            "support experience plus cybersecurity and networking coursework."
        ),
        education_note=(
            "Completed: Intro to Cybersecurity, Attack & Defense Methods, Server Security, and "
            "Networking & Security, including SIEM labs, SOC operations, and incident response."
        ),
    )
    packet_set = _write_packet_set(
        tmp_path,
        [_french_quality_packet_record()],
    )

    report = eval_document_quality.assess_packet_set(packet_set)[0]

    assert report.passed is False
    assert any("declared fr packet" in item for item in report.failures)
    assert report.language_quality is not None
    assert "resume:profile" in report.language_quality.mismatched_sections


def test_document_quality_declared_french_packet_allows_english_technical_terms(
    tmp_path: Path,
) -> None:
    _write_french_quality_artifacts(
        tmp_path,
        profile=(
            "Professionnel des opérations avec plus de 10 ans d'expérience en gestion des "
            "opérations, résolution de problèmes clients, triage et escalade. Apporte une "
            "expérience de support technique et une formation en cybersécurité et réseautique."
        ),
        education_note=(
            "Cours complétés : Intro to Cybersecurity, Attack & Defense Methods, Server Security "
            "et Networking & Security, incluant des laboratoires SIEM, opérations SOC et réponse "
            "aux incidents."
        ),
    )
    packet_set = _write_packet_set(
        tmp_path,
        [_french_quality_packet_record()],
    )

    report = eval_document_quality.assess_packet_set(packet_set)[0]

    assert report.passed is True
    assert report.language_quality is not None
    assert report.language_quality.mismatched_sections == []


def test_document_quality_declared_french_packet_rejects_english_connector_leakage(
    tmp_path: Path,
) -> None:
    _write_french_quality_artifacts(
        tmp_path,
        profile=(
            "Professionnel des opérations avec plus de 10 ans d'expérience en gestion des "
            "opérations, résolution de problèmes clients, triage et escalade. Apporte une "
            "expérience de support technique et une formation en cybersécurité et réseautique."
        ),
        education_note=(
            "Cours complétés : Intro to Cybersecurity, Attack & Defense Methods, Server Security "
            "et Networking & Security, incluant des laboratoires SIEM, opérations SOC, réponse "
            "aux incidents, and threat intelligence."
        ),
    )
    packet_set = _write_packet_set(
        tmp_path,
        [_french_quality_packet_record()],
    )

    report = eval_document_quality.assess_packet_set(packet_set)[0]

    assert report.passed is False
    assert report.language_quality is not None
    assert "resume:education" in report.language_quality.mismatched_sections


def test_document_quality_packet_rejects_duplicate_resume_bullets(tmp_path: Path) -> None:
    _write_french_quality_artifacts(
        tmp_path,
        profile=(
            "Professionnel des opérations avec plus de 10 ans d'expérience en gestion des "
            "opérations, résolution de problèmes clients, triage et escalade. Apporte une "
            "expérience de support technique et une formation en cybersécurité et réseautique."
        ),
        education_note=(
            "Cours complétés : Intro to Cybersecurity, Attack & Defense Methods, Server Security "
            "et Networking & Security, incluant des laboratoires SIEM, opérations SOC et réponse "
            "aux incidents."
        ),
    )
    resume = tmp_path / "resume.txt"
    resume.write_text(
        resume.read_text(encoding="utf-8")
        + "\n• Trié et escaladé les problèmes complexes vers les niveaux supérieurs.\n"
        "• Trié et escaladé les problèmes complexes vers les niveaux supérieurs.\n",
        encoding="utf-8",
    )
    packet_set = _write_packet_set(tmp_path, [_french_quality_packet_record()])

    report = eval_document_quality.assess_packet_set(packet_set)[0]

    assert report.passed is False
    assert any("duplicate resume bullet" in item for item in report.failures)


def test_document_quality_case_floors_cannot_lower_global_floors(tmp_path: Path) -> None:
    resume = tmp_path / "resume.txt"
    cover = tmp_path / "cover.txt"
    packet_set = tmp_path / "packet-set.jsonl"
    resume.write_text("Jane\nNo sections yet", encoding="utf-8")
    cover.write_text("- TODO: write letter\n\nRegards,\nYour Name", encoding="utf-8")
    packet_set.write_text(
        json.dumps(
            {
                "id": "weak",
                "resume_path": "resume.txt",
                "cover_letter_path": "cover.txt",
                "applicant_name": "Jane Doe",
                "keywords": ["Python", "SIEM"],
                "min_dimension_score": 0.0,
                "min_overall_score": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = eval_document_quality.assess_packet_set(
        packet_set,
        min_dimension_score=3.5,
        min_overall_score=3.5,
    )[0]

    assert any("below required 3.50" in item for item in report.failures)


def test_document_quality_unbacked_coherence_terms_reduce_coherence(tmp_path: Path) -> None:
    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(
        tmp_path,
        [
            {
                **_quality_packet_record(packet_id="unbacked"),
                "coherence_terms": ["Kubernetes"],
            }
        ],
    )

    report = eval_document_quality.assess_packet_set(packet_set)[0]

    assert report.dimensions["coherence"] < 3.0
    assert any("coherence terms are not source-backed" in item for item in report.warnings)


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


@pytest.mark.parametrize(
    "flag",
    [
        "--min-cases",
        "--max-artifact-age-days",
        "--required-category",
        "--required-language",
    ],
)
def test_document_quality_cli_rejects_packet_only_flags_for_single_artifact(
    tmp_path: Path,
    flag: str,
) -> None:
    from job_applicator import cli

    resume = tmp_path / "resume.txt"
    resume.write_text(_quality_resume(), encoding="utf-8")

    result = CliRunner().invoke(
        cli.app,
        ["document-quality", "--resume", str(resume), flag, "1"],
    )

    assert result.exit_code == 2


def test_document_quality_cli_missing_optional_packet_set_skips(tmp_path: Path) -> None:
    from job_applicator import cli

    missing = tmp_path / "missing.jsonl"

    result = CliRunner().invoke(cli.app, ["document-quality", "--packet-set", str(missing)])

    assert result.exit_code == 0
    assert "not certified" in result.output


def test_document_quality_cli_missing_optional_packet_set_json_is_valid(tmp_path: Path) -> None:
    from job_applicator import cli

    missing = tmp_path / "missing.jsonl"

    result = CliRunner().invoke(
        cli.app,
        ["document-quality", "--packet-set", str(missing), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["passed"] is False
    assert payload["certified"] is False
    assert payload["reason"] == "missing_packet_set"


def test_document_quality_cli_missing_required_packet_set_fails(tmp_path: Path) -> None:
    from job_applicator import cli

    missing = tmp_path / "missing.jsonl"

    result = CliRunner().invoke(
        cli.app, ["document-quality", "--packet-set", str(missing), "--required"]
    )

    assert result.exit_code == 2
    assert "not certified" in result.output


def test_document_quality_cli_packet_certification_flags_are_reflected_in_json(
    tmp_path: Path,
) -> None:
    from job_applicator import cli

    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(tmp_path, [_quality_packet_record(packet_id="support-en")])

    result = CliRunner().invoke(
        cli.app,
        [
            "document-quality",
            "--packet-set",
            str(packet_set),
            "--required",
            "--min-cases",
            "1",
            "--max-artifact-age-days",
            "14",
            "--required-category",
            "support",
            "--required-language",
            "en",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["thresholds"] == {
        "min_dimension_score": 3.0,
        "min_overall_score": 3.0,
        "min_cases": 1,
        "max_artifact_age_days": 14,
    }
    assert payload["coverage"]["present_categories"] == ["support"]
    assert payload["coverage"]["present_languages"] == ["en"]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--min-dimension-score", "4.1"),
        ("--min-cases", "0"),
        ("--max-artifact-age-days", "-1"),
    ],
)
def test_document_quality_cli_rejects_invalid_certification_values(
    tmp_path: Path,
    flag: str,
    value: str,
) -> None:
    from job_applicator import cli

    missing = tmp_path / "missing.jsonl"

    result = CliRunner().invoke(
        cli.app,
        ["document-quality", "--packet-set", str(missing), flag, value],
    )

    assert result.exit_code != 0


def test_document_quality_cli_private_packet_set_json_output(tmp_path: Path) -> None:
    from job_applicator import cli

    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(tmp_path, [_quality_packet_record(packet_id="acme-security")])

    result = CliRunner().invoke(
        cli.app,
        [
            "document-quality",
            "--packet-set",
            str(packet_set),
            "--required",
            "--min-cases",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert payload["certified"] is True
    assert payload["thresholds"]["min_cases"] == 1
    assert payload["packets"][0]["dimensions"]["coherence"] >= 3.0


def test_document_quality_cli_private_packet_set_uses_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from job_applicator import cli

    _write_quality_artifacts(tmp_path)
    packet_set = _write_packet_set(tmp_path, [_quality_packet_record(packet_id="acme-security")])
    monkeypatch.setenv("DOCUMENT_QUALITY_SET", str(packet_set))

    result = CliRunner().invoke(
        cli.app,
        ["document-quality", "--private-packet-set", "--required", "--min-cases", "1", "--json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["packet_set"] == str(packet_set)
