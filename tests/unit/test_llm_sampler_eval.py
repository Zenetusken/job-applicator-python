"""Tests for the private LLM sampler measurement harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import eval_llm_sampler

from job_applicator.documents.resume import ResumeLoader
from job_applicator.documents.resume_document import ResumeDocument
from job_applicator.documents.source_facts import (
    build_source_fact_catalog,
    is_substantive_source_fact,
)
from job_applicator.documents.source_realization import (
    realize_cover_statements,
    realize_resume_statement,
)


def _resume_text() -> str:
    return (
        "John Doe\n"
        "john@example.com\n"
        "514-555-0199\n\n"
        "Summary\n"
        "Security support analyst with incident triage experience. Uses Python and Linux for "
        "operational checks. Documents evidence and escalation handoffs.\n\n"
        "Experience\n"
        "Security Support Analyst | Acme | 2021 - Present\n"
        "• Used Python, Linux, SIEM, and incident response workflows to triage alerts, "
        "review authentication failures, document evidence, and coordinate escalations "
        "across support and infrastructure teams.\n"
        "• Improved recurring monitoring checks "
        "through scripts, runbooks, and clear handoff notes for analysts across shifts.\n\n"
        "Technical Support Analyst | Beta | 2019 - 2021\n"
        "• Resolved workstation, account, network, and access issues for users while documenting "
        "repeatable fixes.\n"
        "• Partnered with infrastructure staff on monitoring alerts and service-impacting "
        "incidents.\n"
        "• Built Python automation helpers to reduce manual "
        "follow-up and reviewed Linux logs before escalating ambiguous security symptoms.\n\n"
        "Education\n"
        "Certificate in Cybersecurity, 2024\n\n"
        "Skills\n"
        "Python, Linux, SIEM, incident response, IAM, log analysis, alert triage, documentation, "
        "support, networking, automation, troubleshooting, monitoring, escalation."
    )


def _cover_text() -> str:
    return (
        "Dear Acme Security Team,\n\n"
        "I bring practical Python, Linux, SIEM, and incident response experience from support "
        "work where I triaged alerts, reviewed authentication failures, documented root causes, "
        "and coordinated follow-up with technical teams. That background maps directly to the "
        "Security Analyst role's need for reliable investigation habits and clear communication "
        "under pressure.\n\n"
        "In recent projects I used Python to automate checks, review Linux logs, and make "
        "recurring operational work easier to track. I also supported IAM and access issues, "
        "which gave me a careful evidence-first approach to alert triage and escalation decisions "
        "when symptoms were incomplete or noisy.\n\n"
        "I would welcome the chance to discuss how that mix of support discipline, automation, "
        "SIEM analysis, and incident response practice can help Acme move investigations forward "
        "while keeping handoffs accurate and useful.\n\n"
        "Sincerely,\n"
        "John Doe"
    )


def _write_case_file(tmp_path: Path) -> Path:
    source_resume = tmp_path / "source.txt"
    source_resume.write_text(_resume_text(), encoding="utf-8")
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(
        json.dumps(
            [
                {
                    "title": "Security Analyst",
                    "company": "Acme",
                    "url": "https://example.test/jobs/1",
                    "description": "Python Linux SIEM incident response IAM support",
                }
            ]
        ),
        encoding="utf-8",
    )
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        json.dumps(
            {
                "id": "acme-security",
                "jobs_file": str(jobs_file),
                "resume_path": str(source_resume),
                "applicant_name": "John Doe",
                "category": "support",
                "language": "en",
                "keywords": ["Python", "Linux", "SIEM", "incident response"],
                "coherence_terms": ["Python", "Linux", "SIEM", "incident response"],
                "protected_spans": ["Certificate in Cybersecurity", "Technical Support Analyst"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return cases_file


def _stub_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettings:
        def __init__(self) -> None:
            self.llm = SimpleNamespace(model="Qwen/Qwen3-8B-AWQ")

    monkeypatch.setattr(eval_llm_sampler, "AppSettings", FakeSettings)


def test_missing_optional_cases_file_emits_valid_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.jsonl"

    code = eval_llm_sampler.main(
        [
            "--cases-file",
            str(missing),
            "--output-root",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["certified"] is False
    assert payload["reason"] == "missing_sampler_cases"


def test_missing_required_cases_file_fails_with_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.jsonl"

    code = eval_llm_sampler.main(
        [
            "--cases-file",
            str(missing),
            "--output-root",
            str(tmp_path / "runs"),
            "--required",
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["required"] is True
    assert payload["reason"] == "missing_sampler_cases"


def test_dry_run_plans_baseline_and_qwen_sampler_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_settings(monkeypatch)
    cases_file = _write_case_file(tmp_path)

    code = eval_llm_sampler.main(
        [
            "--cases-file",
            str(cases_file),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "sample",
            "--variant",
            "baseline",
            "--variant",
            "qwen-pp12",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    variants = {variant["name"]: variant for variant in payload["variants"]}
    assert variants["baseline"]["cases"][0]["sampler_env"] == {}
    baseline_case = variants["baseline"]["cases"][0]
    assert baseline_case["source_jobs_file"].endswith("jobs.json")
    assert baseline_case["effective_jobs_file"].endswith(
        "runs/sample/baseline/acme-security-r01/input-jobs.json"
    )
    assert baseline_case["effective_jobs_file"] in baseline_case["command"]
    assert baseline_case["target_criteria_cache_dir"].endswith(
        "runs/sample/baseline/acme-security-r01/target-criteria-cache"
    )
    assert variants["qwen-pp12"]["cases"][0]["sampler_env"] == {
        "JOB_APPLICATOR_LLM_TOP_P": "0.8",
        "JOB_APPLICATOR_LLM_TOP_K": "20",
        "JOB_APPLICATOR_LLM_MIN_P": "0.0",
        "JOB_APPLICATOR_LLM_PRESENCE_PENALTY": "1.2",
        "JOB_APPLICATOR_LLM_ENABLE_THINKING": "false",
    }
    assert payload["required_categories"] == ["support"]
    assert payload["required_languages"] == ["en"]


def test_successful_fake_batch_writes_integrity_certified_packet_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_settings(monkeypatch)
    cases_file = _write_case_file(tmp_path)
    monkeypatch.setattr(
        eval_llm_sampler,
        "_git_provenance",
        lambda: {"head": "abc", "dirty": True, "diff_sha256": "def"},
    )

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        output_dir = Path(kwargs["env"]["JOB_APPLICATOR_OUTPUT_DIR"])
        resume = output_dir / "tailored.txt"
        cover = output_dir / "cover.txt"
        source = ResumeLoader().load(tmp_path / "source.txt")
        source_document = ResumeDocument.parse(source.raw_text)
        facts = [
            fact
            for fact in build_source_fact_catalog(source).facts
            if is_substantive_source_fact(fact)
        ]
        summary_statements = [realize_resume_statement(fact) for fact in facts[:3]]
        cover_statements = realize_cover_statements(facts[:3], language="English")
        job_description = "Python Linux SIEM incident response IAM support"
        ranking = {
            "target_criteria": {
                "job_source_sha256": hashlib.sha256(job_description.encode("utf-8")).hexdigest(),
                "criteria": [{"name": "Incident response", "evidence": "incident response"}],
                "extraction_version": "target-criteria-v4",
            },
            "ranked_facts": [
                {
                    "fact_id": fact.fact_id,
                    "score": score,
                    "strongest_similarity": score,
                    "strongest_criterion_index": 0,
                }
                for fact, score in zip(facts[:3], (0.9, 0.8, 0.7), strict=True)
            ],
            "selection_focus": None,
            "algorithm_version": "criterion-embedding-v1",
        }
        generated_resume = source_document.with_summary(
            " ".join(statement.text for statement in summary_statements),
            language="English",
        )
        resume.write_text(generated_resume.render(), encoding="utf-8")
        cover.write_text(
            "Please accept my application for the Security Analyst position at Acme.\n\n"
            + " ".join(statement.text for statement in cover_statements)
            + "\n\nI would welcome the opportunity to discuss my application, these details "
            "from my background, and the needs of the role with your team."
            "\n\nSincerely,\nJohn Doe",
            encoding="utf-8",
        )
        (output_dir / "tailored.meta.json").write_text(
            json.dumps(
                {
                    "overlay": {
                        "summary_sentences": [
                            statement.model_dump() for statement in summary_statements
                        ],
                        "source_body_sha256": source_document.non_summary_sha256(),
                        "source_language": "en",
                        "evidence_ranking": ranking,
                        "architecture_version": "source-overlay-v6",
                    }
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "cover.meta.json").write_text(
            json.dumps(
                {
                    "overlay": {
                        "body_sentences": [
                            statement.model_dump() for statement in cover_statements
                        ],
                        "source_body_sha256": source_document.non_summary_sha256(),
                        "source_language": "en",
                        "evidence_ranking": ranking,
                        "architecture_version": "source-overlay-v6",
                    }
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "batch_summary_20260708_120000.json").write_text(
            json.dumps(
                {
                    "timestamp": "20260708_120000",
                    "resume": "input.pdf",
                    "total_jobs": 1,
                    "matched": 1,
                    "results": [
                        {
                            "title": "Security Analyst",
                            "company": "Acme",
                            "url": "https://example.test/jobs/1",
                            "resume_path": str(resume),
                            "cover_letter_path": str(cover),
                            "tailored": True,
                            "cover_letter": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert cmd[:3] == [eval_llm_sampler.sys.executable, "-m", "job_applicator"]
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(eval_llm_sampler.subprocess, "run", fake_run)

    code = eval_llm_sampler.main(
        [
            "--cases-file",
            str(cases_file),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "sample",
            "--variant",
            "qwen-pp10",
            "--required",
            "--integrity-only",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    variant = payload["variants"][0]
    assert payload["certified"] is False
    assert payload["integrity_certified"] is True
    assert variant["generated_cases"] == 1
    assert variant["quality"]["certified"] is False
    assert variant["quality"]["integrity_certified"] is True
    assert variant["quality"]["prose_qualified"] is False
    packet_set = Path(variant["packet_set"])
    packet = json.loads(packet_set.read_text(encoding="utf-8").strip())
    assert packet["resume_path"].endswith("tailored.txt")
    assert packet["cover_letter_path"].endswith("cover.txt")
    assert packet["source_resume_path"].endswith("source.txt")
    assert packet["id"] == "acme-security-r01"
    assert packet["source_retention"]["body_digest_matches"] is True
    assert packet["source_retention"]["protected_spans_retained"] == 2
    assert packet["cover_letter_overlay"]["architecture_version"] == "source-overlay-v6"
    assert packet["cover_letter_meta_path"].endswith("cover.meta.json")
    assert packet["source_job_url"] == "https://example.test/jobs/1"
    assert Path(payload["summary_path"]).is_file()
    case_payload = variant["cases"][0]
    assert Path(case_payload["effective_jobs_file"]).is_file()
    assert case_payload["effective_jobs_file"] in case_payload["command"]
    assert variant["retention"]["body_digest_match_rate"] == 1.0
    assert variant["retention"]["protected_span_recall"] == 1.0
    assert variant["evidence_ranking"]["passed"] is True
    assert variant["evidence_ranking"]["resume_cover_alignment_rate"] == 1.0
    assert variant["template_coherence"]["passed"] is True
    assert payload["measurements_passed"] is True
    assert packet["job_description"] == "Python Linux SIEM incident response IAM support"


def test_repetitions_rotate_templates_and_expand_required_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_settings(monkeypatch)
    cases_file = _write_case_file(tmp_path)

    code = eval_llm_sampler.main(
        [
            "--cases-file",
            str(cases_file),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "replicated",
            "--variant",
            "qwen-grounded",
            "--repetitions",
            "4",
            "--format",
            "both",
            "--rotate-templates",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    cases = payload["variants"][0]["cases"]
    assert payload["attempted_packets"] == 4
    assert payload["thresholds"]["min_cases"] == 4
    assert [case["replicate"] for case in cases] == [1, 2, 3, 4]
    assert [case["template"] for case in cases] == [
        "modern",
        "classic",
        "minimal",
        "modern",
    ]
    assert all(case["format"] == "both" for case in cases)


def test_baseline_comparison_quantifies_quality_deltas() -> None:
    dimensions = {
        "usefulness": 3.0,
        "specificity": 3.0,
        "coherence": 3.0,
        "writing_quality": 3.0,
        "formatting_polish": 3.0,
    }
    variants = [
        {
            "name": "baseline",
            "generated_cases": 1,
            "failed_cases": ["risk-fr"],
            "quality": {
                "certified": False,
                "passed": False,
                "overall": 3.0,
                "count": 1,
                "dimension_means": dimensions,
                "certification_failures": ["missing required languages: fr"],
            },
        },
        {
            "name": "qwen-pp12",
            "generated_cases": 2,
            "failed_cases": [],
            "quality": {
                "certified": True,
                "passed": True,
                "overall": 3.35,
                "count": 2,
                "dimension_means": {**dimensions, "coherence": 3.5},
                "certification_failures": [],
            },
        },
    ]

    comparison = eval_llm_sampler._baseline_comparison(variants)

    assert comparison["available"] is True
    assert comparison["winner_by_quality"] == "qwen-pp12"
    delta = comparison["deltas"][0]
    assert delta["better_than_baseline"] is True
    assert delta["certified_change"] == "improved"
    assert delta["overall_delta"] == 0.35
    assert delta["dimension_mean_deltas"]["coherence"] == 0.5
    assert delta["generated_cases_delta"] == 1
    assert delta["failed_case_count_delta"] == -1
    assert delta["resolved_failed_cases"] == ["risk-fr"]
    assert delta["resolved_certification_failures"] == ["missing required languages: fr"]


def test_invalid_threshold_values_are_rejected() -> None:
    with pytest.raises(SystemExit):
        eval_llm_sampler.main(["--min-cases", "0"])
