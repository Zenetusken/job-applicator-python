#!/usr/bin/env python
"""Measure LLM sampler variants against generated document packet quality.

This is a private-data harness. It reads local sampler cases, runs the public
``job-applicator batch`` command once per case and sampler variant, writes generated
packet manifests under a local output root, then certifies those manifests through
the same document-quality packet-set evaluator used by the CLI gate.

Default private input:
``~/.job-applicator/document-quality-eval/sampler-cases.jsonl``

Default private output:
``~/.job-applicator/document-quality-eval/sampler-runs/``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from job_applicator import __version__
from job_applicator.config import AppSettings
from job_applicator.documents.quality_eval import assess_packet_set, certify_packet_set
from job_applicator.documents.resume import ResumeLoader
from job_applicator.documents.resume_document import ResumeDocument, protected_span_recall
from job_applicator.embeddings.target_criteria import TARGET_CRITERIA_CACHE_ENV

DEFAULT_CASES_FILE = Path("~/.job-applicator/document-quality-eval/sampler-cases.jsonl")
DEFAULT_OUTPUT_ROOT = Path("~/.job-applicator/document-quality-eval/sampler-runs")
GENERATOR_VERSION = f"job-applicator-{__version__}+source-overlay-v6"
SAMPLER_ENV_KEYS = (
    "JOB_APPLICATOR_LLM_TEMPERATURE",
    "JOB_APPLICATOR_LLM_TOP_P",
    "JOB_APPLICATOR_LLM_TOP_K",
    "JOB_APPLICATOR_LLM_MIN_P",
    "JOB_APPLICATOR_LLM_PRESENCE_PENALTY",
    "JOB_APPLICATOR_LLM_ENABLE_THINKING",
)
DIMENSIONS = ("usefulness", "specificity", "coherence", "writing_quality", "formatting_polish")
ROTATING_TEMPLATES = ("modern", "classic", "minimal")
MIN_PDF_TEXT_RETENTION = 0.99
MAX_RESUME_PAGES = 3


@dataclass(frozen=True)
class SamplerVariant:
    name: str
    sampler: dict[str, Any]
    env_overrides: dict[str, str]


@dataclass(frozen=True)
class SamplerCase:
    case_id: str
    jobs_file: Path
    resume_path: Path | None
    style_guide_path: Path | None
    category: str | None
    language: str | None
    applicant_name: str
    keywords: list[str]
    coherence_terms: list[str]
    protected_spans: list[str]
    top_k: int
    min_score: float
    output_format: str
    template: str | None


VARIANTS: dict[str, SamplerVariant] = {
    "baseline": SamplerVariant(
        name="baseline",
        sampler={"source": "ambient config with sampler env overrides removed"},
        env_overrides={},
    ),
    "qwen-pp10": SamplerVariant(
        name="qwen-pp10",
        sampler={
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.0,
            "enable_thinking": False,
        },
        env_overrides={
            "JOB_APPLICATOR_LLM_TOP_P": "0.8",
            "JOB_APPLICATOR_LLM_TOP_K": "20",
            "JOB_APPLICATOR_LLM_MIN_P": "0.0",
            "JOB_APPLICATOR_LLM_PRESENCE_PENALTY": "1.0",
            "JOB_APPLICATOR_LLM_ENABLE_THINKING": "false",
        },
    ),
    "qwen-grounded": SamplerVariant(
        name="qwen-grounded",
        sampler={
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "enable_thinking": False,
        },
        env_overrides={
            "JOB_APPLICATOR_LLM_TOP_P": "0.8",
            "JOB_APPLICATOR_LLM_TOP_K": "20",
            "JOB_APPLICATOR_LLM_MIN_P": "0.0",
            "JOB_APPLICATOR_LLM_PRESENCE_PENALTY": "0.0",
            "JOB_APPLICATOR_LLM_ENABLE_THINKING": "false",
        },
    ),
    "qwen-pp12": SamplerVariant(
        name="qwen-pp12",
        sampler={
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.2,
            "enable_thinking": False,
        },
        env_overrides={
            "JOB_APPLICATOR_LLM_TOP_P": "0.8",
            "JOB_APPLICATOR_LLM_TOP_K": "20",
            "JOB_APPLICATOR_LLM_MIN_P": "0.0",
            "JOB_APPLICATOR_LLM_PRESENCE_PENALTY": "1.2",
            "JOB_APPLICATOR_LLM_ENABLE_THINKING": "false",
        },
    ),
    "qwen-pp15": SamplerVariant(
        name="qwen-pp15",
        sampler={
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "enable_thinking": False,
        },
        env_overrides={
            "JOB_APPLICATOR_LLM_TOP_P": "0.8",
            "JOB_APPLICATOR_LLM_TOP_K": "20",
            "JOB_APPLICATOR_LLM_MIN_P": "0.0",
            "JOB_APPLICATOR_LLM_PRESENCE_PENALTY": "1.5",
            "JOB_APPLICATOR_LLM_ENABLE_THINKING": "false",
        },
    ),
}


def _expand(path: Path) -> Path:
    return path.expanduser()


def _now_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_provenance() -> dict[str, Any]:
    """Record the exact tracked and uncommitted implementation under measurement."""

    try:
        head = subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["/usr/bin/git", "diff", "--binary"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None, "diff_sha256": None}
    return {
        "head": head,
        "dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return slug.strip("-") or "case"


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_text_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"{field} must be a string or list of strings")


def _resolve_path(value: Any, *, base_dir: Path, field: str, required: bool) -> Path | None:
    text = _as_text(value)
    if text is None:
        if required:
            raise ValueError(f"missing required {field}")
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _record_value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return None


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    records: list[Any]
    if path.suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
        records = loaded.get("cases", []) if isinstance(loaded, dict) else loaded

    if not isinstance(records, list):
        raise ValueError("sampler cases must be JSONL objects, a JSON list, or {'cases': [...]}")

    typed: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"sampler case {index} must be an object")
        typed.append(record)
    return typed


def _to_int(value: Any, *, field: str, default: int, minimum: int, maximum: int) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def _to_float(value: Any, *, field: str, default: float, minimum: float, maximum: float) -> float:
    if value in (None, ""):
        return default
    parsed = float(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def load_cases(
    cases_file: Path,
    *,
    resume_override: Path | None = None,
    style_guide_override: Path | None = None,
    case_ids: set[str] | None = None,
) -> list[SamplerCase]:
    base_dir = cases_file.parent
    cases: list[SamplerCase] = []
    for index, record in enumerate(_read_records(cases_file), start=1):
        raw_id = _record_value(record, "id", "case_id", "name") or f"case-{index}"
        case_id = _slug(str(raw_id))
        if case_ids is not None and case_id not in case_ids:
            continue
        jobs_file = _resolve_path(
            _record_value(record, "jobs_file", "job_file"),
            base_dir=base_dir,
            field="jobs_file",
            required=True,
        )
        resume_path = resume_override or _resolve_path(
            _record_value(record, "resume_path", "input_resume_path", "base_resume_path"),
            base_dir=base_dir,
            field="resume_path",
            required=False,
        )
        style_guide_path = style_guide_override or _resolve_path(
            _record_value(record, "style_guide_path", "style_guide"),
            base_dir=base_dir,
            field="style_guide_path",
            required=False,
        )
        cases.append(
            SamplerCase(
                case_id=case_id,
                jobs_file=jobs_file or Path(),
                resume_path=resume_path,
                style_guide_path=style_guide_path,
                category=_as_text(_record_value(record, "category", "job_category")),
                language=_as_text(_record_value(record, "language", "expected_language")),
                applicant_name=_as_text(_record_value(record, "applicant_name", "profile_name"))
                or "",
                keywords=_as_text_list(
                    _record_value(record, "keywords", "required_keywords"),
                    field="keywords",
                ),
                coherence_terms=_as_text_list(
                    _record_value(record, "coherence_terms", "shared_terms"),
                    field="coherence_terms",
                ),
                protected_spans=_as_text_list(
                    _record_value(record, "protected_spans"),
                    field="protected_spans",
                ),
                top_k=_to_int(
                    _record_value(record, "top_k"),
                    field="top_k",
                    default=1,
                    minimum=1,
                    maximum=25,
                ),
                min_score=_to_float(
                    _record_value(record, "min_score"),
                    field="min_score",
                    default=0.0,
                    minimum=0.0,
                    maximum=1.0,
                ),
                output_format=_as_text(_record_value(record, "format", "output_format")) or "txt",
                template=_as_text(_record_value(record, "template")),
            )
        )
    return cases


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = loaded.get("jobs", []) if isinstance(loaded, dict) else loaded
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _job_text(job: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = job.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _job_for_result(case: SamplerCase, result: dict[str, Any]) -> dict[str, Any]:
    jobs = _load_jobs(case.jobs_file)
    if not jobs:
        return {}
    result_url = _as_text(result.get("url"))
    if result_url:
        for job in jobs:
            if _as_text(job.get("url")) == result_url:
                return job
    return jobs[0]


def _summary_path(output_dir: Path) -> Path | None:
    summaries = sorted(
        output_dir.glob("batch_summary_*.json"),
        key=lambda item: item.stat().st_mtime,
    )
    return summaries[-1] if summaries else None


def _path_from_result(value: Any, *, base_dir: Path) -> Path | None:
    text = _as_text(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _successful_result(summary: dict[str, Any], *, summary_dir: Path) -> dict[str, Any] | None:
    results = summary.get("results", [])
    if not isinstance(results, list):
        return None
    for result in results:
        if not isinstance(result, dict):
            continue
        resume_path = _path_from_result(result.get("resume_path"), base_dir=summary_dir)
        cover_path = _path_from_result(result.get("cover_letter_path"), base_dir=summary_dir)
        if resume_path and cover_path and resume_path.is_file() and cover_path.is_file():
            return result
    return None


def _resume_retention(
    *,
    source_path: Path | None,
    generated_path: Path,
    protected_spans: list[str],
) -> dict[str, Any]:
    """Measure source-body and protected-span retention without editing an artifact."""

    if source_path is None:
        return {
            "source_checked": False,
            "body_digest_matches": False,
            "protected_span_count": len(protected_spans),
            "protected_spans_retained": 0,
            "missing_protected_spans": protected_spans,
            "error": "source resume unavailable",
        }
    try:
        source = ResumeLoader().load(source_path)
        source_document = ResumeDocument.parse(source.raw_text)
        generated_text = generated_path.read_text(encoding="utf-8")
        generated_document = ResumeDocument.parse(generated_text)
        retained, missing = protected_span_recall(generated_text, protected_spans)
    except Exception as exc:
        return {
            "source_checked": False,
            "body_digest_matches": False,
            "protected_span_count": len(protected_spans),
            "protected_spans_retained": 0,
            "missing_protected_spans": protected_spans,
            "error": str(exc),
        }
    source_digest = source_document.non_summary_sha256()
    generated_digest = generated_document.non_summary_sha256()
    return {
        "source_checked": True,
        "source_body_sha256": source_digest,
        "generated_body_sha256": generated_digest,
        "body_digest_matches": source_digest == generated_digest,
        "protected_span_count": len(protected_spans),
        "protected_spans_retained": retained,
        "missing_protected_spans": missing,
    }


def _batch_command(
    case: SamplerCase,
    *,
    run_id: str,
    jobs_file: Path | None = None,
    output_format: str | None = None,
    template: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "job_applicator",
        "batch",
        "--jobs-file",
        str(jobs_file or case.jobs_file),
        "--top-k",
        str(case.top_k),
        "--min-score",
        str(case.min_score),
        "--format",
        output_format or case.output_format,
        "--json",
        "--run-id",
        run_id,
    ]
    if case.resume_path is not None:
        command.extend(["--resume", str(case.resume_path)])
    if case.style_guide_path is not None:
        command.extend(["--style-guide", str(case.style_guide_path)])
    effective_template = template or case.template
    if effective_template is not None:
        command.extend(["--template", effective_template])
    if case.category is not None:
        command.extend(["--category", case.category])
    return command


def _variant_env(variant: SamplerVariant, *, output_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in SAMPLER_ENV_KEYS:
        env.pop(key, None)
    env.update(variant.env_overrides)
    env["JOB_APPLICATOR_OUTPUT_DIR"] = str(output_dir)
    env[TARGET_CRITERIA_CACHE_ENV] = str(output_dir / "target-criteria-cache")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("COLUMNS", "200")
    return env


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _packet_record(
    *,
    case: SamplerCase,
    result: dict[str, Any],
    job: dict[str, Any],
    summary_dir: Path,
    run_id: str,
    model: str,
    packet_id: str,
    output_format: str,
    template: str | None,
) -> dict[str, Any] | None:
    resume_path = _path_from_result(result.get("resume_path"), base_dir=summary_dir)
    cover_path = _path_from_result(result.get("cover_letter_path"), base_dir=summary_dir)
    if resume_path is None or cover_path is None:
        return None

    job_description = _job_text(job, "description", "job_description", "summary")
    packet: dict[str, Any] = {
        "id": packet_id,
        "resume_path": str(resume_path),
        "cover_letter_path": str(cover_path),
        "source_resume_path": str(case.resume_path) if case.resume_path else None,
        "applicant_name": case.applicant_name,
        "job_title": _as_text(result.get("title")) or _job_text(job, "title"),
        "company": _as_text(result.get("company")) or _job_text(job, "company", "employer"),
        "category": case.category,
        "language": case.language,
        "generated_at": _utc_timestamp(),
        "run_id": run_id,
        "source_job_url": _as_text(result.get("url")) or _as_text(job.get("url")),
        "template": template,
        "format": output_format,
        "model": model,
        "generator_version": GENERATOR_VERSION,
    }
    if case.keywords:
        packet["keywords"] = case.keywords
    if job_description:
        packet["job_description"] = job_description
    job_requirements = _as_text_list(job.get("requirements"), field="requirements")
    if job_requirements:
        packet["job_requirements"] = job_requirements
    if case.coherence_terms:
        packet["coherence_terms"] = case.coherence_terms
    if case.protected_spans:
        packet["protected_spans"] = case.protected_spans
    retention = _resume_retention(
        source_path=case.resume_path,
        generated_path=resume_path,
        protected_spans=case.protected_spans,
    )
    packet["source_retention"] = retention
    resume_meta_path = resume_path.with_suffix(".meta.json")
    if resume_meta_path.is_file():
        packet["resume_meta_path"] = str(resume_meta_path)
        try:
            resume_meta = json.loads(resume_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            resume_meta = None
        if isinstance(resume_meta, dict) and isinstance(resume_meta.get("overlay"), dict):
            packet["resume_overlay"] = resume_meta["overlay"]
    cover_meta_path = cover_path.with_suffix(".meta.json")
    if cover_meta_path.is_file():
        packet["cover_letter_meta_path"] = str(cover_meta_path)
        try:
            cover_meta = json.loads(cover_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cover_meta = None
        if isinstance(cover_meta, dict) and isinstance(cover_meta.get("overlay"), dict):
            packet["cover_letter_overlay"] = cover_meta["overlay"]
    pdf_path = _path_from_result(result.get("pdf_path"), base_dir=summary_dir)
    cover_pdf_path = _path_from_result(result.get("cover_letter_pdf_path"), base_dir=summary_dir)
    if pdf_path is not None:
        packet["resume_pdf_path"] = str(pdf_path)
    if cover_pdf_path is not None:
        packet["cover_letter_pdf_path"] = str(cover_pdf_path)
    return {key: value for key, value in packet.items() if value not in (None, "", [])}


def _quality_payload(
    packet_set: Path,
    *,
    reports: list[Any],
    certification: Any,
) -> dict[str, Any]:
    overall = round(sum(report.overall for report in reports) / len(reports), 2) if reports else 0.0
    return {
        "packet_set": str(packet_set),
        "passed": all(report.passed for report in reports),
        "certified": certification.certified,
        "integrity_certified": certification.integrity_certified,
        "prose_qualified": certification.prose_qualified,
        "mode": certification.mode,
        "required": certification.required,
        "thresholds": asdict(certification.thresholds),
        "coverage": asdict(certification.coverage),
        "freshness": asdict(certification.freshness),
        "certification_failures": certification.certification_failures,
        "certification_warnings": certification.certification_warnings,
        "missing_evidence": certification.missing_evidence,
        "manual_review_summary": certification.manual_review_summary,
        "count": len(reports),
        "overall": overall,
        "dimension_means": {
            dimension: round(
                sum(report.dimensions[dimension] for report in reports) / len(reports),
                2,
            )
            if reports
            else 0.0
            for dimension in DIMENSIONS
        },
        "packets": [asdict(report) for report in reports],
    }


def _retention_payload(packets: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = [packet.get("source_retention") or {} for packet in packets]
    checked = [row for row in evidence if row.get("source_checked")]
    body_matches = sum(bool(row.get("body_digest_matches")) for row in checked)
    protected_total = sum(int(row.get("protected_span_count") or 0) for row in checked)
    protected_retained = sum(int(row.get("protected_spans_retained") or 0) for row in checked)
    return {
        "packets": len(packets),
        "source_checked": len(checked),
        "body_digest_matches": body_matches,
        "body_digest_match_rate": round(body_matches / len(checked), 4) if checked else 0.0,
        "protected_spans": protected_total,
        "protected_spans_retained": protected_retained,
        "protected_span_recall": (
            round(protected_retained / protected_total, 4) if protected_total else 1.0
        ),
        "missing_protected_spans": sorted(
            {span for row in checked for span in row.get("missing_protected_spans", [])}
        ),
    }


def _base_packet_id(packet_id: str) -> str:
    return re.sub(r"-r\d{2}$", "", packet_id)


def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ranking_signature(packet: dict[str, Any], overlay_key: str) -> tuple[Any, ...] | None:
    overlay = packet.get(overlay_key)
    if not isinstance(overlay, dict):
        return None
    ranking = overlay.get("evidence_ranking")
    if not isinstance(ranking, dict):
        return None
    ranked_facts = ranking.get("ranked_facts")
    target_criteria = ranking.get("target_criteria")
    if not isinstance(ranked_facts, list) or not isinstance(target_criteria, dict):
        return None
    fact_ids = tuple(
        str(item.get("fact_id"))
        for item in ranked_facts
        if isinstance(item, dict) and item.get("fact_id")
    )
    criteria = target_criteria.get("criteria")
    if not isinstance(criteria, list):
        return None
    criteria_signature = tuple(
        (str(item.get("name")), str(item.get("evidence")))
        for item in criteria
        if isinstance(item, dict)
    )
    return (
        fact_ids,
        str(target_criteria.get("job_source_sha256") or ""),
        criteria_signature,
        str(ranking.get("algorithm_version") or ""),
    )


def _pdf_page_count(path: str) -> int:
    completed = subprocess.run(
        ["/usr/bin/pdfinfo", path],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, re.MULTILINE)
    if match is None:
        raise ValueError(f"pdfinfo did not report a page count for {path}")
    return int(match.group(1))


def _pdf_text_retention(text_path: str, pdf_path: str) -> float:
    completed = subprocess.run(
        ["/usr/bin/pdftotext", "-layout", pdf_path, "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = re.findall(
        r"[A-Za-zÀ-ÿ0-9]+",
        Path(text_path).read_text(encoding="utf-8").casefold(),
    )
    rendered = re.findall(r"[A-Za-zÀ-ÿ0-9]+", completed.stdout.casefold())
    if not expected:
        return 0.0
    expected_counts = Counter(expected)
    rendered_counts = Counter(rendered)
    retained = sum(min(count, rendered_counts[token]) for token, count in expected_counts.items())
    return retained / len(expected)


def _heldout_measurements(
    packets: list[dict[str, Any]],
    *,
    require_template_rotation: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure ranking determinism and rendered-template coherence without judging prose."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        groups[_base_packet_id(str(packet.get("id") or "packet"))].append(packet)

    cases: list[dict[str, Any]] = []
    layout_errors: list[str] = []
    all_layouts: list[dict[str, Any]] = []
    for case_id, group in sorted(groups.items()):
        group.sort(key=lambda packet: str(packet.get("id") or ""))
        resume_signatures = [_ranking_signature(packet, "resume_overlay") for packet in group]
        cover_signatures = [_ranking_signature(packet, "cover_letter_overlay") for packet in group]
        ranking_complete = all(signature is not None for signature in resume_signatures) and all(
            signature is not None for signature in cover_signatures
        )
        selection_stable = (
            ranking_complete
            and len({signature[0] for signature in resume_signatures if signature is not None}) == 1
        )
        criteria_stable = (
            ranking_complete
            and len({signature[1:] for signature in resume_signatures if signature is not None})
            == 1
        )
        resume_cover_aligned = ranking_complete and all(
            resume == cover
            for resume, cover in zip(resume_signatures, cover_signatures, strict=True)
        )
        resume_hashes = {
            _file_sha256(str(packet["resume_path"]))
            for packet in group
            if packet.get("resume_path")
        }
        cover_hashes = {
            _file_sha256(str(packet["cover_letter_path"]))
            for packet in group
            if packet.get("cover_letter_path")
        }
        text_stable = len(resume_hashes) == 1 and len(cover_hashes) == 1
        layouts: list[dict[str, Any]] = []
        for packet in group:
            resume_pdf = packet.get("resume_pdf_path")
            cover_pdf = packet.get("cover_letter_pdf_path")
            if not isinstance(resume_pdf, str) or not isinstance(cover_pdf, str):
                continue
            try:
                layout = {
                    "packet_id": packet["id"],
                    "template": packet.get("template"),
                    "resume_pages": _pdf_page_count(resume_pdf),
                    "cover_pages": _pdf_page_count(cover_pdf),
                    "resume_pdf_text_retention": round(
                        _pdf_text_retention(str(packet["resume_path"]), resume_pdf), 4
                    ),
                    "cover_pdf_text_retention": round(
                        _pdf_text_retention(str(packet["cover_letter_path"]), cover_pdf), 4
                    ),
                }
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                layout_errors.append(f"{packet.get('id')}: {exc}")
                continue
            layouts.append(layout)
            all_layouts.append(layout)
        first = group[0]
        cases.append(
            {
                "case_id": case_id,
                "category": first.get("category"),
                "language": first.get("language"),
                "job_title": first.get("job_title"),
                "company": first.get("company"),
                "ranking_complete": ranking_complete,
                "selection_stable": selection_stable,
                "criteria_stable": criteria_stable,
                "resume_cover_aligned": resume_cover_aligned,
                "text_stable_across_templates": text_stable,
                "templates": sorted(
                    {str(packet["template"]) for packet in group if packet.get("template")}
                ),
                "selected_fact_ids": (
                    list(resume_signatures[0][0]) if resume_signatures[0] is not None else []
                ),
                "layouts": layouts,
            }
        )

    case_count = len(cases)

    def rate(field: str) -> float:
        return (
            round(sum(bool(case[field]) for case in cases) / case_count, 4) if case_count else 0.0
        )

    evidence_ranking = {
        "case_count": case_count,
        "packet_count": len(packets),
        "ranking_complete_rate": rate("ranking_complete"),
        "selection_stability_rate": rate("selection_stable"),
        "criteria_stability_rate": rate("criteria_stable"),
        "resume_cover_alignment_rate": rate("resume_cover_aligned"),
        "passed": bool(cases)
        and all(
            case[check]
            for case in cases
            for check in (
                "ranking_complete",
                "selection_stable",
                "criteria_stable",
                "resume_cover_aligned",
            )
        ),
        "cases": cases,
    }

    templates_present = sorted(
        {str(packet["template"]) for packet in packets if packet.get("template")}
    )
    required_templates = list(ROTATING_TEMPLATES) if require_template_rotation else []
    missing_templates = [
        template for template in required_templates if template not in templates_present
    ]
    pdf_expected = sum(
        1 for packet in packets if str(packet.get("format") or "").casefold() in {"pdf", "both"}
    )
    resume_retentions = [float(layout["resume_pdf_text_retention"]) for layout in all_layouts]
    cover_retentions = [float(layout["cover_pdf_text_retention"]) for layout in all_layouts]
    resume_pages = [int(layout["resume_pages"]) for layout in all_layouts]
    cover_pages = [int(layout["cover_pages"]) for layout in all_layouts]
    pdf_passed = pdf_expected == 0 or (
        len(all_layouts) == pdf_expected
        and not layout_errors
        and bool(resume_retentions)
        and min(resume_retentions) >= MIN_PDF_TEXT_RETENTION
        and min(cover_retentions) >= MIN_PDF_TEXT_RETENTION
        and max(resume_pages) <= MAX_RESUME_PAGES
        and set(cover_pages) == {1}
    )
    text_stability_rate = rate("text_stable_across_templates")
    template_coherence = {
        "case_count": case_count,
        "packet_count": len(packets),
        "text_stability_rate": text_stability_rate,
        "templates_present": templates_present,
        "required_templates": required_templates,
        "missing_templates": missing_templates,
        "pdf_packets_expected": pdf_expected,
        "pdf_packets_checked": len(all_layouts),
        "minimum_resume_pdf_text_retention": (
            round(min(resume_retentions), 4) if resume_retentions else None
        ),
        "minimum_cover_pdf_text_retention": (
            round(min(cover_retentions), 4) if cover_retentions else None
        ),
        "resume_page_counts": sorted(set(resume_pages)),
        "cover_page_counts": sorted(set(cover_pages)),
        "layout_errors": layout_errors,
        "thresholds": {
            "minimum_pdf_text_retention": MIN_PDF_TEXT_RETENTION,
            "maximum_resume_pages": MAX_RESUME_PAGES,
            "cover_pages": 1,
        },
        "passed": bool(cases)
        and text_stability_rate == 1.0
        and not missing_templates
        and pdf_passed,
    }
    return evidence_ranking, template_coherence


def _selected_variants(names: Sequence[str]) -> list[SamplerVariant]:
    return [VARIANTS[name] for name in names]


def _case_requirements(
    cases: list[SamplerCase],
    *,
    required_categories: Sequence[str],
    required_languages: Sequence[str],
) -> tuple[list[str], list[str]]:
    categories = list(required_categories)
    languages = list(required_languages)
    if not categories:
        categories = sorted({case.category for case in cases if case.category})
    if not languages:
        languages = sorted({case.language for case in cases if case.language})
    return categories, languages


def _run_case(
    *,
    case: SamplerCase,
    variant: SamplerVariant,
    run_id: str,
    output_dir: Path,
    timeout_seconds: int,
    dry_run: bool,
    model: str,
    replicate: int,
    output_format: str,
    template: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    effective_jobs_file = output_dir / "input-jobs.json"
    command = _batch_command(
        case,
        run_id=run_id,
        jobs_file=effective_jobs_file,
        output_format=output_format,
        template=template,
    )
    log_stdout = output_dir / "batch.stdout.log"
    log_stderr = output_dir / "batch.stderr.log"
    case_payload: dict[str, Any] = {
        "case_id": case.case_id,
        "replicate": replicate,
        "run_id": run_id,
        "command": command,
        "source_jobs_file": str(case.jobs_file),
        "effective_jobs_file": str(effective_jobs_file),
        "output_dir": str(output_dir),
        "stdout_log": str(log_stdout),
        "stderr_log": str(log_stderr),
        "sampler_env": variant.env_overrides,
        "target_criteria_cache_dir": str(output_dir / "target-criteria-cache"),
        "format": output_format,
        "template": template,
        "dry_run": dry_run,
    }
    if dry_run:
        case_payload["returncode"] = None
        return case_payload, None

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(case.jobs_file, effective_jobs_file)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            env=_variant_env(variant, output_dir=output_dir),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        log_stdout.write_text(exc.stdout or "", encoding="utf-8")
        log_stderr.write_text(exc.stderr or "", encoding="utf-8")
        case_payload.update(
            {
                "returncode": None,
                "elapsed_seconds": round(elapsed, 2),
                "error": f"batch timed out after {timeout_seconds}s",
            }
        )
        return case_payload, None

    elapsed = time.monotonic() - started
    log_stdout.write_text(completed.stdout, encoding="utf-8")
    log_stderr.write_text(completed.stderr, encoding="utf-8")
    case_payload.update(
        {
            "returncode": completed.returncode,
            "elapsed_seconds": round(elapsed, 2),
        }
    )
    if completed.returncode != 0:
        case_payload["error"] = "batch command failed"
        return case_payload, None

    summary = _summary_path(output_dir)
    if summary is None:
        case_payload["error"] = "batch did not write a batch_summary_*.json file"
        return case_payload, None
    case_payload["batch_summary"] = str(summary)

    try:
        summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        case_payload["error"] = f"invalid batch summary JSON: {exc}"
        return case_payload, None

    result = _successful_result(summary_payload, summary_dir=summary.parent)
    if result is None:
        case_payload["error"] = "batch summary has no successful resume+cover-letter packet"
        return case_payload, None

    job = _job_for_result(case, result)
    packet = _packet_record(
        case=case,
        result=result,
        job=job,
        summary_dir=summary.parent,
        run_id=run_id,
        model=model,
        packet_id=f"{case.case_id}-r{replicate:02d}",
        output_format=output_format,
        template=template,
    )
    if packet is None:
        case_payload["error"] = "could not construct packet manifest row"
        return case_payload, None
    case_payload["packet_id"] = packet["id"]
    return case_payload, packet


def _run_variant(
    *,
    variant: SamplerVariant,
    cases: list[SamplerCase],
    args: argparse.Namespace,
    model: str,
) -> dict[str, Any]:
    variant_dir = args.output_root / args.run_id / variant.name
    case_results: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    started = time.monotonic()
    for case in cases:
        for replicate in range(1, args.repetitions + 1):
            replicate_id = f"{case.case_id}-r{replicate:02d}"
            case_run_id = f"{args.run_id}-{variant.name}-{replicate_id}"
            case_output_dir = variant_dir / replicate_id
            template = (
                ROTATING_TEMPLATES[(replicate - 1) % len(ROTATING_TEMPLATES)]
                if args.rotate_templates
                else case.template
            )
            case_result, packet = _run_case(
                case=case,
                variant=variant,
                run_id=case_run_id,
                output_dir=case_output_dir,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
                model=model,
                replicate=replicate,
                output_format=args.output_format or case.output_format,
                template=template,
            )
            case_results.append(case_result)
            if packet is not None:
                packets.append(packet)

    elapsed = round(time.monotonic() - started, 2)
    packet_set = variant_dir / "packet-set.jsonl"
    quality: dict[str, Any] | None = None
    if packets and not args.dry_run:
        _write_jsonl(packet_set, packets)
        reports = assess_packet_set(
            packet_set,
            min_dimension_score=args.min_dimension_score,
            min_overall_score=args.min_overall_score,
        )
        certification = certify_packet_set(
            reports,
            required=True,
            min_dimension_score=args.min_dimension_score,
            min_overall_score=args.min_overall_score,
            min_cases=args.min_cases or len(cases) * args.repetitions,
            max_artifact_age_days=args.max_artifact_age_days,
            required_categories=args.effective_required_categories,
            required_languages=args.effective_required_languages,
        )
        quality = _quality_payload(packet_set, reports=reports, certification=certification)

    evidence_ranking, template_coherence = _heldout_measurements(
        packets,
        require_template_rotation=bool(args.rotate_templates),
    )

    return {
        "name": variant.name,
        "sampler": variant.sampler,
        "output_dir": str(variant_dir),
        "packet_set": str(packet_set),
        "elapsed_seconds": elapsed,
        "generated_cases": len(packets),
        "failed_cases": [
            f"{item['case_id']}-r{item['replicate']:02d}"
            for item in case_results
            if item.get("error")
        ],
        "cases": case_results,
        "retention": _retention_payload(packets),
        "evidence_ranking": evidence_ranking,
        "template_coherence": template_coherence,
        "quality": quality,
    }


def _write_summary_file(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    if args.dry_run:
        return payload
    summary_dir = args.output_root / args.run_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "sampler-summary.json"
    payload["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _rank_tuple(variant: dict[str, Any]) -> tuple[int, int, float, int, int]:
    quality = variant.get("quality") or {}
    return (
        1 if quality.get("certified") else 0,
        1 if quality.get("passed") else 0,
        float(quality.get("overall") or 0.0),
        -len(variant.get("failed_cases", [])),
        int(variant.get("generated_cases") or 0),
    )


def _certified_change(*, baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    baseline_certified = bool((baseline.get("quality") or {}).get("certified"))
    candidate_certified = bool((candidate.get("quality") or {}).get("certified"))
    if candidate_certified and not baseline_certified:
        return "improved"
    if baseline_certified and not candidate_certified:
        return "regressed"
    if candidate_certified:
        return "unchanged_certified"
    return "unchanged_not_certified"


def _list_difference(left: Sequence[str], right: Sequence[str]) -> list[str]:
    return sorted(set(left) - set(right))


def _quality_comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "variant": candidate["name"],
        "baseline": baseline["name"],
        "better_than_baseline": _rank_tuple(candidate) > _rank_tuple(baseline),
        "certified_change": _certified_change(baseline=baseline, candidate=candidate),
        "generated_cases_delta": candidate["generated_cases"] - baseline["generated_cases"],
        "failed_case_count_delta": len(candidate["failed_cases"]) - len(baseline["failed_cases"]),
        "resolved_failed_cases": _list_difference(
            baseline.get("failed_cases", []),
            candidate.get("failed_cases", []),
        ),
        "new_failed_cases": _list_difference(
            candidate.get("failed_cases", []),
            baseline.get("failed_cases", []),
        ),
    }
    baseline_quality = baseline.get("quality")
    candidate_quality = candidate.get("quality")
    if not baseline_quality or not candidate_quality:
        comparison["quality_delta_available"] = False
        comparison["reason"] = "baseline_or_variant_quality_unavailable"
        return comparison

    baseline_dimensions = baseline_quality.get("dimension_means", {})
    candidate_dimensions = candidate_quality.get("dimension_means", {})
    comparison.update(
        {
            "quality_delta_available": True,
            "overall_delta": round(candidate_quality["overall"] - baseline_quality["overall"], 2),
            "packet_count_delta": candidate_quality["count"] - baseline_quality["count"],
            "dimension_mean_deltas": {
                dimension: round(
                    candidate_dimensions.get(dimension, 0.0)
                    - baseline_dimensions.get(dimension, 0.0),
                    2,
                )
                for dimension in DIMENSIONS
            },
            "resolved_certification_failures": _list_difference(
                baseline_quality.get("certification_failures", []),
                candidate_quality.get("certification_failures", []),
            ),
            "new_certification_failures": _list_difference(
                candidate_quality.get("certification_failures", []),
                baseline_quality.get("certification_failures", []),
            ),
        }
    )
    return comparison


def _baseline_comparison(
    variants: list[dict[str, Any]],
    *,
    baseline_name: str = "baseline",
) -> dict[str, Any]:
    baseline = next((variant for variant in variants if variant["name"] == baseline_name), None)
    if baseline is None:
        return {
            "available": False,
            "baseline": baseline_name,
            "reason": "baseline_variant_not_selected",
            "deltas": [],
            "winner_by_quality": None,
        }

    comparable = [variant for variant in variants if variant is not baseline]
    winner = max(variants, key=_rank_tuple)["name"] if variants else None
    return {
        "available": bool(baseline.get("quality")),
        "baseline": baseline_name,
        "reason": None if baseline.get("quality") else "baseline_quality_unavailable",
        "winner_by_quality": winner,
        "deltas": [_quality_comparison(variant, baseline) for variant in comparable],
    }


def _unavailable_payload(
    *,
    cases_file: Path,
    reason: str,
    message: str,
    required: bool,
    run_id: str,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "cases_file": str(cases_file),
        "output_root": str(output_root),
        "passed": False,
        "certified": False,
        "reason": reason,
        "message": message,
        "required": required,
        "variants": [],
    }


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2))
        return
    if payload.get("reason"):
        print(payload["message"])
        return
    print(f"LLM sampler eval run: {payload['run_id']}")
    print(f"cases: {payload['case_count']} from {payload['cases_file']}")
    for variant in payload["variants"]:
        quality = variant.get("quality") or {}
        verdict = "CERTIFIED" if quality.get("certified") else "NOT CERTIFIED"
        print(
            f"{variant['name']}: {verdict}; generated={variant['generated_cases']} "
            f"failed={len(variant['failed_cases'])} packet_set={variant['packet_set']}"
        )
        if quality.get("overall") is not None:
            print(f"  overall={quality['overall']:.2f}/4")
        print(
            "  evidence ranking="
            f"{'PASS' if variant['evidence_ranking']['passed'] else 'FAIL'}; "
            "template coherence="
            f"{'PASS' if variant['template_coherence']['passed'] else 'FAIL'}"
        )
        for failure in quality.get("certification_failures", []):
            print(f"  certification failure: {failure}")
        for case_id in variant["failed_cases"]:
            print(f"  generation failure: {case_id}")
    comparison = payload.get("baseline_comparison", {})
    if comparison.get("available"):
        print("baseline comparison:")
        for delta in comparison["deltas"]:
            if not delta.get("quality_delta_available"):
                print(f"  {delta['variant']}: quality delta unavailable")
                continue
            dimension_text = ", ".join(
                f"{name} {value:+.2f}" for name, value in delta["dimension_mean_deltas"].items()
            )
            print(
                f"  {delta['variant']}: overall {delta['overall_delta']:+.2f}; "
                f"generated {delta['generated_cases_delta']:+d}; "
                f"failed {delta['failed_case_count_delta']:+d}; "
                f"certification {delta['certified_change']}; {dimension_text}"
            )


def _run(args: argparse.Namespace) -> int:
    args.cases_file = _expand(args.cases_file)
    args.output_root = _expand(args.output_root)
    if not args.cases_file.exists():
        payload = _unavailable_payload(
            cases_file=args.cases_file,
            reason="missing_sampler_cases",
            message=f"no sampler cases at {args.cases_file} - LLM sampler variants not measured",
            required=args.required,
            run_id=args.run_id,
            output_root=args.output_root,
        )
        _emit(payload, json_output=args.json)
        return 2 if args.required else 0
    if not args.cases_file.is_file():
        payload = _unavailable_payload(
            cases_file=args.cases_file,
            reason="invalid_sampler_cases",
            message=f"sampler cases path is not a file: {args.cases_file}",
            required=args.required,
            run_id=args.run_id,
            output_root=args.output_root,
        )
        _emit(payload, json_output=args.json)
        return 2

    try:
        cases = load_cases(
            args.cases_file,
            resume_override=args.resume,
            style_guide_override=args.style_guide,
            case_ids=set(args.case_id) if args.case_id else None,
        )
    except (OSError, ValueError) as exc:
        payload = _unavailable_payload(
            cases_file=args.cases_file,
            reason="invalid_sampler_cases",
            message=f"sampler cases are invalid: {exc}",
            required=args.required,
            run_id=args.run_id,
            output_root=args.output_root,
        )
        _emit(payload, json_output=args.json)
        return 2

    if not cases:
        payload = _unavailable_payload(
            cases_file=args.cases_file,
            reason="empty_sampler_cases",
            message=f"sampler cases have no selected cases at {args.cases_file}",
            required=args.required,
            run_id=args.run_id,
            output_root=args.output_root,
        )
        _emit(payload, json_output=args.json)
        return 2 if args.required else 0

    missing_source_cases = [case.case_id for case in cases if case.resume_path is None]
    if args.required and missing_source_cases:
        payload = _unavailable_payload(
            cases_file=args.cases_file,
            reason="missing_source_resumes",
            message=(
                "required sampler cases need source resumes: " + ", ".join(missing_source_cases)
            ),
            required=True,
            run_id=args.run_id,
            output_root=args.output_root,
        )
        _emit(payload, json_output=args.json)
        return 2

    required_categories, required_languages = _case_requirements(
        cases,
        required_categories=args.required_category,
        required_languages=args.required_language,
    )
    args.effective_required_categories = required_categories
    args.effective_required_languages = required_languages
    model = AppSettings().llm.model

    variants = [
        _run_variant(variant=variant, cases=cases, args=args, model=model)
        for variant in _selected_variants(args.variant)
    ]
    failed = False
    if not args.dry_run:
        failed = any(
            variant["failed_cases"]
            or not variant.get("quality")
            or not variant["quality"].get(
                "integrity_certified" if args.integrity_only else "certified"
            )
            or not variant["evidence_ranking"].get("passed")
            or not variant["template_coherence"].get("passed")
            for variant in variants
        )
    all_measurements_passed = bool(variants) and all(
        bool(variant["evidence_ranking"].get("passed"))
        and bool(variant["template_coherence"].get("passed"))
        for variant in variants
    )
    all_integrity_certified = (
        bool(variants)
        and all_measurements_passed
        and all(
            bool((variant.get("quality") or {}).get("integrity_certified")) for variant in variants
        )
    )
    all_certified = (
        bool(variants)
        and all_measurements_passed
        and all(bool((variant.get("quality") or {}).get("certified")) for variant in variants)
    )
    payload = {
        "run_id": args.run_id,
        "provenance": _git_provenance(),
        "cases_file": str(args.cases_file),
        "output_root": str(args.output_root),
        "required": args.required,
        "integrity_only": args.integrity_only,
        "dry_run": args.dry_run,
        "case_count": len(cases),
        "repetitions": args.repetitions,
        "attempted_packets": len(cases) * args.repetitions,
        "variants_requested": args.variant,
        "required_categories": required_categories,
        "required_languages": required_languages,
        "thresholds": {
            "min_dimension_score": args.min_dimension_score,
            "min_overall_score": args.min_overall_score,
            "min_cases": args.min_cases or len(cases) * args.repetitions,
            "max_artifact_age_days": args.max_artifact_age_days,
        },
        "passed": not failed,
        "integrity_certified": all_integrity_certified and not args.dry_run,
        "certified": all_certified and not args.dry_run,
        "measurements_passed": all_measurements_passed and not args.dry_run,
        "variants": variants,
        "baseline_comparison": _baseline_comparison(variants),
    }
    payload = _write_summary_file(args, payload)
    _emit(payload, json_output=args.json)
    return 1 if args.required and failed else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=DEFAULT_CASES_FILE,
        help="Private sampler cases JSONL/JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Local root for generated sampler-run artifacts.",
    )
    parser.add_argument("--run-id", default=_now_id(), help="Stable run id for this measurement.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        default=None,
        help="Sampler variant to run. Repeatable. Defaults to all variants.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Case id to run. Repeatable. Defaults to every case in the file.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Independent runs per selected case and variant.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("txt", "pdf", "both"),
        default=None,
        help="Override the artifact format declared by each case.",
    )
    parser.add_argument(
        "--rotate-templates",
        action="store_true",
        help="Rotate repetitions through modern, classic, and minimal templates.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Override input resume path.")
    parser.add_argument(
        "--style-guide",
        type=Path,
        default=None,
        help="Override input style-guide path.",
    )
    parser.add_argument(
        "--min-dimension-score",
        type=float,
        default=3.0,
        help="Minimum packet dimension score for certification.",
    )
    parser.add_argument(
        "--min-overall-score",
        type=float,
        default=3.0,
        help="Minimum packet overall score for certification.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=None,
        help="Minimum passing packet count. Defaults to selected case count.",
    )
    parser.add_argument(
        "--max-artifact-age-days",
        type=int,
        default=14,
        help="Maximum generated packet age for certification.",
    )
    parser.add_argument(
        "--required-category",
        action="append",
        default=[],
        help="Required packet category. Repeatable. Defaults to categories in cases.",
    )
    parser.add_argument(
        "--required-language",
        action="append",
        default=[],
        help="Required packet language. Repeatable. Defaults to languages in cases.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-case batch timeout.",
    )
    parser.add_argument(
        "--required",
        action="store_true",
        help="Exit non-zero when private inputs, generation, or certification fail.",
    )
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help=(
            "Use deterministic integrity certification as the required experiment gate; prose "
            "still remains unqualified until a manual review sidecar passes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan commands and output directories without running batch.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    args.variant = args.variant or ["baseline", "qwen-grounded"]
    if args.min_dimension_score < 0 or args.min_dimension_score > 4:
        parser.error("--min-dimension-score must be between 0 and 4")
    if args.min_overall_score < 0 or args.min_overall_score > 4:
        parser.error("--min-overall-score must be between 0 and 4")
    if args.min_cases is not None and args.min_cases < 1:
        parser.error("--min-cases must be at least 1")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.max_artifact_age_days < 1:
        parser.error("--max-artifact-age-days must be at least 1")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1")
    if args.resume is not None:
        args.resume = args.resume.expanduser()
    if args.style_guide is not None:
        args.style_guide = args.style_guide.expanduser()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
