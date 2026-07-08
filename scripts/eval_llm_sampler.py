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
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from job_applicator import __version__
from job_applicator.config import AppSettings
from job_applicator.documents.quality_eval import assess_packet_set, certify_packet_set

DEFAULT_CASES_FILE = Path("~/.job-applicator/document-quality-eval/sampler-cases.jsonl")
DEFAULT_OUTPUT_ROOT = Path("~/.job-applicator/document-quality-eval/sampler-runs")
GENERATOR_VERSION = f"job-applicator-{__version__}"
SAMPLER_ENV_KEYS = (
    "JOB_APPLICATOR_LLM_TOP_P",
    "JOB_APPLICATOR_LLM_TOP_K",
    "JOB_APPLICATOR_LLM_MIN_P",
    "JOB_APPLICATOR_LLM_PRESENCE_PENALTY",
    "JOB_APPLICATOR_LLM_ENABLE_THINKING",
)
DIMENSIONS = ("usefulness", "specificity", "coherence", "writing_quality", "formatting_polish")


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


def _batch_command(case: SamplerCase, *, run_id: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "job_applicator",
        "batch",
        "--jobs-file",
        str(case.jobs_file),
        "--top-k",
        str(case.top_k),
        "--min-score",
        str(case.min_score),
        "--format",
        case.output_format,
        "--json",
        "--run-id",
        run_id,
    ]
    if case.resume_path is not None:
        command.extend(["--resume", str(case.resume_path)])
    if case.style_guide_path is not None:
        command.extend(["--style-guide", str(case.style_guide_path)])
    if case.template is not None:
        command.extend(["--template", case.template])
    if case.category is not None:
        command.extend(["--category", case.category])
    return command


def _variant_env(variant: SamplerVariant, *, output_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in SAMPLER_ENV_KEYS:
        env.pop(key, None)
    env.update(variant.env_overrides)
    env["JOB_APPLICATOR_OUTPUT_DIR"] = str(output_dir)
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
) -> dict[str, Any] | None:
    resume_path = _path_from_result(result.get("resume_path"), base_dir=summary_dir)
    cover_path = _path_from_result(result.get("cover_letter_path"), base_dir=summary_dir)
    if resume_path is None or cover_path is None:
        return None

    job_description = _job_text(job, "description", "job_description", "summary")
    packet: dict[str, Any] = {
        "id": case.case_id,
        "resume_path": str(resume_path),
        "cover_letter_path": str(cover_path),
        "applicant_name": case.applicant_name,
        "job_title": _as_text(result.get("title")) or _job_text(job, "title"),
        "company": _as_text(result.get("company")) or _job_text(job, "company", "employer"),
        "category": case.category,
        "language": case.language,
        "generated_at": _utc_timestamp(),
        "run_id": run_id,
        "source_job_url": _as_text(result.get("url")) or _as_text(job.get("url")),
        "template": case.template,
        "format": case.output_format,
        "model": model,
        "generator_version": GENERATOR_VERSION,
    }
    if case.keywords:
        packet["keywords"] = case.keywords
    elif job_description:
        packet["job_description"] = job_description
    if case.coherence_terms:
        packet["coherence_terms"] = case.coherence_terms
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
        "mode": certification.mode,
        "required": certification.required,
        "thresholds": asdict(certification.thresholds),
        "coverage": asdict(certification.coverage),
        "freshness": asdict(certification.freshness),
        "certification_failures": certification.certification_failures,
        "certification_warnings": certification.certification_warnings,
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
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    command = _batch_command(case, run_id=run_id)
    log_stdout = output_dir / "batch.stdout.log"
    log_stderr = output_dir / "batch.stderr.log"
    case_payload: dict[str, Any] = {
        "case_id": case.case_id,
        "run_id": run_id,
        "command": command,
        "output_dir": str(output_dir),
        "stdout_log": str(log_stdout),
        "stderr_log": str(log_stderr),
        "sampler_env": variant.env_overrides,
        "dry_run": dry_run,
    }
    if dry_run:
        case_payload["returncode"] = None
        return case_payload, None

    output_dir.mkdir(parents=True, exist_ok=True)
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
        case_run_id = f"{args.run_id}-{variant.name}-{case.case_id}"
        case_output_dir = variant_dir / case.case_id
        case_result, packet = _run_case(
            case=case,
            variant=variant,
            run_id=case_run_id,
            output_dir=case_output_dir,
            timeout_seconds=args.timeout_seconds,
            dry_run=args.dry_run,
            model=model,
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
            min_cases=args.min_cases or len(cases),
            max_artifact_age_days=args.max_artifact_age_days,
            required_categories=args.effective_required_categories,
            required_languages=args.effective_required_languages,
        )
        quality = _quality_payload(packet_set, reports=reports, certification=certification)

    return {
        "name": variant.name,
        "sampler": variant.sampler,
        "output_dir": str(variant_dir),
        "packet_set": str(packet_set),
        "elapsed_seconds": elapsed,
        "generated_cases": len(packets),
        "failed_cases": [item["case_id"] for item in case_results if item.get("error")],
        "cases": case_results,
        "quality": quality,
    }


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
            or not variant["quality"].get("certified")
            for variant in variants
        )
    payload = {
        "run_id": args.run_id,
        "cases_file": str(args.cases_file),
        "output_root": str(args.output_root),
        "required": args.required,
        "dry_run": args.dry_run,
        "case_count": len(cases),
        "variants_requested": args.variant,
        "required_categories": required_categories,
        "required_languages": required_languages,
        "thresholds": {
            "min_dimension_score": args.min_dimension_score,
            "min_overall_score": args.min_overall_score,
            "min_cases": args.min_cases or len(cases),
            "max_artifact_age_days": args.max_artifact_age_days,
        },
        "passed": not failed,
        "certified": not failed and not args.dry_run,
        "variants": variants,
        "baseline_comparison": _baseline_comparison(variants),
    }
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
        "--dry-run",
        action="store_true",
        help="Plan commands and output directories without running batch.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    args.variant = args.variant or sorted(VARIANTS)
    if args.min_dimension_score < 0 or args.min_dimension_score > 4:
        parser.error("--min-dimension-score must be between 0 and 4")
    if args.min_overall_score < 0 or args.min_overall_score > 4:
        parser.error("--min-overall-score must be between 0 and 4")
    if args.min_cases is not None and args.min_cases < 1:
        parser.error("--min-cases must be at least 1")
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
