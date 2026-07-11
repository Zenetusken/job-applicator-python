"""Exact, job-only criteria extraction for deterministic evidence retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import instructor
from litellm import acompletion
from pydantic import BaseModel, Field

from job_applicator.config import LLMConfig
from job_applicator.documents.source_facts import format_job_target_context
from job_applicator.exceptions import LLMError
from job_applicator.models import JobListing, TargetCriteria, TargetCriterion
from job_applicator.utils.llm import (
    LLMRuntime,
    litellm_completion_kwargs,
    litellm_model,
    llm_call_error,
    quiet_litellm,
)
from job_applicator.utils.logging import get_logger

logger = get_logger("embeddings.target_criteria")

TARGET_CRITERIA_VERSION = "target-criteria-v4"
TARGET_CRITERIA_CACHE_ENV = "JOB_APPLICATOR_TARGET_CRITERIA_CACHE_DIR"
TARGET_CRITERIA_PROMPT = (
    "Select four to six retrieval criteria that best describe the concrete work performed in "
    "THIS job. You see only the job and must not make applicant claims.\n\n"
    "PRIORITY ORDER:\n"
    "1. Day-to-day actions and responsibilities the person performs, especially support, "
    "troubleshooting, monitoring, triage, incident handling, risk/compliance work, and network "
    "or system administration.\n"
    "2. Technical tools or methods directly used to perform those actions.\n"
    "Use a qualification or certification only when fewer than four concrete work criteria "
    "exist.\n\n"
    "NEVER select company background, products, benefits, location, work arrangement, schedule, "
    "on-call availability, years of experience, education, language proficiency, compensation, "
    "or generic traits. Do not fill the list with secondary criteria.\n\n"
    "For every criterion return a concise, faithful name in the SAME LANGUAGE as its evidence "
    "and an evidence span copied EXACTLY from the job text. Do not translate, paraphrase, or "
    "change punctuation in evidence. Each evidence value must be one contiguous sentence or "
    "bullet line; never combine separate sentences or bullets."
)


class _TargetCriteriaOutput(BaseModel):
    """Structured extraction payload before deterministic span validation."""

    criteria: list[TargetCriterion] = Field(min_length=1, max_length=6)

    model_config = {"extra": "forbid"}


def job_target_source_text(job: JobListing) -> str:
    """Return only the job text permitted to ground retrieval criteria."""

    parts: list[str] = []
    if job.description.strip():
        parts.append(job.description.strip())
    parts.extend(requirement.strip() for requirement in job.requirements if requirement.strip())
    return "\n".join(parts)


def _span_grounded(span: str, source: str) -> bool:
    """Require verbatim words and punctuation, allowing only whitespace reflow."""

    normalized_span = re.sub(r"\s+", " ", span.casefold()).strip()
    normalized_source = re.sub(r"\s+", " ", source.casefold()).strip()
    return bool(normalized_span and normalized_span in normalized_source)


class TargetCriteriaExtractor:
    """Extract exact role criteria without exposing applicant facts to the model."""

    def __init__(self, config: LLMConfig, *, cache_dir: Path | None = None) -> None:
        self._config = config.model_copy(update={"temperature": 0.0})
        configured_cache = os.environ.get(TARGET_CRITERIA_CACHE_ENV, "").strip()
        self._cache_dir = cache_dir or (
            Path(configured_cache).expanduser()
            if configured_cache
            else Path.home() / ".job-applicator" / "target-criteria"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _source_digest(job: JobListing) -> str:
        return hashlib.sha256(job_target_source_text(job).encode("utf-8")).hexdigest()

    def _cache_path(self, job: JobListing) -> Path:
        request_shape = self._config.model_dump(
            mode="json",
            include={
                "api_base",
                "model",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "enable_thinking",
            },
        )
        content = json.dumps(
            {
                "version": TARGET_CRITERIA_VERSION,
                "job_source_sha256": self._source_digest(job),
                "request_shape": request_shape,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, job: JobListing) -> TargetCriteria | None:
        path = self._cache_path(job)
        if not path.is_file():
            return None
        try:
            result = TargetCriteria.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("Target-criteria cache read failed: %s", exc)
            return None
        if result.job_source_sha256 != self._source_digest(job):
            return None
        return result

    def _save_cache(self, job: JobListing, result: TargetCriteria) -> None:
        try:
            self._cache_path(job).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write target-criteria cache: %s", exc)

    @staticmethod
    def _ground_criteria(criteria: list[TargetCriterion], source: str) -> list[TargetCriterion]:
        grounded: list[TargetCriterion] = []
        seen: set[tuple[str, str]] = set()
        for criterion in criteria:
            if not _span_grounded(criterion.evidence, source):
                logger.warning(
                    "Dropping target criterion %r: evidence span %r is absent from the job",
                    criterion.name,
                    criterion.evidence,
                )
                continue
            key = (criterion.name.casefold().strip(), criterion.evidence.casefold().strip())
            if key in seen:
                continue
            seen.add(key)
            grounded.append(criterion)
        return grounded

    def build_result(self, job: JobListing, criteria: list[TargetCriterion]) -> TargetCriteria:
        """Build the source-bound result after exact-span validation."""

        grounded = self._ground_criteria(criteria, job_target_source_text(job))
        if not grounded:
            raise LLMError(
                "No exact role criteria could be grounded in the job responsibilities or "
                "requirements. Provide a substantive job description before generating documents."
            )
        return TargetCriteria(
            job_source_sha256=self._source_digest(job),
            criteria=grounded,
            extraction_version=TARGET_CRITERIA_VERSION,
        )

    async def extract(
        self,
        job: JobListing,
        *,
        runtime: LLMRuntime,
        use_cache: bool = True,
    ) -> TargetCriteria:
        """Extract and cache exact criteria from the job, failing closed on no evidence."""

        source = job_target_source_text(job)
        if not source.strip():
            raise LLMError(
                "Document targeting requires job responsibilities or requirements; the job "
                "title alone is insufficient evidence."
            )
        if use_cache:
            cached = self._load_cache(job)
            if cached is not None:
                return cached

        context = format_job_target_context(job, max_description_chars=6_000)

        async def call(previous_error: LLMError | None) -> list[TargetCriterion]:
            quiet_litellm()
            messages: list[dict[str, str]] = [
                {"role": "system", "content": TARGET_CRITERIA_PROMPT},
                {"role": "user", "content": context},
            ]
            if previous_error is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The prior result had no exact evidence spans. Return only criteria "
                            "whose evidence is copied verbatim from the supplied job text."
                        ),
                    }
                )
            try:
                client: Any = instructor.from_litellm(acompletion)
                response = await client.create(
                    model=litellm_model(self._config),
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    messages=messages,
                    response_model=_TargetCriteriaOutput,
                    max_retries=2,
                    **litellm_completion_kwargs(
                        self._config,
                        temperature=0.0,
                        max_tokens=1_600,
                    ),
                )
            except Exception as exc:
                raise llm_call_error(exc, self._config.api_base or "") from exc
            return self._ground_criteria(list(response.criteria), source)

        def validate(criteria: list[TargetCriterion]) -> None:
            if not criteria:
                raise LLMError("Target-criteria extraction returned no exact job evidence spans")

        grounded = await runtime.run(call, validate)
        result = self.build_result(job, grounded)
        if use_cache:
            self._save_cache(job, result)
        return result
