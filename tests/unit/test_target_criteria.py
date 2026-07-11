"""Tests for exact, job-only target-criteria extraction."""

from __future__ import annotations

from job_applicator.config import LLMConfig
from job_applicator.embeddings.target_criteria import (
    TARGET_CRITERIA_CACHE_ENV,
    TargetCriteriaExtractor,
    job_target_source_text,
)
from job_applicator.models import JobBoard, JobListing, TargetCriterion


def _job() -> JobListing:
    return JobListing(
        title="Support Analyst",
        company="Target Corp",
        url="https://example.test/job",
        description=(
            "Triage incidents, troubleshoot Windows workstations, and document escalations."
        ),
        requirements=["ServiceNow ticketing"],
        board=JobBoard.INDEED,
    )


def test_ground_criteria_keeps_only_exact_job_evidence() -> None:
    source = job_target_source_text(_job())

    grounded = TargetCriteriaExtractor._ground_criteria(
        [
            TargetCriterion(name="Incident triage", evidence="Triage incidents"),
            TargetCriterion(name="Ticketing", evidence="ServiceNow ticketing"),
            TargetCriterion(name="Invented cloud work", evidence="AWS administration"),
        ],
        source,
    )

    assert [(item.name, item.evidence) for item in grounded] == [
        ("Incident triage", "Triage incidents"),
        ("Ticketing", "ServiceNow ticketing"),
    ]


def test_ground_criteria_deduplicates_without_rewriting() -> None:
    source = job_target_source_text(_job())
    criterion = TargetCriterion(name="Windows support", evidence="Windows workstations")

    grounded = TargetCriteriaExtractor._ground_criteria([criterion, criterion], source)

    assert grounded == [criterion]


def test_ground_criteria_rejects_punctuation_rewrites() -> None:
    source = "Technical expertise - hands-on support for enterprise WiFi."

    grounded = TargetCriteriaExtractor._ground_criteria(
        [
            TargetCriterion(
                name="WiFi support",
                evidence="Technical expertise: hands-on support for enterprise WiFi.",
            )
        ],
        source,
    )

    assert grounded == []


def test_target_criteria_cache_round_trip_is_source_bound(tmp_path) -> None:
    extractor = TargetCriteriaExtractor(LLMConfig(model="test"), cache_dir=tmp_path)
    job = _job()
    criteria = extractor.build_result(
        job,
        [TargetCriterion(name="Incident triage", evidence="Triage incidents")],
    )

    extractor._save_cache(job, criteria)

    assert extractor._load_cache(job) == criteria
    changed = job.model_copy(update={"description": "Administer Linux systems."})
    assert extractor._load_cache(changed) is None


def test_target_criteria_cache_is_bound_to_request_shape(tmp_path) -> None:
    baseline = TargetCriteriaExtractor(
        LLMConfig(model="test", presence_penalty=0.0), cache_dir=tmp_path
    )
    changed = TargetCriteriaExtractor(
        LLMConfig(model="test", presence_penalty=1.2), cache_dir=tmp_path
    )

    assert baseline._cache_path(_job()) != changed._cache_path(_job())


def test_target_criteria_cache_directory_can_be_isolated_by_environment(
    monkeypatch, tmp_path
) -> None:
    isolated = tmp_path / "isolated"
    monkeypatch.setenv(TARGET_CRITERIA_CACHE_ENV, str(isolated))

    extractor = TargetCriteriaExtractor(LLMConfig(model="test"))

    assert extractor._cache_path(_job()).parent == isolated
