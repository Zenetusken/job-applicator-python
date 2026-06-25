from __future__ import annotations

from job_applicator.documents.job_category import detect_job_category
from job_applicator.models import JobListing


def test_detects_cybersecurity() -> None:
    job = JobListing(
        title="Cybersecurity Analyst",
        company="Acme",
        url="https://example.com/job/1",
        description="Monitor SOC",
        board="linkedin",
    )
    assert detect_job_category(job) == "cybersecurity"


def test_default_when_no_match() -> None:
    job = JobListing(
        title="Unicorn Wrangler",
        company="Acme",
        url="https://example.com/job/2",
        description="Magic",
        board="linkedin",
    )
    assert detect_job_category(job) == "default"


def test_none_returns_default() -> None:
    assert detect_job_category(None) == "default"
