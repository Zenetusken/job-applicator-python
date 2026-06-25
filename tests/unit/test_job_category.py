from __future__ import annotations

from job_applicator.documents.job_category import detect_job_category
from job_applicator.models import JobListing


def _job(title: str, description: str = "") -> JobListing:
    return JobListing(
        title=title,
        company="Acme",
        url="https://example.com/job/1",
        description=description,
        board="linkedin",
    )


def test_detects_cybersecurity() -> None:
    job = _job("Cybersecurity Analyst", "Monitor SOC")
    assert detect_job_category(job) == "cybersecurity"


def test_default_when_no_match() -> None:
    job = _job("Unicorn Wrangler", "Magic")
    assert detect_job_category(job) == "default"


def test_none_returns_default() -> None:
    assert detect_job_category(None) == "default"


def test_security_guard_is_not_cybersecurity() -> None:
    """Generic 'security' roles should not be classified as cybersecurity."""
    job = _job("Security Guard", "Patrol premises and monitor access")
    assert detect_job_category(job) == "default"


def test_ambiguous_job_uses_category_priority() -> None:
    """When multiple categories could match, the most specific one wins."""
    job = _job("Cybersecurity Network Administrator", "SOC monitoring and firewall rules")
    assert detect_job_category(job) == "cybersecurity"


def test_multi_word_keyword_systems_administrator() -> None:
    job = _job("Systems Administrator", "Maintain Linux and Windows servers")
    assert detect_job_category(job) == "systems-administration"
