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


def test_detects_network_administration() -> None:
    job = _job("Network Engineer", "Configure Cisco routers and firewall rules")
    assert detect_job_category(job) == "network-administration"


def test_detects_data_engineering() -> None:
    job = _job("Data Engineer", "Build ETL pipelines and data pipelines")
    assert detect_job_category(job) == "data-engineering"


def test_detects_tech_support() -> None:
    job = _job("Technical Support Specialist", "Provide IT support and help desk services")
    assert detect_job_category(job) == "tech-support"


def test_detects_software_engineering() -> None:
    job = _job("Software Engineer", "Develop scalable software as a developer")
    assert detect_job_category(job) == "software-engineering"


def test_default_when_no_match() -> None:
    job = _job("Unicorn Wrangler", "Magic")
    assert detect_job_category(job) == "default"


def test_none_returns_default() -> None:
    assert detect_job_category(None) == "default"


def test_security_guard_is_not_cybersecurity() -> None:
    """Generic 'security' roles should not be classified as cybersecurity."""
    job = _job("Security Guard", "Patrol premises and monitor access")
    assert detect_job_category(job) == "default"


def test_network_effect_is_not_network_administration() -> None:
    """Whole-word matching should not match 'network' inside unrelated phrases."""
    job = _job("Product Manager", "Leverage network effect to grow platform")
    assert detect_job_category(job) == "default"


def test_data_entry_is_not_data_engineering() -> None:
    """'Data entry' is not a data-engineering role."""
    job = _job("Data Entry Clerk", "Enter data into spreadsheets")
    assert detect_job_category(job) == "default"


def test_tech_savvy_is_not_tech_support() -> None:
    """'Tech savvy' should not match tech-support keywords."""
    job = _job("Sales Associate", "Must be tech savvy and customer focused")
    assert detect_job_category(job) == "default"


def test_ambiguous_job_uses_category_priority() -> None:
    """When multiple categories could match, the most specific one wins."""
    job = _job("Cybersecurity Network Administrator", "SOC monitoring and firewall rules")
    assert detect_job_category(job) == "cybersecurity"


def test_multi_word_keyword_systems_administrator() -> None:
    job = _job("Systems Administrator", "Maintain Linux and Windows servers")
    assert detect_job_category(job) == "systems-administration"
