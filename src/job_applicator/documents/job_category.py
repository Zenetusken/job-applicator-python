from __future__ import annotations

import re
from typing import ClassVar

from job_applicator.models import JobListing


class JobCategoryDetector:
    """Classify a job listing into a PDF-rendering category via keyword matching.

    The detector concatenates the job title and description, lowercases them, and
    returns the first category whose keyword list matches whole words in the text.
    Categories are ordered from most specific to least specific; the first match wins.
    If no keyword matches, the job is classified as ``default``.

    Category priority (most specific first):

    1. ``cybersecurity`` - security engineering/analyst roles, SOC, forensics, pentest.
    2. ``network-administration`` - network administration/engineering, Cisco, firewalls, VPN.
    3. ``systems-administration`` - systems administration, sysadmin, Linux/Windows admin.
    4. ``data-engineering`` - data engineering, ETL, data pipelines.
    5. ``tech-support`` - technical/IT support, help desk, support technician.
    6. ``software-engineering`` - software engineering/development, programmer.
    """

    CATEGORIES: ClassVar[dict[str, list[str]]] = {
        "cybersecurity": [
            "cybersecurity",
            "cyber security",
            "security engineer",
            "security analyst",
            "information security",
            "soc",
            "forensics",
            "pentest",
            "penetration test",
        ],
        "network-administration": [
            "network administrator",
            "network admin",
            "network engineer",
            "cisco",
            "firewall",
            "vpn",
        ],
        "systems-administration": [
            "systems administrator",
            "system administrator",
            "sysadmin",
            "linux admin",
            "windows admin",
        ],
        "data-engineering": [
            "data engineer",
            "etl",
            "data pipeline",
        ],
        "tech-support": [
            "technical support",
            "it support",
            "help desk",
            "support technician",
        ],
        "software-engineering": [
            "software engineer",
            "software developer",
            "developer",
            "programmer",
        ],
    }

    def detect(self, job: JobListing | None) -> str:
        """Return the first matching category by priority order for ``job``.

        The method lowercases the job title and description and searches for whole-word
        keyword matches. The first category (in priority order) with any matching keyword
        is returned. If ``job`` is ``None`` or no keyword matches, ``"default"`` is returned.
        """
        if job is None:
            return "default"
        text = f"{job.title or ''} {job.description or ''}".lower()
        for category, keywords in self.CATEGORIES.items():
            if any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords):
                return category
        return "default"


def detect_job_category(job: JobListing | None) -> str:
    """Convenience wrapper for :meth:`JobCategoryDetector.detect`."""
    return JobCategoryDetector().detect(job)
