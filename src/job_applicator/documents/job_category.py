from __future__ import annotations

from typing import ClassVar

from job_applicator.models import JobListing


class JobCategoryDetector:
    CATEGORIES: ClassVar[dict[str, list[str]]] = {
        "cybersecurity": ["security", "cyber", "soc", "forensics", "pentest"],
        "network-administration": ["network", "cisco", "firewall", "vpn"],
        "systems-administration": [
            "sysadmin",
            "systems administrator",
            "linux admin",
            "windows admin",
        ],
        "tech-support": ["support", "help desk", "it support", "technical support"],
        "software-engineering": ["software engineer", "developer", "programmer"],
        "data-engineering": ["data engineer", "etl", "data pipeline"],
    }

    def detect(self, job: JobListing | None) -> str:
        if job is None:
            return "default"
        text = f"{job.title or ''} {job.description or ''}".lower()
        for category, keywords in self.CATEGORIES.items():
            if any(keyword in text for keyword in keywords):
                return category
        return "default"


def detect_job_category(job: JobListing | None) -> str:
    return JobCategoryDetector().detect(job)
