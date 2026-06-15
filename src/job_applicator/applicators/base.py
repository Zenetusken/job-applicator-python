"""Abstract applicator interface — all applicators implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from job_applicator.models import ApplicationResult, JobListing


class BaseApplicator(ABC):
    """Abstract base class for job application submitters."""

    @abstractmethod
    async def apply(
        self, job: JobListing, cover_letter: str | None = None, submit: bool = False
    ) -> ApplicationResult:
        """Prepare/submit a job application. Returns the result.

        When ``submit`` is False (the default), the application is prepared but
        NOT actually submitted — a safety default so automated runs never send
        real applications without explicit opt-in.
        """

    @abstractmethod
    async def check_already_applied(self, job: JobListing) -> bool:
        """Check if we've already applied to this job."""
