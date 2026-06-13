"""Indeed applicator — Phase 2 implementation."""

from __future__ import annotations

from job_applicator.applicators.base import BaseApplicator
from job_applicator.exceptions import ApplicatorError
from job_applicator.models import ApplicationResult, JobListing


class IndeedApplicator(BaseApplicator):
    """Submits job applications on Indeed (stub — Phase 2)."""

    async def apply(self, job: JobListing, cover_letter: str | None = None) -> ApplicationResult:
        raise ApplicatorError("Indeed applicator not yet implemented (Phase 2)")

    async def check_already_applied(self, job: JobListing) -> bool:
        raise ApplicatorError("Indeed applicator not yet implemented (Phase 2)")
