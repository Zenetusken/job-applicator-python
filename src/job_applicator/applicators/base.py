"""Abstract applicator interface — all applicators implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from job_applicator.models import ApplicationResult, ApplicationStatus, JobListing


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

    async def _gated_submit(
        self,
        *,
        submit: bool,
        job: JobListing,
        cover_letter: str | None,
        do_submit: Callable[[], Awaitable[ApplicationResult]],
        dry_run_note: str,
    ) -> ApplicationResult:
        """Enforce the dry-run-by-default gate in ONE place.

        Invokes ``do_submit`` (the board-specific final submission) only when
        ``submit`` is True; otherwise returns a SKIPPED result describing the
        dry run. Concrete applicators MUST route their final submission through
        this method so a new board cannot send a real application by forgetting
        the ``if not submit`` check.
        """
        if not submit:
            return ApplicationResult(
                job=job,
                status=ApplicationStatus.SKIPPED,
                cover_letter=cover_letter,
                notes=dry_run_note,
            )
        return await do_submit()
