"""Abstract scraper interface — all scrapers implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from job_applicator.models import JobBoard, JobListing


@dataclass
class SearchParams:
    """Parameters for job search."""

    query: str
    location: str = ""
    remote_only: bool = False
    max_results: int = 25
    board: JobBoard = JobBoard.LINKEDIN


class BaseScraper(ABC):
    """Abstract base class for job board scrapers."""

    @property
    @abstractmethod
    def board(self) -> JobBoard:
        """Which job board this scraper targets."""

    @abstractmethod
    async def scrape(self, params: SearchParams) -> list[JobListing]:
        """Scrape job listings matching search parameters."""

    @abstractmethod
    async def login(self, email: str, password: str) -> bool:
        """Authenticate with the job board. Returns True on success."""
