"""Indeed job scraper — Phase 2 implementation."""

from __future__ import annotations

from job_applicator.exceptions import ScraperError
from job_applicator.models import JobBoard, JobListing
from job_applicator.scrapers.base import BaseScraper, SearchParams


class IndeedScraper(BaseScraper):
    """Scrapes job listings from Indeed (stub — Phase 2)."""

    @property
    def board(self) -> JobBoard:
        return JobBoard.INDEED

    async def scrape(self, params: SearchParams) -> list[JobListing]:
        raise ScraperError("Indeed scraper not yet implemented (Phase 2)")

    async def login(self, email: str, password: str) -> bool:
        raise ScraperError("Indeed scraper not yet implemented (Phase 2)")
