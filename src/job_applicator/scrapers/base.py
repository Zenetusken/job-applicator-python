"""Abstract scraper interface — all scrapers implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from job_applicator.models import JobBoard, JobListing, SessionHealth

if TYPE_CHECKING:
    from job_applicator.browser.manager import BrowserManager
    from job_applicator.config import AppSettings


@dataclass
class SearchParams:
    """Parameters for job search."""

    query: str
    location: str = ""
    remote_only: bool = False
    max_results: int = 25
    board: JobBoard = JobBoard.LINKEDIN


@dataclass(frozen=True)
class BrowserPolicy:
    """A board's browser requirements, driven by its anti-bot defenses.

    Lives with the board (not the CLI) so the requirement can't drift from what
    the scraper actually needs and so any caller that builds a browser for a board
    gets it right. ``headed`` forces a visible/real browser (overriding the
    configured headless); ``ephemeral_profile`` uses a fresh throwaway profile per
    run; ``virtual_display`` runs the headed browser windowless via Xvfb.
    """

    headed: bool = False
    ephemeral_profile: bool = False
    virtual_display: bool = False


class BaseScraper(ABC):
    """Abstract base class for job board scrapers."""

    def __init__(self, browser: BrowserManager, config: AppSettings) -> None:
        self._browser = browser
        self._config = config

    @classmethod
    def browser_policy(cls) -> BrowserPolicy:
        """Browser requirements for this board (default: headless, persistent)."""
        return BrowserPolicy()

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

    @abstractmethod
    async def check_session(self) -> SessionHealth:
        """Best-effort check that a usable session exists for this board.

        For authenticated boards this should verify the session (e.g., load the
        feed). For public boards it may simply report that no login is required.
        Transient network failures should be surfaced in ``details`` rather than
        raised, so callers can decide whether to block or warn.
        """
