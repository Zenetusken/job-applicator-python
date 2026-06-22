"""Shared test fixtures and the location-based marker hook."""

from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from job_applicator.config import AppSettings, BrowserConfig, LLMConfig, TargetConfig
from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark tests by location so marker selection works without decorating
    every test:

      tests/unit/**        -> ``unit``         (fast, no infra; the green gate)
      tests/integration/** -> ``integration``  (reserved; empty today)
      tests/*.py (root)    -> ``live``         (need vLLM @ localhost:8000 + GPU)

    Markers are registered in pyproject.toml; ``--strict-markers`` requires that.
    """
    tests_root = Path(__file__).parent
    for item in items:
        try:
            rel = item.path.relative_to(tests_root)
        except ValueError:
            continue  # not under tests/ — leave unmarked
        if rel.parts[0] == "unit":
            item.add_marker(pytest.mark.unit)
        elif rel.parts[0] == "integration":
            item.add_marker(pytest.mark.integration)
        elif len(rel.parts) == 1:  # a test file directly under tests/ root
            item.add_marker(pytest.mark.live)


def _vllm_endpoint_reachable() -> bool:
    """Best-effort probe of the configured LLM endpoint (default localhost:8000).

    Used to skip live tests when the external vLLM service is not running.
    """
    settings = AppSettings()
    url = urlparse(settings.llm.api_base or "http://localhost:8000/v1")
    host = url.hostname or "localhost"
    port = url.port or 8000
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


@pytest.fixture(autouse=True)
def skip_live_if_no_vllm(request: pytest.FixtureRequest) -> None:
    """Live tests require the external vLLM endpoint; skip cleanly if absent."""
    if request.node.get_closest_marker("live") and not _vllm_endpoint_reachable():
        pytest.skip("vLLM endpoint not reachable; start it or run `pytest -m unit`")


@pytest.fixture(autouse=True)
def _isolate_local_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every local-state store at a throwaway dir so NO test can read or write the
    real ``~/.job-applicator/applications.db`` (the user's funnel / dedupe / batch state).

    Tests that pass an explicit ``db_path`` are unaffected; this guards the *no-arg*
    default — e.g. a command's ``_get_jobs_store()`` / the tailor→``mark_tailored`` hook,
    which would otherwise persist into real state during ``pytest -m unit``.
    """
    db = tmp_path / "ja-state" / "applications.db"
    for module in ("jobs_store", "state", "batch_state"):
        monkeypatch.setattr(f"job_applicator.{module}.DEFAULT_DB_PATH", db)


@pytest.fixture
def browser_config() -> BrowserConfig:
    return BrowserConfig(headless=True, slow_mo=0, timeout_ms=5000)


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(api_base="http://localhost:8000/v1", model="test-model")


@pytest.fixture
def app_settings(
    browser_config: BrowserConfig, llm_config: LLMConfig, tmp_path: object
) -> AppSettings:
    import pathlib

    output_dir = pathlib.Path(str(tmp_path)) / "test_output"
    return AppSettings(
        profile_name="Test User",
        resume_path="",
        output_dir=str(output_dir),
        browser=browser_config,
        llm=llm_config,
        target=TargetConfig(),
    )


@pytest.fixture
def sample_job() -> JobListing:
    return JobListing(
        title="Senior Python Developer",
        company="TechCorp",
        url="https://linkedin.com/jobs/12345",
        description="We are looking for a senior Python developer...",
        location="San Francisco, CA",
        board=JobBoard.LINKEDIN,
    )


@pytest.fixture
def sample_resume() -> ResumeData:
    return ResumeData(
        raw_text="John Doe\njohn@example.com\n555-0123\nSkills: Python, FastAPI, Playwright",
        name="John Doe",
        email="john@example.com",
        phone="555-0123",
        summary="Experienced Python developer",
        skills=["Python", "FastAPI", "Playwright"],
    )


@pytest.fixture
def sample_user() -> UserProfile:
    return UserProfile(
        first_name="John",
        last_name="Doe",
        email="john@example.com",
        phone="555-0123",
        resume_path="",
    )
