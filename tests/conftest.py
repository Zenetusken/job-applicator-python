"""Shared test fixtures and the location-based marker hook."""

from __future__ import annotations

from pathlib import Path

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
