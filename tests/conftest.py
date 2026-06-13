"""Shared test fixtures."""

from __future__ import annotations

import pytest

from job_applicator.config import AppSettings, BrowserConfig, LLMConfig, TargetConfig
from job_applicator.models import JobBoard, JobListing, ResumeData, UserProfile


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
