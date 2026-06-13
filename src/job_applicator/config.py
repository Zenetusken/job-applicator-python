"""Centralized configuration — single source of truth.

Loaded from config.toml + JOB_APPLICATOR_* env vars.
Validated immediately at startup via Pydantic Settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrowserConfig(BaseSettings):
    """Playwright browser options."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_BROWSER_")

    headless: bool = True
    slow_mo: int = 0
    viewport_width: int = 1920
    viewport_height: int = 1080
    timeout_ms: int = 30_000
    user_agent: str | None = None


class LLMConfig(BaseSettings):
    """LLM API configuration for cover letter generation."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_LLM_")

    api_base: str = "http://localhost:8000/v1"
    api_key: str = "not-needed-for-local"
    model: str = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
    max_tokens: int = 1024
    temperature: float = 0.7


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration for semantic matching."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_EMBEDDING_")

    model_name: str = "mixedbread-ai/mxbai-embed-large-v1"
    device: str = "cuda"
    batch_size: int = 32
    max_seq_length: int = 512
    memory_limit_gb: float = 1.5  # VRAM allocation
    normalize_embeddings: bool = True


class TargetConfig(BaseSettings):
    """Job board target settings."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_TARGET_")

    max_applications_per_day: int = 20
    delay_between_applications_s: float = 2.0
    linkedin_email: str = ""
    linkedin_password: str = ""
    indeed_email: str = ""
    indeed_password: str = ""


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="JOB_APPLICATOR_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    profile_name: str = "default"
    resume_path: str = ""
    style_guide_path: str = ""  # Example resume/cover letter to mimic style
    output_dir: str = "output"
    log_level: str = "INFO"
    screenshot_on_error: bool = True

    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)

    @field_validator("output_dir")
    @classmethod
    def ensure_output_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    def get_resume_path(self) -> Path:
        """Get resolved resume path."""
        path = Path(self.resume_path)
        if not path.exists():
            from job_applicator.exceptions import ResumeNotFoundError

            raise ResumeNotFoundError(f"Resume not found: {path}")
        return path
