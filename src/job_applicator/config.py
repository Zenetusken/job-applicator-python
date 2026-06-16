"""Centralized configuration — single source of truth.

Loaded from config.toml + JOB_APPLICATOR_* env vars.
Validated immediately at startup via Pydantic Settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# Path to the TOML config file. Overridable via JOB_APPLICATOR_CONFIG_FILE so
# tests and alternate deployments can point at a different file.
CONFIG_FILE_ENV_VAR = "JOB_APPLICATOR_CONFIG_FILE"
DEFAULT_CONFIG_FILE = "config.toml"


class BrowserConfig(BaseSettings):
    """Playwright browser options."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_BROWSER_")

    headless: bool = True
    slow_mo: int = 0
    viewport_width: int = 1920
    viewport_height: int = 1080
    timeout_ms: int = 30_000
    user_agent: str | None = None
    # Empty = auto-detect from the host. The timezone in particular drives how
    # geo-aware sites (Indeed/LinkedIn) locate the browser; set to pin a region.
    locale: str = ""
    timezone: str = ""


class LLMConfig(BaseSettings):
    """LLM API configuration for cover letter generation."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_LLM_")

    api_base: str = "http://localhost:8000/v1"
    api_key: str = "not-needed-for-local"
    model: str = "cyankiwi/Qwen3.5-4B-AWQ-4bit"
    # Upper bound for a single completion. Sized for full résumé tailoring;
    # cover letters and style analysis stay well under this cap.
    max_tokens: int = 4096
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
    # Empty = auto-detect the regional Indeed host from the host timezone (e.g.
    # ca.indeed.com in Canada). Set to pin one explicitly, e.g. "ca.indeed.com" /
    # "uk.indeed.com" / "www.indeed.com".
    indeed_domain: str = ""


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load values from config.toml in addition to env vars.

        Priority (highest first): explicit init args > environment variables >
        .env file > file secrets > config.toml. The TOML file only fills in
        values that were not supplied by a higher-priority source.
        """
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ]
        toml_file = os.environ.get(CONFIG_FILE_ENV_VAR, DEFAULT_CONFIG_FILE)
        if Path(toml_file).is_file():
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=Path(toml_file)))
        return tuple(sources)

    def ensure_output_dir(self) -> Path:
        """Create the output directory if needed and return it.

        Directory creation is an explicit, opt-in side effect performed by
        callers right before they write output — constructing settings must
        stay free of filesystem side effects.
        """
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_resume_path(self) -> Path:
        """Get resolved resume path."""
        path = Path(self.resume_path)
        if not path.exists():
            from job_applicator.exceptions import ResumeNotFoundError

            raise ResumeNotFoundError(f"Resume not found: {path}")
        return path
