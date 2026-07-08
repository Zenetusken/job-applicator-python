"""Centralized configuration — single source of truth.

Loaded from config.toml + JOB_APPLICATOR_* env vars.
Validated immediately at startup via Pydantic Settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from job_applicator.utils.path import set_owner_only

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
    # Browser engine channel. "chrome" launches the host's REAL Google Chrome (Playwright's own
    # matching channel) instead of the bundled Chromium: no `HeadlessChrome` client-hint leak,
    # real-GPU WebGL, and UA == Sec-CH-UA (all one version). Falls back to bundled Chromium (with a
    # warning) if no host Chrome is installed. Set to None/"" to force the bundled engine.
    channel: str | None = "chrome"
    # Empty = auto-detect from the host. The timezone in particular drives how
    # geo-aware sites (Indeed/LinkedIn) locate the browser; set to pin a region.
    locale: str = ""
    timezone: str = ""


class LLMConfig(BaseSettings):
    """LLM API configuration for cover letter generation."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_LLM_")

    api_base: str = "http://localhost:8000/v1"
    api_key: str = "not-needed-for-local"
    # Qwen3-8B-AWQ (genuine AWQ 4-bit, text-only, ~6.1 GB) — fits the 12 GB card alongside the
    # embeddings and grounds stack-heavy job descriptions the 4B couldn't (measured: cover-letter
    # employer-stack overclaim 5/6 → 0/5). The 4B (cyankiwi/Qwen3.5-4B-AWQ-4bit) stays a smaller,
    # faster fallback you can pin via JOB_APPLICATOR_LLM_MODEL / [llm] model.
    model: str = "Qwen/Qwen3-8B-AWQ"
    # Upper bound for a single completion. Sized for full résumé tailoring;
    # cover letters and style analysis stay well under this cap.
    max_tokens: int = 4096
    temperature: float = 0.7
    # Optional sampler knobs. Defaults intentionally preserve the previous request shape; set these
    # for measured Qwen/vLLM tuning without changing task-specific temperature overrides.
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    enable_thinking: bool = False
    # Output language for generated documents: "auto" (mirror the job posting's language), "en",
    # or "fr". Lives on [llm] so the cover-letter override (cover_letter_llm) inherits it — the CV
    # and the cover letter always resolve the SAME language, so one application never mixes them.
    # Resolved + logged per job by utils.language.resolve_output_language.
    language: str = "auto"


class LLMResilienceConfig(BaseSettings):
    """Circuit-breaker + content-retry policy for all LLM consumers.

    Process-wide resilience policy (governs cover-letter generation, résumé
    tailoring, and style analysis alike) — kept separate from the LLM *connection*
    config above. Defaults preserve prior hardcoded behavior.
    """

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_LLM_RESILIENCE_")

    failure_threshold: int = 3
    window_seconds: float = 60.0
    recovery_timeout_seconds: float = 30.0
    validation_max_retries: int = 1  # feeds ValidatedOutput(max_retries=...)


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration for semantic matching."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_EMBEDDING_")

    model_name: str = "mixedbread-ai/mxbai-embed-large-v1"
    device: str = "cuda"
    batch_size: int = 32
    max_seq_length: int = 512
    memory_limit_gb: float = 1.3  # minimum free VRAM preflight budget for CUDA embeddings
    normalize_embeddings: bool = True


class SkillConfig(BaseSettings):
    """Skill-extraction / grounding policy.

    ``grounding_mode`` selects how an LLM-extracted skill is verified against the source text:
    - ``evidence_span`` (default): the model returns the exact source phrase per skill and we
      verify that span occurs in the text — domain- AND language-general.
    - ``keyword``: substring + tech-tuned compound/stopword heuristics (software-only). The
      legacy mode; kept for opt-out / comparison.

    Default flipped keyword→evidence_span on 2026-06-28 after a live A/B on 42 real Montréal JDs:
    keyword left 70% of French-language postings coverage-blind (skills dropped → semantic-only →
    buried), evidence_span recovered them (French coverage 30%→91%) with zero regression on what
    keyword already grounded. See ``docs/compose/specs/2026-06-26-semantic-skill-grounding.md``.
    """

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_SKILLS_")

    grounding_mode: Literal["keyword", "evidence_span"] = "evidence_span"


class TargetRoleRule(BaseModel):
    """One declared target-role family: a job-title pattern and its ranking boost.

    A RANKING-ONLY preference signal — it never enters generated documents (no honesty
    surface). Deterministic title regex, not embeddings: measured on the 44-job gold set
    (2026-07-02), embedding interest against role phrases was UNDISCRIMINATING within-domain
    (a true AI-red-team job scored 0.635 vs 0.636 for support-at-a-security-vendor), while
    title patterns fired on exactly the intended rows with zero false tags.
    """

    name: str
    title_pattern: str
    boost: float = Field(default=0.10, ge=0.0, le=0.5)

    @field_validator("title_pattern")
    @classmethod
    def _pattern_compiles(cls, v: str) -> str:
        """Fail at config load (typed ConfigError via _get_settings), not at first match."""
        import re

        try:
            re.compile(v, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid title_pattern regex {v!r}: {exc}") from exc
        return v

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("target role name must be non-empty")
        return v.strip()


class MatchingConfig(BaseSettings):
    """Ranking-preference knobs layered over the fit score.

    ``target_roles``: ordered rules (FIRST match wins); a job whose TITLE matches gets
    ``boost`` added to its combined score (clamped to 1.0) and carries the rule's name as
    ``MatchResult.target_role``. Use it to rescue preference-important role families the
    CV is lexically far from (measured: AI-red-team / IAM ranked below the review floor on
    an SOC CV — fit is honest, preference needs its own signal), or to order same-fit
    families (a small sysadmin boost separates admin from help-desk). The boosted score is
    what ``match`` persists to the funnel — the stored ranking IS the preference-adjusted
    one."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_MATCHING_")

    target_roles: list[TargetRoleRule] = Field(default_factory=list)


class TargetConfig(BaseSettings):
    """Job board target settings."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_TARGET_")

    max_applications_per_day: int = 20
    delay_between_applications_s: float = 2.0
    # Proactive SEARCH-volume budget (anti-detection): keep an authenticated session unremarkable.
    # A per-day cap on scrapes (the top risk from the anti-detection audit — searches were uncapped
    # in code) + an optional inter-search cooldown (seconds; 0 = off) paced between scrapes.
    max_searches_per_day: int = 30
    search_cooldown_s: float = 0.0
    linkedin_email: str = ""
    linkedin_password: str = ""
    indeed_email: str = ""
    indeed_password: str = ""
    # Empty = auto-detect the regional Indeed host from the host timezone (e.g.
    # ca.indeed.com in Canada). Set to pin one explicitly, e.g. "ca.indeed.com" /
    # "uk.indeed.com" / "www.indeed.com".
    indeed_domain: str = ""


class OutputConfig(BaseSettings):
    """Default output format and template selection for generated artifacts."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_OUTPUT_")

    default_format: Literal["txt", "pdf", "both"] = "txt"
    resume_template: str = "modern"
    cover_letter_template: str = "modern"
    template_dir: Path | None = None


class CoverLetterConfig(BaseSettings):
    """Optional model override for the cover-letter step.

    The local 4B is excellent for the structured/honesty work (skill extraction, the overclaim
    guards) but leans on ornate, generic prose. Set any of these to route ONLY cover-letter
    generation (the prose + its PDF formatting) to a different model or endpoint — a larger local
    model, or a cloud API — while the rest of the pipeline stays on ``[llm]``. Each unset field
    inherits ``[llm]``, so the default behaviour is unchanged."""

    model_config = SettingsConfigDict(env_prefix="JOB_APPLICATOR_COVER_LETTER_")

    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None


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
    llm_resilience: LLMResilienceConfig = Field(default_factory=LLMResilienceConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    skills: SkillConfig = Field(default_factory=SkillConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cover_letter: CoverLetterConfig = Field(default_factory=CoverLetterConfig)

    def cover_letter_llm(self) -> LLMConfig:
        """LLMConfig for the cover-letter step: ``[llm]`` with any ``[cover_letter]`` overrides
        (model / api_base / api_key) applied. Returns ``[llm]`` unchanged when nothing is
        overridden, so default behaviour is identical to before this option existed."""
        overrides = {
            field: value
            for field, value in (
                ("model", self.cover_letter.model),
                ("api_base", self.cover_letter.api_base),
                ("api_key", self.cover_letter.api_key),
            )
            if value is not None
        }
        return self.llm.model_copy(update=overrides) if overrides else self.llm

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
        set_owner_only(path, 0o700)  # tailored résumés / cover letters are the user's data
        return path

    def get_resume_path(self) -> Path:
        """Get resolved resume path."""
        path = Path(self.resume_path)
        if not path.exists():
            from job_applicator.exceptions import ResumeNotFoundError

            raise ResumeNotFoundError(f"Resume not found: {path}")
        return path
