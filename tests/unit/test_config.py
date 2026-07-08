"""Unit tests for config."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_applicator.config import (
    AppSettings,
    BrowserConfig,
    CoverLetterConfig,
    LLMConfig,
    SkillConfig,
)


def test_browser_config_defaults() -> None:
    config = BrowserConfig()
    assert config.headless is True
    assert config.slow_mo == 0
    assert config.viewport_width == 1920


def test_skill_config_default_is_evidence_span() -> None:
    # Production default flipped keyword→evidence_span (2026-06-28): domain/language-general
    # grounding by default, after a live A/B showed keyword buried French-language SOC postings
    # (French coverage 30%→91%, zero software regression). The match/TUI/batch/apply paths all
    # read this, so the default IS the production behavior.
    assert SkillConfig().grounding_mode == "evidence_span"


def test_skill_config_env_override_to_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    # The legacy keyword mode stays selectable for opt-out / comparison.
    monkeypatch.setenv("JOB_APPLICATOR_SKILLS_GROUNDING_MODE", "keyword")
    assert SkillConfig().grounding_mode == "keyword"


def test_cover_letter_config_defaults_to_no_override() -> None:
    # No [cover_letter] override → all fields None.
    cl = CoverLetterConfig()
    assert cl.model is None and cl.api_base is None and cl.api_key is None


def test_cover_letter_llm_inherits_main_llm_when_unset() -> None:
    # Default: the cover-letter step uses [llm] UNCHANGED — same object, identical behaviour.
    settings = AppSettings()
    assert settings.cover_letter_llm() is settings.llm


def test_cover_letter_llm_applies_overrides_without_touching_main_llm() -> None:
    # An override routes ONLY the cover-letter step; unset fields inherit [llm]; [llm] is untouched.
    settings = AppSettings(cover_letter={"model": "big-prose-model"})  # type: ignore[arg-type]
    cl = settings.cover_letter_llm()
    assert cl.model == "big-prose-model"
    assert cl.api_base == settings.llm.api_base  # unset → inherits [llm]
    assert cl.api_key == settings.llm.api_key
    assert cl.temperature == settings.llm.temperature
    assert settings.llm.model != "big-prose-model"  # [llm] itself unchanged


def test_cover_letter_llm_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # Settable via env: route the prose step to a cloud model while [llm] stays local.
    monkeypatch.setenv("JOB_APPLICATOR_COVER_LETTER_MODEL", "claude-opus")
    monkeypatch.setenv("JOB_APPLICATOR_COVER_LETTER_API_BASE", "https://api.anthropic.example/v1")
    settings = AppSettings()
    cl = settings.cover_letter_llm()
    assert cl.model == "claude-opus"
    assert cl.api_base == "https://api.anthropic.example/v1"
    assert cl.api_key == settings.llm.api_key  # api_key unset → inherits [llm]


def test_llm_config_defaults() -> None:
    config = LLMConfig()
    # Default base model is the 8B (text-only AWQ, fits 12 GB, grounds stack-heavy JDs the 4B
    # couldn't); the 4B stays a pinnable fallback via JOB_APPLICATOR_LLM_MODEL.
    assert config.model == "Qwen/Qwen3-8B-AWQ"
    assert config.temperature == 0.7
    # Sized for full résumé tailoring (not the old 1024 cap).
    assert config.max_tokens == 4096
    # Phase-1 sampler migration is measure-only: optional knobs default to omitted request fields.
    assert config.top_p is None
    assert config.top_k is None
    assert config.min_p is None
    assert config.presence_penalty is None
    assert config.enable_thinking is False


def test_llm_sampler_config_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampler knobs are configurable before they become measured defaults."""
    toml = tmp_path / "config.toml"
    toml.write_text(
        "[llm]\n"
        "top_p = 0.8\n"
        "top_k = 20\n"
        "min_p = 0.0\n"
        "presence_penalty = 1.2\n"
        "enable_thinking = true\n"
    )
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(toml))

    config = AppSettings().llm

    assert config.top_p == 0.8
    assert config.top_k == 20
    assert config.min_p == 0.0
    assert config.presence_penalty == 1.2
    assert config.enable_thinking is True


def test_config_toml_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """config.toml values must actually populate AppSettings (incl. nested tables)."""
    toml = tmp_path / "config.toml"
    toml.write_text(
        'profile_name = "from_toml"\n'
        'resume_path = "/toml/resume.pdf"\n'
        "\n"
        "[llm]\n"
        'model = "toml-model"\n'
        "max_tokens = 2222\n"
        "\n"
        "[target]\n"
        'linkedin_email = "toml@example.com"\n'
    )
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(toml))
    settings = AppSettings()
    assert settings.profile_name == "from_toml"
    assert settings.resume_path == "/toml/resume.pdf"
    assert settings.llm.model == "toml-model"
    assert settings.llm.max_tokens == 2222
    assert settings.target.linkedin_email == "toml@example.com"


def test_env_overrides_config_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables take precedence over config.toml."""
    toml = tmp_path / "config.toml"
    toml.write_text('profile_name = "from_toml"\n')
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(toml))
    monkeypatch.setenv("JOB_APPLICATOR_PROFILE_NAME", "from_env")
    settings = AppSettings()
    assert settings.profile_name == "from_env"


def test_missing_config_toml_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-existent config file must not raise; defaults apply."""
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(tmp_path / "does_not_exist.toml"))
    settings = AppSettings()
    assert settings.profile_name == "default"


def test_app_settings_construction_has_no_filesystem_side_effect(tmp_path: object) -> None:
    """Constructing settings must NOT create the output dir (no validator side effects)."""
    import pathlib

    output = pathlib.Path(str(tmp_path)) / "test_output"
    AppSettings(output_dir=str(output))
    assert not output.exists()


def test_ensure_output_dir_creates_and_returns_path(tmp_path: object) -> None:
    """ensure_output_dir() explicitly creates the directory and returns it."""
    import pathlib

    output = pathlib.Path(str(tmp_path)) / "test_output"
    settings = AppSettings(output_dir=str(output))
    assert not output.exists()
    returned = settings.ensure_output_dir()
    assert returned == output
    assert output.exists()


def test_app_settings_browser_override() -> None:
    settings = AppSettings(browser=BrowserConfig(headless=False))
    assert settings.browser.headless is False


def test_env_override(monkeypatch: object) -> None:
    # type: ignore[arg-type]
    monkeypatch.setenv("JOB_APPLICATOR_LOG_LEVEL", "DEBUG")  # type: ignore[attr-defined]
    settings = AppSettings()
    assert settings.log_level == "DEBUG"


def test_output_default_format_is_txt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(tmp_path / "nonexistent.toml"))
    settings = AppSettings()
    assert settings.output.default_format == "txt"


def test_output_default_format_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(tmp_path / "nonexistent.toml"))
    monkeypatch.setenv("JOB_APPLICATOR_OUTPUT_DEFAULT_FORMAT", "pdf")
    settings = AppSettings()
    assert settings.output.default_format == "pdf"


def test_version_flag() -> None:
    """--version must report the package version and exit cleanly."""
    from typer.testing import CliRunner

    from job_applicator import __version__
    from job_applicator.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_pyproject_version_matches_runtime_version() -> None:
    """pyproject.toml version must match the runtime __version__."""
    import tomllib

    from job_applicator import __version__

    pyproject = Path("pyproject.toml").read_bytes()
    data = tomllib.loads(pyproject.decode("utf-8"))
    assert data["project"]["version"] == __version__


def test_get_settings_wraps_malformed_config_as_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed config.toml must surface as a typed ConfigError, not a raw TOMLDecodeError."""
    from job_applicator.cli import _get_settings
    from job_applicator.exceptions import ConfigError

    bad = tmp_path / "config.toml"
    bad.write_text("this is = not valid toml [[[")
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(bad))
    with pytest.raises(ConfigError):
        _get_settings()


def test_doctor_reports_malformed_config_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`doctor` (whose job is diagnosing config) must report a malformed config as a clean
    failure (exit 1, no escaped raw exception), not crash before it can run."""
    from typer.testing import CliRunner

    from job_applicator import cli

    bad = tmp_path / "config.toml"
    bad.write_text("nope = [[[")
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(bad))
    result = CliRunner().invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    # Caught + reported, not escaped as a raw exception (SystemExit = a clean typer.Exit).
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_check_session_reports_typed_error_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check-session must surface a typed error (bad config / no browser) as a clean exit 1, not
    a raw traceback — it previously ran asyncio.run bare with no JobApplicatorError wrapper."""
    from typer.testing import CliRunner

    from job_applicator import cli

    bad = tmp_path / "config.toml"
    bad.write_text("x = [[[")
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(bad))
    result = CliRunner().invoke(cli.app, ["check-session"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
