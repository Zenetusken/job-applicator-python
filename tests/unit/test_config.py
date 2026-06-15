"""Unit tests for config."""

from __future__ import annotations

from pathlib import Path

import pytest

from job_applicator.config import AppSettings, BrowserConfig, LLMConfig


def test_browser_config_defaults() -> None:
    config = BrowserConfig()
    assert config.headless is True
    assert config.slow_mo == 0
    assert config.viewport_width == 1920


def test_llm_config_defaults() -> None:
    config = LLMConfig()
    assert config.model == "cyankiwi/Qwen3.5-4B-AWQ-4bit"
    assert config.temperature == 0.7
    # Sized for full résumé tailoring (not the old 1024 cap).
    assert config.max_tokens == 4096


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


def test_app_settings_output_dir_created(tmp_path: object) -> None:
    import pathlib

    output = pathlib.Path(str(tmp_path)) / "test_output"
    AppSettings(output_dir=str(output))
    assert output.exists()


def test_app_settings_browser_override() -> None:
    settings = AppSettings(browser=BrowserConfig(headless=False))
    assert settings.browser.headless is False


def test_env_override(monkeypatch: object) -> None:
    # type: ignore[arg-type]
    monkeypatch.setenv("JOB_APPLICATOR_LOG_LEVEL", "DEBUG")  # type: ignore[attr-defined]
    settings = AppSettings()
    assert settings.log_level == "DEBUG"
