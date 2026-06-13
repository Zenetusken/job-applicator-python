"""Unit tests for config."""

from __future__ import annotations

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
