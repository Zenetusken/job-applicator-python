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
