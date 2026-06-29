"""Unit tests for output-language resolution (utils.language)."""

from __future__ import annotations

from job_applicator.config import LLMConfig
from job_applicator.utils.language import detect_language, resolve_output_language

FR = (
    "Nous recherchons un analyste SOC pour surveiller les alertes de sécurité et gérer les "
    "incidents au sein de notre équipe à Montréal. Une expérience avec les systèmes SIEM est "
    "requise pour ce poste, ainsi que des compétences en réseau."
)
EN = (
    "We are looking for a SOC analyst to monitor security alerts and manage incidents within our "
    "team. Experience with SIEM systems and networking skills is required for this role."
)


def test_detect_language_french() -> None:
    assert detect_language(FR) == "fr"


def test_detect_language_english() -> None:
    assert detect_language(EN) == "en"


def test_detect_language_empty_defaults_english() -> None:
    assert detect_language("") == "en"


def test_resolve_auto_mirrors_jd() -> None:
    assert resolve_output_language("auto", FR) == "French"
    assert resolve_output_language("auto", EN) == "English"


def test_resolve_forced_overrides_detection() -> None:
    # A forced setting ignores the JD's language (so a misdetect can't override an explicit choice).
    assert resolve_output_language("en", FR) == "English"
    assert resolve_output_language("fr", EN) == "French"


def test_resolve_accepts_full_names() -> None:
    assert resolve_output_language("french", EN) == "French"
    assert resolve_output_language("English", FR) == "English"


def test_llm_config_language_defaults_to_auto() -> None:
    # Default "auto" so a packet mirrors the posting's language out of the box.
    assert LLMConfig().language == "auto"
