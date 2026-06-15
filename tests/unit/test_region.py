"""Tests for host locale/timezone detection."""

from __future__ import annotations

import pytest

from job_applicator.utils import region


def test_detect_locale_from_lang(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("locale.getlocale", lambda: (None, None))
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.setenv("LANG", "fr_CA.UTF-8")
    assert region.detect_locale() == "fr-CA"


def test_detect_locale_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("locale.getlocale", lambda: (None, None))
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.setenv("LANG", "C")
    assert region.detect_locale() == "en-US"


def test_detect_timezone_returns_iana_like_string() -> None:
    tz = region.detect_timezone()
    assert isinstance(tz, str) and "/" in tz  # e.g. America/Toronto
