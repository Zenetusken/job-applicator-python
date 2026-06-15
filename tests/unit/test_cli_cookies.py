"""Unit tests for the import-cookies cookie normalizer."""

from __future__ import annotations

from job_applicator.cli import _normalize_cookie


def test_normalize_extension_format() -> None:
    """Browser-extension fields (expirationDate, sameSite) map to Playwright's."""
    out = _normalize_cookie(
        {
            "name": "li_at",
            "value": "abc",
            "domain": ".linkedin.com",
            "path": "/",
            "expirationDate": 1893456000.0,
            "secure": True,
            "httpOnly": True,
            "sameSite": "no_restriction",
        }
    )
    assert out == {
        "name": "li_at",
        "value": "abc",
        "domain": ".linkedin.com",
        "path": "/",
        "expires": 1893456000.0,
        "httpOnly": True,
        "secure": True,
        "sameSite": "None",
    }


def test_normalize_requires_name_and_value() -> None:
    assert _normalize_cookie({"value": "x"}) is None
    assert _normalize_cookie({"name": "li_at"}) is None
    assert _normalize_cookie("not a dict") is None


def test_normalize_falls_back_to_url_without_domain() -> None:
    out = _normalize_cookie({"name": "li_at", "value": "abc"})
    assert out is not None
    assert out["url"] == "https://www.linkedin.com"
    assert "domain" not in out
