"""Unit tests for the import-cookies cookie normalizer."""

from __future__ import annotations

from types import SimpleNamespace

from job_applicator.cli import _cookiejar_to_playwright, _normalize_cookie


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


def test_cookiejar_to_playwright_converts_cookie() -> None:
    """A browser_cookie3 / cookiejar cookie maps to a Playwright cookie dict."""
    ck = SimpleNamespace(
        name="li_at",
        value="abc",
        domain=".linkedin.com",
        path="/",
        secure=True,
        expires=1893456000,
    )
    assert _cookiejar_to_playwright(ck) == {
        "name": "li_at",
        "value": "abc",
        "domain": ".linkedin.com",
        "path": "/",
        "expires": 1893456000.0,
        "secure": True,
    }


def test_cookiejar_to_playwright_handles_missing_expiry() -> None:
    ck = SimpleNamespace(
        name="JSESSIONID",
        value="x",
        domain=".www.linkedin.com",
        path="/",
        secure=False,
        expires=None,
    )
    out = _cookiejar_to_playwright(ck)
    assert out is not None
    assert "expires" not in out
    assert out["secure"] is False
