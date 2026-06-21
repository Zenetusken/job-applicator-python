"""Unit tests for the import-cookies cookie normalizer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from job_applicator.utils.cookies import _cookiejar_to_playwright, _normalize_cookie


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


def test_cookiejar_to_playwright_propagates_httponly() -> None:
    ck = SimpleNamespace(
        name="li_at",
        value="abc",
        domain=".linkedin.com",
        path="/",
        secure=True,
        expires=1893456000,
        _rest={"HttpOnly": ""},
    )
    out = _cookiejar_to_playwright(ck)
    assert out is not None
    assert out["httpOnly"] is True


def test_cookies_from_browser_raises_typed_cookieerror() -> None:
    """Increment 3: the browser-store reader raises a typed CookieError (a
    JobApplicatorError) instead of typer.Exit/console — so cli stays the only
    typer/console layer and the reader is unit-testable."""
    from job_applicator.exceptions import CookieError, JobApplicatorError
    from job_applicator.utils.cookies import _cookies_from_browser

    with pytest.raises(CookieError) as excinfo:
        _cookies_from_browser("definitely-not-a-browser", "linkedin.com")
    assert isinstance(excinfo.value, JobApplicatorError)


def test_site_specs_board_rules() -> None:
    """Increment 4: per-board cookie-import policy lives in utils.cookies. LinkedIn
    requires li_at (+ session flags + feed verify); Indeed requires nothing but
    prefers cf_clearance."""
    from job_applicator.utils.cookies import _site_specs

    specs = _site_specs()
    assert specs["linkedin"].required_cookie == "li_at"
    assert specs["linkedin"].session_flags is True
    assert specs["linkedin"].feed_verify is True
    assert specs["indeed"].required_cookie is None
    assert specs["indeed"].preferred_cookie == "cf_clearance"
    assert specs["indeed"].session_flags is False
