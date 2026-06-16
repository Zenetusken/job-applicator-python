"""Tests for host locale/timezone/user-agent detection."""

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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en_US.UTF-8", "en-US"),
        ("es_419", "es-419"),  # UN M.49 numeric region subtag
        ("zh_Hans_CN", "zh-CN"),  # script subtag skipped, region kept
        ("pt-BR", "pt-BR"),  # hyphen separator
        ("en", "en"),  # language only, no region
        ("de_DE@euro", "de-DE"),  # modifier stripped
        ("toolongtoken", "en-US"),  # not a valid language subtag -> default
        ("", "en-US"),  # empty -> default
    ],
)
def test_parse_locale_tolerates_varied_forms(raw: str, expected: str) -> None:
    assert region._parse_locale(raw) == expected


def test_detect_timezone_returns_iana_like_string() -> None:
    tz = region.detect_timezone()
    assert isinstance(tz, str) and "/" in tz  # e.g. America/Toronto


def test_detect_timezone_prefers_valid_tz_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "Europe/Paris")
    assert region.detect_timezone() == "Europe/Paris"


def test_detect_timezone_strips_posix_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    # The posix/ and right/ zoneinfo sub-trees resolve in the tz db but ICU
    # (Playwright's timezone_id) rejects them; they must be normalised away.
    monkeypatch.setenv("TZ", "posix/America/New_York")
    assert region.detect_timezone() == "America/New_York"


def test_detect_timezone_skips_invalid_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    # An invalid candidate is skipped in favour of the next valid source rather
    # than returned blindly (which would crash Playwright at launch).
    monkeypatch.setattr(
        region, "_timezone_candidates", lambda: iter(["Not/AZone", "Europe/Berlin"])
    )
    assert region.detect_timezone() == "Europe/Berlin"


def test_detect_timezone_falls_back_when_all_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    # When no candidate validates, fall back to the default — never emit garbage.
    monkeypatch.setattr(region, "_timezone_candidates", lambda: iter(["Not/AZone", "posix/Bogus"]))
    assert region.detect_timezone() == region._DEFAULT_TIMEZONE


@pytest.mark.parametrize("bad", ["US", "America", "Etc", "Not/AZone", "posix"])
def test_detect_timezone_rejects_directory_and_bogus_keys(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    # Keys that resolve to a zoneinfo directory (US/America) or don't exist must
    # never be returned — they'd crash Playwright's timezone_id at launch.
    monkeypatch.setattr(region, "_timezone_candidates", lambda: iter([bad]))
    assert region.detect_timezone() == region._DEFAULT_TIMEZONE


def test_normalize_zone_strips_prefixes_and_colon() -> None:
    assert region._normalize_zone(":America/Toronto") == "America/Toronto"
    assert region._normalize_zone("right/Europe/London") == "Europe/London"
    assert region._normalize_zone("posix/America/New_York") == "America/New_York"


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("Google Chrome 149.0.6723.91", "149"),
        ("Chromium 130.0.6723.91", "130"),
        # Brave prints its own 1.x version first, then the real Chromium major;
        # the regex must pick 130, not the leading "71".
        ("Brave Browser 1.71.123 Chromium: 130.0.6723.91", "130"),
        ("totally unparseable", None),
    ],
)
def test_detect_chrome_major_parses_version(
    monkeypatch: pytest.MonkeyPatch, stdout: str, expected: str | None
) -> None:
    import subprocess

    region._detect_chrome_major.cache_clear()
    monkeypatch.setattr(region.shutil, "which", lambda exe: "/usr/bin/" + exe)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(region.subprocess, "run", fake_run)
    region._detect_chrome_major.cache_clear()
    assert region._detect_chrome_major() == expected
    region._detect_chrome_major.cache_clear()


def test_timezone_country_maps_via_zone_tab() -> None:
    # America/Toronto resolves to CA via the OS zone1970.tab on Linux/macOS.
    country = region._timezone_country("America/Toronto")
    assert country in ("CA", "")  # CA where the tz db ships zone1970.tab; "" on Windows


@pytest.mark.parametrize(
    ("country", "expected"),
    [
        ("CA", "ca.indeed.com"),
        ("US", "www.indeed.com"),
        ("GB", "uk.indeed.com"),  # Indeed uses uk, not gb
        ("DE", "de.indeed.com"),
        ("", "www.indeed.com"),  # unknown → US default
        # Countries with NO Indeed site must NOT produce a dead <cc>.indeed.com —
        # they fall back to www.indeed.com (which always resolves).
        ("BS", "www.indeed.com"),  # Bahamas: bs.indeed.com does not exist
        ("IS", "www.indeed.com"),  # Iceland: is.indeed.com does not exist
    ],
)
def test_detect_indeed_domain_from_country(
    monkeypatch: pytest.MonkeyPatch, country: str, expected: str
) -> None:
    monkeypatch.setattr(region, "detect_timezone", lambda: "Some/Zone")
    monkeypatch.setattr(region, "_timezone_country", lambda tz: country)
    assert region.detect_indeed_domain() == expected


def test_detect_chrome_user_agent_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    region._detect_chrome_major.cache_clear()
    region.detect_chrome_user_agent.cache_clear()
    monkeypatch.setattr(region.shutil, "which", lambda exe: None)  # no browser found
    assert region.detect_chrome_user_agent() == region._DEFAULT_USER_AGENT
    region._detect_chrome_major.cache_clear()
    region.detect_chrome_user_agent.cache_clear()
