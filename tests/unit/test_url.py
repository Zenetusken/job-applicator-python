"""Tests for the shared host-matching helper."""

from __future__ import annotations

from job_applicator.utils.url import host_matches


def test_host_matches_exact_and_subdomain() -> None:
    assert host_matches("indeed.com", "indeed.com") is True
    assert host_matches("ca.indeed.com", "indeed.com") is True
    assert host_matches("www.linkedin.com", "linkedin.com") is True


def test_host_matches_strips_leading_dot() -> None:
    # cookie-domain notation (".indeed.com") must match the bare base
    assert host_matches(".indeed.com", "indeed.com") is True
    assert host_matches(".www.linkedin.com", "linkedin.com") is True
    # ...and the base may also be given in dotted/upper form
    assert host_matches("CA.Indeed.com", ".indeed.com") is True


def test_host_matches_rejects_lookalikes() -> None:
    assert host_matches("notindeed.com", "indeed.com") is False
    assert host_matches("indeed.com.evil.example", "indeed.com") is False
    assert host_matches("notlinkedin.com", "linkedin.com") is False


def test_host_matches_rejects_empty() -> None:
    assert host_matches("", "indeed.com") is False
    assert host_matches("indeed.com", "") is False
