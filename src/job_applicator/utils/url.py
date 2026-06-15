"""Shared URL/host helpers for scrapers and cookie import."""

from __future__ import annotations


def host_matches(host: str, base: str) -> bool:
    """True if ``host`` is ``base`` or a subdomain of it (not a look-alike).

    e.g. base ``linkedin.com`` matches ``www.linkedin.com`` and the cookie-domain
    form ``.linkedin.com``, but NOT ``notlinkedin.com`` or
    ``linkedin.com.evil.example``. Both sides are lower-cased and any leading dot
    (cookie-domain notation) is stripped before comparison.
    """
    h = host.strip().lstrip(".").lower()
    b = base.strip().lstrip(".").lower()
    if not h or not b:
        return False
    return h == b or h.endswith("." + b)
