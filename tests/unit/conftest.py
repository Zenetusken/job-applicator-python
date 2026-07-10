"""Unit-test-only fixtures (this conftest scopes to tests/unit/, so the live tier is unaffected)."""

from __future__ import annotations

import pytest

from job_applicator.exceptions import GroundingUnavailableError


@pytest.fixture(autouse=True)
def _grounding_verifier_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must never fire the live résumé grounding verifier.

    Default its client to the fail-safe path so résumé tests cannot make a real, slow,
    nondeterministic LLM call. Cover-letter entailment is inline and uses injected scorers in its
    unit tests; it does not route through this verifier.

    Tests that genuinely exercise the verifier override ``_get_client`` or ``verify`` on their own
    instance, which shadows this class-level patch. The live gold-set measurement lives at the
    tests/ root, outside this conftest, so it keeps the real verifier.
    """

    def _disabled(_self: object) -> None:
        raise GroundingUnavailableError("grounding verifier disabled in unit tests")

    monkeypatch.setattr(
        "job_applicator.documents.grounding_verifier.GroundingVerifier._get_client", _disabled
    )
