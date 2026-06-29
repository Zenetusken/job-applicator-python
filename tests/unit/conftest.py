"""Unit-test-only fixtures (this conftest scopes to tests/unit/, so the live tier is unaffected)."""

from __future__ import annotations

import pytest

from job_applicator.exceptions import GroundingUnavailableError


@pytest.fixture(autouse=True)
def _grounding_verifier_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must never fire the live grounding verifier — that would be a real LLM call, slow
    and nondeterministic. Default it to the fail-safe path: ``verify()`` raises
    ``GroundingUnavailableError``, so ``generate_verified()`` passes straight through to
    ``generate()`` (the behaviour every existing cover-letter test expects).

    Tests that genuinely exercise the verifier override ``_get_client`` or ``verify`` on their own
    instance, which shadows this class-level patch. The live gold-set measurement lives at the
    tests/ root, outside this conftest, so it keeps the real verifier.
    """

    def _disabled(_self: object) -> None:
        raise GroundingUnavailableError("grounding verifier disabled in unit tests")

    monkeypatch.setattr(
        "job_applicator.documents.grounding_verifier.GroundingVerifier._get_client", _disabled
    )
