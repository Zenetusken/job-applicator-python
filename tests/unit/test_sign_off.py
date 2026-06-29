"""Empirical tests for cover-letter sign-off behavior.

These tests drive the implementation: they pin what a valid sign-off looks like,
how the user's name is resolved, and how invalid sign-offs are rejected.
"""

from __future__ import annotations

import pytest

from job_applicator.exceptions import LLMError
from job_applicator.models import UserProfile
from job_applicator.utils.profile import _load_user_profile


class TestExtractSignOff:
    """Cover-letter sign-off extraction from free-form text."""

    @pytest.mark.parametrize(
        ("closing", "signature"),
        [
            ("Sincerely,", "Andrei Petrov"),
            ("Best regards,", "Andrei Petrov"),
            ("Regards,", "A. Petrov"),
            ("Warm regards,", "ANDREI PETROV"),
            ("Yours truly,", "Andrei Petrov"),
            ("Respectfully,", "Andrei Petrov"),
            ("Thank you,", "Andrei Petrov"),
            # French closings — a French packet for a French posting must be recognized too,
            # else extract_sign_off returns None and the PDF formatter rejects the letter.
            ("Cordialement,", "Andrei Petrov"),
            ("Bien cordialement,", "Andrei Petrov"),
            ("Salutations distinguées,", "Andrei Petrov"),
        ],
    )
    def test_extracts_common_two_line_sign_offs(self, closing: str, signature: str) -> None:
        from job_applicator.documents.sign_off import extract_sign_off

        text = f"Dear Manager,\n\nBody.\n\n{closing}\n{signature}"
        result = extract_sign_off(text)
        assert result is not None
        assert result[1] == signature

    @pytest.mark.parametrize(
        "closing_line",
        [
            "Sincerely, Andrei Petrov",
            "Best regards, Andrei Petrov",
            "Regards, A. Petrov",
            "Cordialement, Andrei Petrov",
        ],
    )
    def test_extracts_common_single_line_sign_offs(self, closing_line: str) -> None:
        from job_applicator.documents.sign_off import extract_sign_off

        text = f"Dear Manager,\n\nBody.\n\n{closing_line}"
        result = extract_sign_off(text)
        assert result is not None
        assert "Andrei" in result[1] or "Petrov" in result[1]

    def test_returns_none_when_sign_off_missing(self) -> None:
        from job_applicator.documents.sign_off import extract_sign_off

        assert extract_sign_off("Dear Manager,\n\nBody.\n\nThanks.") is None

    def test_returns_none_for_single_line_letter(self) -> None:
        from job_applicator.documents.sign_off import extract_sign_off

        assert extract_sign_off("Hello") is None


class TestValidateSignOff:
    """Hard validation rules for generated sign-offs."""

    def test_accepts_full_name_signature(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Andrei",
            last_name="Petrov",
            email="a@example.com",
            phone="",
        )
        validate_sign_off("...\n\nSincerely,\nAndrei Petrov", user)

    def test_accepts_full_name_with_punctuation(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Andrei",
            last_name="Petrov",
            email="a@example.com",
            phone="",
        )
        validate_sign_off("...\n\nSincerely,\nAndrei Petrov.", user)

    def test_accepts_reordered_full_name_signature(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Andrei",
            last_name="Petrov",
            email="a@example.com",
            phone="",
        )
        validate_sign_off("...\n\nSincerely,\nPetrov, Andrei", user)

    def test_rejects_first_name_only_when_last_name_known(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Andrei",
            last_name="Petrov",
            email="a@example.com",
            phone="",
        )
        with pytest.raises(LLMError, match="signed as"):
            validate_sign_off("...\n\nBest,\nAndrei", user)

    def test_accepts_only_known_name_part(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Andrei",
            last_name="",
            email="a@example.com",
            phone="",
        )
        validate_sign_off("...\n\nBest,\nAndrei", user)

    def test_rejects_substring_name_match(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Sam",
            last_name="Sample",
            email="a@example.com",
            phone="",
        )
        with pytest.raises(LLMError, match="signed as"):
            validate_sign_off("...\n\nSincerely,\nSamantha Smithson", user)
        with pytest.raises(LLMError, match="signed as"):
            validate_sign_off("...\n\nSincerely,\nSamir Sampleton", user)

    def test_rejects_missing_sign_off(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Andrei",
            last_name="Petrov",
            email="a@example.com",
            phone="",
        )
        with pytest.raises(LLMError, match="missing a proper sign-off"):
            validate_sign_off("Dear Manager,\n\nBody.\n\nThanks.", user)

    def test_rejects_wrong_name_signature(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Andrei",
            last_name="Petrov",
            email="a@example.com",
            phone="",
        )
        with pytest.raises(LLMError, match="signed as"):
            validate_sign_off("...\n\nSincerely,\nSamir Patel", user)

    def test_allows_when_user_name_unknown(self) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="",
            last_name="",
            email="a@example.com",
            phone="",
        )
        validate_sign_off("...\n\nSincerely,\nAnyone", user)

    @pytest.mark.parametrize(
        "closing",
        [
            "Sincerely,",
            "Best,",
            "Regards,",
            "Warm regards,",
            "Best regards,",
        ],
    )
    def test_accepts_style_guide_example_closings(self, closing: str) -> None:
        from job_applicator.documents.sign_off import validate_sign_off

        user = UserProfile(
            first_name="Jordan",
            last_name="Miller",
            email="a@example.com",
            phone="",
        )
        validate_sign_off(f"...\n\n{closing}\nJordan Miller", user)


class TestLoadUserProfileNameFallback:
    """The cover-letter signature must derive from the actual résumé by default."""

    def test_uses_profile_name_when_set(self) -> None:
        from job_applicator.config import AppSettings

        settings = AppSettings(profile_name="Jane Doe")
        profile = _load_user_profile(settings, resume_name="Wrong Name")
        assert profile.first_name == "Jane"
        assert profile.last_name == "Doe"

    def test_falls_back_to_resume_name_when_profile_default(self) -> None:
        from job_applicator.config import AppSettings

        settings = AppSettings(profile_name="default")
        profile = _load_user_profile(settings, resume_name="Andrei Petrov")
        assert profile.first_name == "Andrei"
        assert profile.last_name == "Petrov"

    def test_falls_back_to_resume_name_when_profile_empty(self) -> None:
        from job_applicator.config import AppSettings

        settings = AppSettings(profile_name="")
        profile = _load_user_profile(settings, resume_name="Andrei Petrov")
        assert profile.first_name == "Andrei"
        assert profile.last_name == "Petrov"

    def test_falls_back_to_user_when_nothing_available(self) -> None:
        from job_applicator.config import AppSettings

        settings = AppSettings(profile_name="")
        profile = _load_user_profile(settings, resume_name="")
        assert profile.first_name == "User"
        assert profile.last_name == ""

    def test_handles_whitespace_only_profile_name(self) -> None:
        from job_applicator.config import AppSettings

        settings = AppSettings(profile_name="   ")
        profile = _load_user_profile(settings, resume_name="Andrei Petrov")
        assert profile.first_name == "Andrei"
        assert profile.last_name == "Petrov"
