"""ATS compatibility checker for resumes."""

from __future__ import annotations

import re

from job_applicator.models import ATSCompatibilityResult, ResumeData
from job_applicator.utils.logging import get_logger

logger = get_logger("documents.ats_checker")

_MIN_TEXT_LENGTH = 200
_STANDARD_SECTIONS = {"experience", "education", "skills"}
_OPTIONAL_SECTIONS = {"certifications", "languages"}
_TABLE_PATTERN = re.compile(r"\+[-]+\+")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_PATTERN = re.compile(r"[\+]?[\d\s\-\(\)]{10,}")


def _phone_has_digits(phone: str) -> bool:
    """Return True if phone string contains at least 10 actual digits."""
    if not phone:
        return False
    for match in _PHONE_PATTERN.finditer(phone):
        if sum(c.isdigit() for c in match.group(0)) >= 10:
            return True
    return False


class ATSChecker:
    """Check resume compatibility with Applicant Tracking Systems."""

    def check(self, resume: ResumeData) -> ATSCompatibilityResult:
        """Run all ATS compatibility checks and return results."""
        checks: list[dict[str, object]] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        self._check_email(resume, checks, warnings)
        self._check_phone(resume, checks, warnings)
        self._check_sections(resume, checks, warnings)
        self._check_optional_sections(resume, checks)
        self._check_text_length(resume, checks, warnings)
        self._check_tables(resume, checks, warnings)

        score = self._calculate_score(checks)
        is_compatible = score >= 0.6

        self._generate_suggestions(checks, suggestions)

        logger.info("ATS check complete: score=%.2f, compatible=%s", score, is_compatible)
        return ATSCompatibilityResult(
            score=score,
            checks=checks,
            warnings=warnings,
            suggestions=suggestions,
        )

    def _check_email(
        self,
        resume: ResumeData,
        checks: list[dict[str, object]],
        warnings: list[str],
    ) -> None:
        has_email = bool(resume.email and _EMAIL_PATTERN.search(resume.email))
        checks.append({"name": "email_present", "passed": has_email, "details": "Email address"})
        if not has_email:
            warnings.append("No email address found. ATS requires contact information.")

    def _check_phone(
        self,
        resume: ResumeData,
        checks: list[dict[str, object]],
        warnings: list[str],
    ) -> None:
        has_phone = _phone_has_digits(resume.phone)
        checks.append({"name": "phone_present", "passed": has_phone, "details": "Phone number"})
        if not has_phone:
            warnings.append("No phone number found. ATS requires contact information.")

    def _check_sections(
        self,
        resume: ResumeData,
        checks: list[dict[str, object]],
        warnings: list[str],
    ) -> None:
        text_lower = resume.raw_text.lower()
        for section in _STANDARD_SECTIONS:
            found = section in text_lower
            checks.append(
                {
                    "name": f"{section}_section",
                    "passed": found,
                    "details": f"'{section.title()}' section header",
                }
            )
            if not found:
                msg = f"Missing '{section.title()}' section. ATS expects standard headers."
                warnings.append(msg)

    def _check_optional_sections(
        self,
        resume: ResumeData,
        checks: list[dict[str, object]],
    ) -> None:
        text_lower = resume.raw_text.lower()
        for section in _OPTIONAL_SECTIONS:
            found = section in text_lower
            checks.append(
                {
                    "name": f"{section}_section",
                    "passed": found,
                    "details": f"'{section.title()}' section (optional)",
                }
            )

    def _check_text_length(
        self,
        resume: ResumeData,
        checks: list[dict[str, object]],
        warnings: list[str],
    ) -> None:
        text_len = len(resume.raw_text.strip())
        is_long_enough = text_len >= _MIN_TEXT_LENGTH
        checks.append(
            {
                "name": "text_length",
                "passed": is_long_enough,
                "details": f"Text length: {text_len} chars (min: {_MIN_TEXT_LENGTH})",
            }
        )
        if not is_long_enough:
            msg = f"Resume text is too short ({text_len} chars). ATS needs sufficient content."
            warnings.append(msg)

    def _check_tables(
        self,
        resume: ResumeData,
        checks: list[dict[str, object]],
        warnings: list[str],
    ) -> None:
        has_tables = bool(_TABLE_PATTERN.search(resume.raw_text))
        no_tables = not has_tables
        checks.append({"name": "no_tables", "passed": no_tables, "details": "No ASCII tables"})
        if has_tables:
            warnings.append("ASCII table formatting detected. ATS may not parse tables correctly.")

    def _calculate_score(self, checks: list[dict[str, object]]) -> float:
        if not checks:
            return 0.0
        # Only count required checks (not optional sections) toward score
        required_sections = {"experience_section", "education_section", "skills_section"}
        required = [
            c
            for c in checks
            if not str(c["name"]).endswith("_section") or c["name"] in required_sections
        ]
        passed = sum(1 for c in required if c["passed"])
        return round(passed / len(required), 2)

    def _generate_suggestions(
        self,
        checks: list[dict[str, object]],
        suggestions: list[str],
    ) -> None:
        failed = [c for c in checks if not c["passed"]]
        for check in failed:
            name = check["name"]
            if name == "email_present":
                suggestions.append("Add a professional email address to your contact info.")
            elif name == "phone_present":
                suggestions.append("Add a phone number to your contact info.")
            elif name == "experience_section":
                suggestions.append("Add an 'Experience' section with standard header.")
            elif name == "education_section":
                suggestions.append("Add an 'Education' section with standard header.")
            elif name == "skills_section":
                suggestions.append("Add a 'Skills' section listing your technical skills.")
            elif name == "text_length":
                suggestions.append("Expand your resume with more detail about your experience.")
            elif name == "no_tables":
                suggestions.append("Replace table formatting with plain text lists.")
            elif name == "certifications_section":
                suggestions.append("Add a 'Certifications' section if you have relevant certs.")
            elif name == "languages_section":
                suggestions.append("Add a 'Languages' section to highlight language skills.")
