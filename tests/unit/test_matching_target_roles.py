"""[matching] target_roles — config validation + ranking-boost application.

The mechanism the 2026-07-02 gold-set calibration selected: deterministic title patterns
(embedding interest was undiscriminating within-domain), ordered first-match-wins, boost
clamped to 1.0, ranking-only. The calibration patterns themselves are pinned as a
regression test — including the decoy titles they must NOT fire on (zero false tags was
the measured property that made the mechanism shippable).
"""

from typing import ClassVar

import pydantic
import pytest

from job_applicator.config import EmbeddingConfig, MatchingConfig, TargetRoleRule
from job_applicator.embeddings.matching import JobMatcher, MatchResult
from job_applicator.models import JobBoard, JobListing


def _matcher(rules: list[TargetRoleRule]) -> JobMatcher:
    return JobMatcher(
        EmbeddingConfig(device="cpu", memory_limit_gb=0.5),
        matching=MatchingConfig(target_roles=rules),
    )


def _job(title: str) -> JobListing:
    return JobListing(
        title=title,
        company="Acme",
        url="https://example.com/jobs/1",
        board=JobBoard.LINKEDIN,
    )


class TestConfigValidation:
    def test_invalid_regex_rejected_at_load(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="invalid title_pattern"):
            TargetRoleRule(name="bad", title_pattern="[unclosed")

    def test_boost_bounds_enforced(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            TargetRoleRule(name="x", title_pattern="x", boost=0.9)
        with pytest.raises(pydantic.ValidationError):
            TargetRoleRule(name="x", title_pattern="x", boost=-0.1)

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="non-empty"):
            TargetRoleRule(name="  ", title_pattern="x")

    def test_default_config_is_empty_noop(self) -> None:
        assert MatchingConfig().target_roles == []


class TestBoostApplication:
    def test_no_rules_is_noop(self) -> None:
        m = _matcher([])
        assert m._apply_target_boost("AI Safety Expert - Red Team", 0.4) == (0.4, None)

    def test_match_boosts_and_tags(self) -> None:
        rule = TargetRoleRule(name="red-team", title_pattern=r"\bred[ -]?team\b", boost=0.15)
        score, tag = _matcher([rule])._apply_target_boost("AI Safety Expert - Red Team", 0.355)
        assert tag == "red-team"
        assert score == pytest.approx(0.505)

    def test_no_match_unchanged(self) -> None:
        rule = TargetRoleRule(name="red-team", title_pattern=r"\bred[ -]?team\b", boost=0.15)
        assert _matcher([rule])._apply_target_boost("Business Help Desk Specialist", 0.34) == (
            0.34,
            None,
        )

    def test_first_match_wins_ordered(self) -> None:
        m = _matcher(
            [
                TargetRoleRule(name="first", title_pattern="analyst", boost=0.05),
                TargetRoleRule(name="second", title_pattern="analyst", boost=0.30),
            ]
        )
        score, tag = m._apply_target_boost("Security Analyst", 0.5)
        assert tag == "first"
        assert score == pytest.approx(0.55)

    def test_boost_clamped_at_one(self) -> None:
        m = _matcher([TargetRoleRule(name="x", title_pattern="analyst", boost=0.5)])
        score, _ = m._apply_target_boost("Analyst", 0.9)
        assert score == 1.0

    def test_case_insensitive_and_accented(self) -> None:
        m = _matcher(
            [TargetRoleRule(name="iam", title_pattern=r"gestion des identit[ée]s", boost=0.1)]
        )
        _, tag = m._apply_target_boost("Analyste en GESTION DES IDENTITÉS et des accès", 0.4)
        assert tag == "iam"

    def test_match_result_default_untagged(self) -> None:
        r = MatchResult(
            job=_job("X"),
            score=0.5,
            semantic_score=0.5,
            skill_score=0.0,
            matched_skills=[],
            missing_skills=[],
            summary="",
        )
        assert r.target_role is None


class TestCalibrationPatterns:
    """The 2026-07-02 gold-set patterns: fire on the intended titles, NEVER on the decoys
    that made embedding-interest unusable (zero false tags was the shippable property)."""

    RULES: ClassVar[list[TargetRoleRule]] = [
        TargetRoleRule(
            name="red-team",
            title_pattern=r"\bred[ -]?team\b|\bpurple[ -]?team\b|\bai safety\b",
            boost=0.15,
        ),
        TargetRoleRule(
            name="iam",
            title_pattern=(
                r"\biam\b|\bidentity and access\b|\bgestion des identit[ée]s\b"
                r"|\bidentit[ée]s et des acc[èe]s\b"
            ),
            boost=0.15,
        ),
        TargetRoleRule(
            name="sysadmin",
            title_pattern=(
                r"\badministrat(or|eur|rice)\b|\bsystem administrat"
                r"|\badministrateur de syst|\bnetwork administrat"
            ),
            boost=0.04,
        ),
    ]

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("AI Safety Expert - Red Team", "red-team"),
            ("Analyste en gestion des identités et des accès", "iam"),
            ("Windows System Administrator", "sysadmin"),
            ("Administrateur Middleware", "sysadmin"),
            ("Network Administrator", "sysadmin"),
            # decoys — semantically adjacent titles that must NOT fire:
            ("Architecte de solution IA", None),
            ("Technical Support Specialist (Bilingual-French)", None),
            ("Ingénieur de données", None),
            ("Security Operations Center Analyst", None),  # already ranks on fit
            ("Gestionnaire des incidents", None),  # ITSM masquerade
        ],
    )
    def test_calibration_firing(self, title: str, expected: str | None) -> None:
        m = _matcher(self.RULES)
        _, tag = m._apply_target_boost(title, 0.4)
        assert tag == expected
