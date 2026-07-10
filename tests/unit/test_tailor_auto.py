"""Tests for unattended source-overlay tailoring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from job_applicator.models import GroundingReport, ResumeData, TailoredResume
from job_applicator.workflows.tailor_auto import tailor_auto_verified_saveable


async def test_auto_tailor_passes_user_summary_preference_without_policy_prefix() -> None:
    tailored = TailoredResume(
        original_path="",
        tailored_text="generated",
        job_title="Support Analyst",
        job_company="Acme",
        match_score=0.8,
        semantic_score=0.7,
        skill_score=0.6,
        changes_summary="summary overlay",
        grounding_report=GroundingReport(),
    )
    engine = MagicMock()
    engine.tailor_verified = AsyncMock(return_value=tailored)
    resume = ResumeData(raw_text="source")

    with patch(
        "job_applicator.workflows.tailor_auto.assert_tailored_auto_saveable"
    ) as assert_saveable:
        result = await tailor_auto_verified_saveable(
            tailor_engine=engine,
            resume=resume,
            job=MagicMock(),
            user_instructions="Use a concise summary",
        )

    assert result.tailored is tailored
    assert result.attempts == 1
    assert engine.tailor_verified.await_args.kwargs["user_instructions"] == (
        "Use a concise summary"
    )
    assert_saveable.assert_called_once_with(tailored, "source")
