"""Integration tests for LLMSkillExtractor."""

from __future__ import annotations

import pytest

from job_applicator.config import LLMConfig
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor, _ExtractionResult


class TestSkillExtractionIntegration:
    async def test_extracts_python_from_description_with_mocked_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = LLMSkillExtractor(LLMConfig(model="test"))

        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Python", "FastAPI", "PostgreSQL"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract(
            "We are looking for a backend engineer with Python, FastAPI, and PostgreSQL."
        )
        assert set(result) == {"FastAPI", "PostgreSQL", "Python"}

    async def test_unmapped_skill_grounded_by_token_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = LLMSkillExtractor(LLMConfig(model="test"))

        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Salesforce"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("Experience with Salesforce is required.")
        assert "Salesforce" in result
