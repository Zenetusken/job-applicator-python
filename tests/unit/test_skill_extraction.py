"""Unit tests for LLMSkillExtractor."""

from __future__ import annotations

import pytest

from job_applicator.config import LLMConfig
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor, _ExtractionResult


@pytest.fixture
def extractor(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> LLMSkillExtractor:
    """Create an LLMSkillExtractor that writes its cache under a temp directory."""
    inst = LLMSkillExtractor(LLMConfig(model="test"))
    monkeypatch.setattr(inst, "_cache_dir", tmp_path / "skill-extraction")
    inst._cache_dir.mkdir(parents=True, exist_ok=True)
    return inst


class TestSkillExtraction:
    async def test_extracts_python_from_description_with_mocked_llm(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Salesforce"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("Experience with Salesforce is required.")
        assert "Salesforce" in result
