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

    async def test_extract_raises_on_llm_failure_not_empty(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An LLM-call FAILURE must raise LLMError — never return [] (indistinguishable from a job
        that genuinely lists no skills, which would silently degrade the match downstream)."""
        from job_applicator.exceptions import LLMError

        async def boom(description: str) -> _ExtractionResult:
            raise ConnectionError("connection refused")

        monkeypatch.setattr(extractor, "_call_llm", boom)
        with pytest.raises(LLMError):
            await extractor.extract("Senior Python engineer with Django and PostgreSQL.")

    async def test_extract_returns_empty_on_successful_no_skills(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SUCCESSFUL call that finds no skills legitimately returns [] — not every empty is a
        failure. This is the distinction the no-masking rule hinges on (failure→raise, empty→ok)."""

        async def none_found(description: str) -> _ExtractionResult:
            return _ExtractionResult(skills=[], method="instructor", fallback=False)

        monkeypatch.setattr(extractor, "_call_llm", none_found)
        result = await extractor.extract("We value teamwork and a positive attitude.")
        assert result == []

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

    async def test_multiword_skill_grounded_by_exact_phrase(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Machine Learning"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We apply machine learning to our products.")
        assert "Machine Learning" in result

    async def test_multiword_skill_not_grounded_as_substring(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["REST APIs"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We expose REST APIsolutions only.")
        assert "REST APIs" not in result

    async def test_single_word_skill_accepted_when_no_compound(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["React"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We use React.")
        assert "React" in result

    async def test_single_word_skill_rejected_when_lowercase_compound_follows(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["React"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("we need a react native engineer.")
        assert "React" not in result

    async def test_single_word_skill_accepted_when_prose_word_follows(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["React"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We need React experience for this role.")
        assert "React" in result

    async def test_version_like_suffix_does_not_reject_single_word(
        self, extractor: LLMSkillExtractor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(description: str) -> _ExtractionResult:
            return _ExtractionResult(
                skills=["Python"],
                method="instructor",
                fallback=False,
            )

        monkeypatch.setattr(extractor, "_call_llm", fake_llm)

        result = await extractor.extract("We use Python 3.11 on the backend.")
        assert "Python" in result
