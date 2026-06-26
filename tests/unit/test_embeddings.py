"""Unit tests for embedding service and matching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from instructor.core import InstructorError

from job_applicator.config import EmbeddingConfig, LLMConfig
from job_applicator.embeddings.service import EmbeddingService
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor, _ExtractionResult


class TestEmbeddingConfig:
    """Tests for embedding configuration."""

    def test_default_config(self) -> None:
        config = EmbeddingConfig()
        assert config.model_name == "mixedbread-ai/mxbai-embed-large-v1"
        assert config.device == "cuda"
        assert config.memory_limit_gb == 1.5
        assert config.normalize_embeddings is True

    def test_custom_config(self) -> None:
        config = EmbeddingConfig(
            model_name="test-model",
            device="cpu",
            memory_limit_gb=0.5,
        )
        assert config.model_name == "test-model"
        assert config.device == "cpu"
        assert config.memory_limit_gb == 0.5


class TestEmbeddingService:
    """Tests for embedding service."""

    @pytest.fixture
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(device="cpu", memory_limit_gb=0.5)

    @pytest.fixture
    def service(self, config: EmbeddingConfig) -> EmbeddingService:
        return EmbeddingService(config)

    def test_cache_key_generation(self, service: EmbeddingService) -> None:
        """Test cache key consistency."""
        text = "Test text for caching"
        key1 = service._get_cache_key(text)
        key2 = service._get_cache_key(text)
        assert key1 == key2
        assert len(key1) == 32  # Full MD5 hex digest

    def test_cache_key_includes_model_name(self) -> None:
        """Cache key must differ when model name changes."""
        config1 = EmbeddingConfig(device="cpu", memory_limit_gb=0.5, model_name="model-a")
        config2 = EmbeddingConfig(device="cpu", memory_limit_gb=0.5, model_name="model-b")
        svc1 = EmbeddingService(config1)
        svc2 = EmbeddingService(config2)
        assert svc1._get_cache_key("hello") != svc2._get_cache_key("hello")

    def test_cache_key_includes_normalize_flag(self) -> None:
        """Cache key must differ when normalize_embeddings changes."""
        config1 = EmbeddingConfig(device="cpu", memory_limit_gb=0.5, normalize_embeddings=True)
        config2 = EmbeddingConfig(device="cpu", memory_limit_gb=0.5, normalize_embeddings=False)
        svc1 = EmbeddingService(config1)
        svc2 = EmbeddingService(config2)
        assert svc1._get_cache_key("hello") != svc2._get_cache_key("hello")

    def test_similarity_fast_path_normalized(self) -> None:
        """When normalize_embeddings=True, similarity uses dot product."""
        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5, normalize_embeddings=True)
        svc = EmbeddingService(config)
        # Pre-normalized vectors: dot product = cosine similarity
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.707, 0.707, 0.0], dtype=np.float32)
        result = svc.similarity(vec1, vec2)
        assert result == pytest.approx(0.707, abs=0.01)

    def test_cache_key_different_text(self, service: EmbeddingService) -> None:
        """Test different texts get different keys."""
        key1 = service._get_cache_key("Text A")
        key2 = service._get_cache_key("Text B")
        assert key1 != key2

    def test_similarity_identical_vectors(self, service: EmbeddingService) -> None:
        """Test similarity of identical vectors is 1."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert service.similarity(vec, vec) == pytest.approx(1.0)

    def test_similarity_orthogonal_vectors(self, service: EmbeddingService) -> None:
        """Test similarity of orthogonal vectors is 0."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert service.similarity(vec1, vec2) == pytest.approx(0.0)

    def test_similarity_opposite_vectors(self, service: EmbeddingService) -> None:
        """Test similarity of opposite vectors is -1."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert service.similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_find_most_similar(self, service: EmbeddingService) -> None:
        """Test finding most similar vectors."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Orthogonal
            np.array([0.9, 0.1, 0.0], dtype=np.float32),  # Similar
            np.array([0.5, 0.5, 0.0], dtype=np.float32),  # Moderate
        ]

        results = service.find_most_similar(query, candidates, top_k=2)
        assert len(results) == 2
        assert results[0][0] == 1  # Index 1 is most similar
        assert results[0][1] > results[1][1]


class TestJobMatcher:
    """Tests for job matching."""

    @pytest.fixture
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(device="cpu", memory_limit_gb=0.5)

    def test_match_result_creation(self) -> None:
        """Test MatchResult dataclass."""
        from job_applicator.embeddings.matching import MatchResult
        from job_applicator.models import JobBoard, JobListing

        job = JobListing(
            title="Test Job",
            company="Test Co",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
        )

        result = MatchResult(
            job=job,
            score=0.85,
            semantic_score=0.6,
            skill_score=0.4,
            matched_skills=["Python"],
            missing_skills=["Java"],
            summary="Strong match",
        )

        assert result.score == 0.85
        assert result.matched_skills == ["Python"]
        assert result.summary == "Strong match"

    def test_skill_matching_structure(self) -> None:
        """Test skill matching returns correct structure."""
        from job_applicator.embeddings.matching import JobMatcher

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config)

        # Test with empty skills returns empty lists
        matched, missing = matcher._match_skills([], ["Python", "FastAPI"])
        assert matched == []
        assert missing == ["Python", "FastAPI"]

        # Test with empty requirements
        matched, missing = matcher._match_skills(["Python", "FastAPI"], [])
        assert matched == []
        assert missing == []

    def test_skill_match_threshold_rejects_false_positive_band(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the 0.75 skill-match threshold. A best-cosine in the false-positive band
        (0.59-0.73 — e.g. React~Python measured 0.62) is correctly MISSING; a genuine-match
        score (>=0.78) is covered. At the old 0.55 the false-positive band was wrongly
        'covered' (a Python résumé reported missing_skills=[] for a React job)."""
        from job_applicator.embeddings.matching import JobMatcher

        matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5))
        svc = matcher._service
        # Stub the model out — control the cosine directly (no model load).
        monkeypatch.setattr(svc, "embed_batch", lambda texts, **kw: [[1.0]] * len(texts))

        monkeypatch.setattr(svc, "similarity", lambda a, b: 0.62)  # false-positive band
        matched, missing = matcher._match_skills(["Python"], ["React"])
        assert matched == [] and missing == ["React"]  # NOT covered at 0.75 (was at 0.55)

        monkeypatch.setattr(svc, "similarity", lambda a, b: 0.80)  # genuine match
        matched, missing = matcher._match_skills(["Python"], ["React"])
        assert matched == ["Python"] and missing == []

    def test_embed_text_with_prefix(self) -> None:
        """embed_text should prepend the prefix when provided."""
        from job_applicator.embeddings.matching import JobMatcher

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config)

        # Verify the method exists and accepts a prefix parameter
        import inspect

        sig = inspect.signature(matcher.embed_text)
        assert "prefix" in sig.parameters

    def test_compute_resume_embedding_uses_prefix(self) -> None:
        """Resume embedding should use the search prefix for asymmetric retrieval."""
        from job_applicator.embeddings.matching import JobMatcher
        from job_applicator.models import ResumeData

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config)

        resume = ResumeData(
            raw_text="John Doe\nSkills: Python",
            name="John Doe",
            skills=["Python"],
        )

        # Patch embed to capture the text passed to it
        with patch.object(
            matcher._service, "embed", return_value=np.zeros(1024, dtype=np.float32)
        ) as mock_embed:
            matcher.compute_resume_embedding(resume)
            call_text = mock_embed.call_args[0][0]
            assert "Represent this sentence for searching relevant passages" in call_text

    def test_is_pii_or_noise_filters_generically(self) -> None:
        """PII filtering must be generic — no hardcoded names/emails."""
        from job_applicator.embeddings.matching import JobMatcher

        # Bullet glyphs are noise.
        assert JobMatcher._is_pii_or_noise("•", "") is True
        # The candidate's own name (any name) is filtered.
        assert JobMatcher._is_pii_or_noise("JANE SMITH", "jane smith") is True
        # A bare email/contact line is filtered.
        assert JobMatcher._is_pii_or_noise("jane.smith@example.com", "jane smith") is True
        # Real content is kept.
        assert JobMatcher._is_pii_or_noise("Built data pipelines in Python", "jane smith") is False

    def test_compute_resume_embedding_drops_name_and_email(self) -> None:
        """The raw-text fallback must not embed the candidate's name/email."""
        from job_applicator.embeddings.matching import JobMatcher
        from job_applicator.models import ResumeData

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config)

        # Sparse structured data forces the raw-text fallback path.
        resume = ResumeData(
            raw_text=(
                "JANE SMITH\njane.smith@example.com\n"
                "Skills\nPython\nKubernetes\n"
                "Experience\nSenior Engineer at Acme"
            ),
            name="JANE SMITH",
            skills=["•"],
        )

        with patch.object(
            matcher._service, "embed", return_value=np.zeros(1024, dtype=np.float32)
        ) as mock_embed:
            matcher.compute_resume_embedding(resume)
            call_text = mock_embed.call_args[0][0]

        assert "JANE SMITH" not in call_text
        assert "jane.smith@example.com" not in call_text
        assert "Python" in call_text  # real skills survive

    def test_match_skills_shares_best_available_skill(self) -> None:
        """Two requirements that both prefer one skill must not falsely mark one missing."""
        from job_applicator.embeddings.matching import JobMatcher

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config)

        # Pass strings straight through as their own "embeddings".
        matcher._service.embed_batch = lambda texts: list(texts)  # type: ignore[method-assign]

        # Synthetic scores (not real cosines) — all "matched" values are above the 0.75
        # threshold so this test exercises the used-skills CLAIMING logic, not the threshold.
        sim_table = {
            ("Python", "Python"): 0.90,
            ("Python", "Java"): 0.20,
            ("Python development", "Python"): 0.85,
            ("Python development", "Java"): 0.80,
        }
        matcher._service.similarity = lambda a, b: sim_table[(a, b)]  # type: ignore[method-assign]

        matched, missing = matcher._match_skills(
            ["Python", "Java"],
            ["Python programming", "Python development"],
        )

        # Both requirements are satisfied: the second falls back to its best
        # *available* skill (Java) instead of being marked missing.
        assert missing == []
        assert set(matched) == {"Python", "Java"}


class TestDescriptionSkillExtraction:
    """Tests for fallback skill extraction from job descriptions."""

    def test_extract_requirements_from_description_finds_known_skills(self) -> None:
        from job_applicator.embeddings.matching import JobMatcher

        matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5))
        description = "We need Python, Kubernetes, and PostgreSQL experience."
        reqs = matcher._extract_requirements_from_description(description)
        assert "Python" in reqs
        assert "Kubernetes" in reqs
        assert "PostgreSQL" in reqs

    def test_extract_requirements_ignores_hard_negatives_and_unknowns(self) -> None:
        from job_applicator.embeddings.matching import JobMatcher

        matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5))
        description = "Looking for a team player with communication skills and Python."
        reqs = matcher._extract_requirements_from_description(description)
        assert "Python" in reqs
        assert "team player" not in reqs
        assert "communication" not in reqs

    def test_extract_requirements_empty_for_empty_description(self) -> None:
        from job_applicator.embeddings.matching import JobMatcher

        matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5))
        assert matcher._extract_requirements_from_description("") == []
        assert matcher._extract_requirements_from_description("   ") == []


class TestLLMSkillExtractor:
    """Tests for LLM-driven skill extraction."""

    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        return LLMConfig(
            model="test-model",
            api_base="http://localhost:8000/v1",
            api_key="not-needed",
        )

    @pytest.fixture
    def extractor(self, llm_config: LLMConfig) -> LLMSkillExtractor:
        return LLMSkillExtractor(llm_config)

    def test_cache_key_includes_model_and_description(self) -> None:
        """Cache key must differ when model or description changes."""
        config1 = LLMConfig(model="model-a", api_base="http://localhost:8000/v1")
        config2 = LLMConfig(model="model-b", api_base="http://localhost:8000/v1")
        extractor1 = LLMSkillExtractor(config1)
        extractor2 = LLMSkillExtractor(config2)

        # Different models -> different keys for the same description.
        assert extractor1._get_cache_key("same description") != extractor2._get_cache_key(
            "same description"
        )

        # Different descriptions -> different keys for the same model.
        key1 = extractor1._get_cache_key("description one")
        key2 = extractor1._get_cache_key("description two")
        assert key1 != key2
        assert len(key1) == 16

    def test_empty_description_returns_empty_list(self, extractor: LLMSkillExtractor) -> None:
        """Empty or whitespace descriptions return an empty list without LLM call."""
        import asyncio

        with patch.object(extractor, "_call_llm") as mock_call:
            assert asyncio.run(extractor.extract("")) == []
            assert asyncio.run(extractor.extract("   ")) == []
            assert asyncio.run(extractor.extract("\n\t")) == []
            mock_call.assert_not_called()

    def test_cache_hit_returns_cached_skills(self, extractor: LLMSkillExtractor) -> None:
        """Cache hit returns cached skills without calling the LLM."""
        import asyncio

        description = "We need Python, Kubernetes."
        cache_path = extractor._get_cache_path(description)
        cache_path.write_text('{"skills": ["Python", "Kubernetes"]}', encoding="utf-8")

        try:
            with patch.object(extractor, "_call_llm") as mock_call:
                result = asyncio.run(extractor.extract(description))
                assert result == ["Kubernetes", "Python"]
                mock_call.assert_not_called()
        finally:
            cache_path.unlink(missing_ok=True)

    def test_cache_miss_writes_cleaned_skills(
        self,
        extractor: LLMSkillExtractor,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cache miss calls LLM, filters hard negatives, and writes cleaned skills."""
        import asyncio
        import json

        description = "We need Python."
        monkeypatch.setattr(extractor, "_cache_dir", tmp_path)

        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(
                skills=["Python", "team player"], method="instructor", fallback=False
            ),
        ) as mock_call:
            result = asyncio.run(extractor.extract(description, use_cache=True))
            assert "Python" in result
            assert "team player" not in result
            mock_call.assert_called_once()

        cache_path = extractor._get_cache_path(description)
        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data["skills"] == ["Python"]

    def test_llm_failure_returns_empty_list(self, extractor: LLMSkillExtractor) -> None:
        """LLM failure returns [] and does not crash."""
        import asyncio

        with patch.object(extractor, "_call_llm", side_effect=RuntimeError("boom")):
            result = asyncio.run(extractor.extract("We need Python.", use_cache=False))
            assert result == []

    def test_hallucinated_skills_are_dropped(self, extractor: LLMSkillExtractor) -> None:
        """Skills not grounded in the description are dropped."""
        import asyncio

        description = "We need Python."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(
                skills=["Python", "Rust", "Kubernetes"], method="instructor", fallback=False
            ),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert result == ["Python"]

    def test_aliases_kept_when_grounded(self, extractor: LLMSkillExtractor) -> None:
        """Skills matching known aliases but not canonical form are kept."""
        import asyncio

        description = "We use postgres, vuejs."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(
                skills=["postgres", "vuejs"], method="instructor", fallback=False
            ),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "PostgreSQL" in result
            assert "Vue.js" in result

    def test_react_native_compound_rejection(self, extractor: LLMSkillExtractor) -> None:
        """React inside React Native is rejected; React Native explicit is kept."""
        import asyncio

        description_lower = "we are hiring a react native engineer."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(skills=["React"], method="instructor", fallback=False),
        ):
            result = asyncio.run(extractor.extract(description_lower, use_cache=False))
            assert "React" not in result

        description_explicit = "Experience with React Native is required."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(
                skills=["React Native"], method="instructor", fallback=False
            ),
        ):
            result = asyncio.run(extractor.extract(description_explicit, use_cache=False))
            assert "React Native" in result

    def test_single_word_skill_followed_by_common_word_is_grounded(
        self, extractor: LLMSkillExtractor
    ) -> None:
        """A single-word skill followed by a function word stays grounded."""
        import asyncio

        description = "React is great. We also use React Native."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(skills=["React"], method="instructor", fallback=False),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "React" in result

    def test_react_native_trailing_punctuation_rejection(
        self, extractor: LLMSkillExtractor
    ) -> None:
        """React followed by 'Native.' (with punctuation) is rejected as a compound."""
        import asyncio

        description = "Experience with React Native."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(skills=["React"], method="instructor", fallback=False),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "React" not in result

    def test_version_number_keeps_base_skill(self, extractor: LLMSkillExtractor) -> None:
        """Version numbers next to a skill should not suppress the base skill."""
        import asyncio

        description = "Experience with Java 8."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(skills=["Java"], method="instructor", fallback=False),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "Java" in result

    def test_wildcard_version_number_keeps_base_skill(self, extractor: LLMSkillExtractor) -> None:
        """Wildcard version numbers next to a skill should not suppress the base skill."""
        import asyncio

        description = "Experience with Java 3.x."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(skills=["Java"], method="instructor", fallback=False),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "Java" in result

    def test_react_native_still_rejects_base_skill(self, extractor: LLMSkillExtractor) -> None:
        """Non-version compounds still reject the bare base skill."""
        import asyncio

        description = "Experience with React Native."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(skills=["React"], method="instructor", fallback=False),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "React" not in result

    def test_direct_fallback_handles_empty_content(self, extractor: LLMSkillExtractor) -> None:
        """Direct litellm fallback returns [] when choices are empty or content is None."""
        import asyncio

        description = "We need Python."
        mock_instructor = MagicMock()
        mock_instructor.from_litellm.return_value.create = AsyncMock(
            side_effect=InstructorError("instructor failed")
        )

        # Empty choices
        fake_response_empty = AsyncMock()
        fake_response_empty.choices = []

        with patch("job_applicator.embeddings.skill_extraction.instructor", mock_instructor):
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response_empty,
            ):
                result = asyncio.run(extractor.extract(description, use_cache=False))
                assert result == []

        # None content
        fake_response_none = AsyncMock()
        fake_response_none.choices = [AsyncMock()]
        fake_response_none.choices[0].message.content = None

        with patch("job_applicator.embeddings.skill_extraction.instructor", mock_instructor):
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response_none,
            ):
                result = asyncio.run(extractor.extract(description, use_cache=False))
                assert result == []

    def test_direct_fallback_parses_json_array(self, extractor: LLMSkillExtractor) -> None:
        """Direct litellm fallback parses a raw JSON array response."""
        import asyncio

        description = "We need Python and FastAPI."
        fake_response = AsyncMock()
        fake_response.choices = [AsyncMock()]
        fake_response.choices[0].message.content = '["Python", "FastAPI"]'

        mock_instructor = MagicMock()
        mock_instructor.from_litellm.return_value.create = AsyncMock(
            side_effect=InstructorError("instructor failed")
        )

        with patch("job_applicator.embeddings.skill_extraction.instructor", mock_instructor):
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response,
            ):
                result = asyncio.run(extractor.extract(description, use_cache=False))
                assert "Python" in result
                assert "FastAPI" in result

    def test_direct_fallback_parses_markdown_json(self, extractor: LLMSkillExtractor) -> None:
        """Direct litellm fallback parses JSON embedded in a markdown block."""
        import asyncio

        description = "We need Python."
        fake_response = AsyncMock()
        fake_response.choices = [AsyncMock()]
        fake_response.choices[0].message.content = '```json\n{"skills": ["Python"]}\n```'

        mock_instructor = MagicMock()
        mock_instructor.from_litellm.return_value.create = AsyncMock(
            side_effect=InstructorError("instructor failed")
        )

        with patch("job_applicator.embeddings.skill_extraction.instructor", mock_instructor):
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response,
            ):
                result = asyncio.run(extractor.extract(description, use_cache=False))
                assert "Python" in result

    def test_instructor_fallback_exercised(self, extractor: LLMSkillExtractor) -> None:
        """If instructor fails, direct litellm is used."""
        import asyncio

        description = "We need Python."

        fake_response = AsyncMock()
        fake_response.choices = [AsyncMock()]
        fake_response.choices[0].message.content = '{"skills": ["Python"]}'

        mock_instructor = MagicMock()
        mock_instructor.from_litellm.return_value.create = AsyncMock(
            side_effect=InstructorError("instructor failed")
        )

        with patch("job_applicator.embeddings.skill_extraction.instructor", mock_instructor):
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response,
            ) as mock_acompletion:
                result = asyncio.run(extractor.extract(description, use_cache=False))
                assert result == ["Python"]
                mock_instructor.from_litellm.assert_called_once()
                mock_acompletion.assert_awaited_once()

    def test_corrupt_cache_treated_as_miss(self, extractor: LLMSkillExtractor) -> None:
        """Corrupt cache entries are treated as misses."""
        import asyncio

        description = "We need Python."
        cache_path = extractor._get_cache_path(description)
        cache_path.write_text("not json", encoding="utf-8")

        try:
            with patch.object(
                extractor,
                "_call_llm",
                return_value=_ExtractionResult(
                    skills=["Python"], method="instructor", fallback=False
                ),
            ) as mock_call:
                result = asyncio.run(extractor.extract(description))
                assert result == ["Python"]
                mock_call.assert_called_once()
        finally:
            cache_path.unlink(missing_ok=True)

    def test_cache_key_changes_when_model_changes(self, extractor: LLMSkillExtractor) -> None:
        """Cache key must differ when llm.model changes."""
        config_a = LLMConfig(model="model-a", api_base="http://localhost:8000/v1")
        config_b = LLMConfig(model="model-b", api_base="http://localhost:8000/v1")
        extractor_a = LLMSkillExtractor(config_a)
        extractor_b = LLMSkillExtractor(config_b)
        assert extractor_a._get_cache_key("same") != extractor_b._get_cache_key("same")

    def test_duplicate_and_empty_skills_cleaned(self, extractor: LLMSkillExtractor) -> None:
        """Duplicate and empty skill strings are cleaned."""
        import asyncio

        description = "We need Python, AWS."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(
                skills=["Python", "Python", "", "  ", "AWS"],
                method="instructor",
                fallback=False,
            ),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert result == ["AWS", "Python"]

    def test_user_message_is_exact_truncated_description(
        self, extractor: LLMSkillExtractor
    ) -> None:
        """The user message contains exactly the first 1500 characters."""
        import asyncio

        description = "x" * 2000
        expected = description[:1500]

        fake_response = AsyncMock()
        fake_response.choices = [AsyncMock()]
        fake_response.choices[0].message.content = '{"skills": []}'

        with patch("job_applicator.embeddings.skill_extraction.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value.create.side_effect = InstructorError(
                "instructor failed"
            )
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response,
            ) as mock_acompletion:
                asyncio.run(extractor.extract(description, use_cache=False))
                call_kwargs = mock_acompletion.await_args.kwargs
                messages = call_kwargs["messages"]
                assert messages[1]["role"] == "user"
                assert messages[1]["content"] == expected

    def test_call_args_use_quiet_litellm_enable_thinking_and_model_prefix(
        self, extractor: LLMSkillExtractor
    ) -> None:
        """quiet_litellm, enable_thinking=False, and openai/ model prefix are used."""
        import asyncio

        description = "We need Python."
        fake_response = AsyncMock()
        fake_response.choices = [AsyncMock()]
        fake_response.choices[0].message.content = '{"skills": ["Python"]}'

        with patch("job_applicator.embeddings.skill_extraction.quiet_litellm") as mock_quiet:
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response,
            ) as mock_acompletion:
                with patch(
                    "job_applicator.embeddings.skill_extraction.instructor"
                ) as mock_instructor:
                    mock_instructor.from_litellm.return_value.create.side_effect = InstructorError(
                        "instructor failed"
                    )
                    result = asyncio.run(extractor.extract(description, use_cache=False))
                    assert result == ["Python"]
                    mock_quiet.assert_called_once()
                    mock_acompletion.assert_awaited_once()
                    call_kwargs = mock_acompletion.await_args.kwargs
                    assert call_kwargs["model"] == "openai/test-model"
                    assert call_kwargs.get("extra_body") == {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                    messages = call_kwargs["messages"]
                    assert messages[0]["role"] == "system"
                    assert messages[1]["role"] == "user"
                    assert messages[1]["content"] == description[:1500]

    def test_reporter_records_cache_hit(self, extractor: LLMSkillExtractor) -> None:
        """Reporter records cache hit event."""
        import asyncio

        from job_applicator.utils.verbose import VerboseReporter

        description = "We need Python."
        cache_path = extractor._get_cache_path(description)
        cache_path.write_text('{"skills": ["Python"]}', encoding="utf-8")
        reporter = VerboseReporter(command="test", args={}, config={})

        try:
            with patch.object(extractor, "_call_llm") as mock_call:
                asyncio.run(extractor.extract(description, reporter=reporter))
                mock_call.assert_not_called()
                details = [call["details"] for call in reporter.report.llm.calls]
                assert {"skill_extraction": "cache_hit"} in details
        finally:
            cache_path.unlink(missing_ok=True)

    def test_reporter_records_cache_miss_and_instructor_call(
        self, extractor: LLMSkillExtractor
    ) -> None:
        """Reporter records cache miss and instructor llm_call event."""
        import asyncio

        from job_applicator.utils.verbose import VerboseReporter

        description = "We need Python."
        reporter = VerboseReporter(command="test", args={}, config={})

        fake_response = AsyncMock()
        fake_response.skills = ["Python"]

        with patch("job_applicator.embeddings.skill_extraction.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value.create = AsyncMock(return_value=fake_response)
            asyncio.run(extractor.extract(description, use_cache=False, reporter=reporter))

        details = [call["details"] for call in reporter.report.llm.calls]
        assert {"skill_extraction": "cache_miss"} in details
        assert {"skill_extraction": "llm_call", "method": "instructor"} in details

    def test_reporter_records_fallback_and_direct_call(self, extractor: LLMSkillExtractor) -> None:
        """Reporter records fallback and direct llm_call event."""
        import asyncio

        from job_applicator.utils.verbose import VerboseReporter

        description = "We need Python."
        reporter = VerboseReporter(command="test", args={}, config={})

        fake_response = AsyncMock()
        fake_response.choices = [AsyncMock()]
        fake_response.choices[0].message.content = '{"skills": ["Python"]}'

        with patch("job_applicator.embeddings.skill_extraction.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value.create.side_effect = InstructorError(
                "instructor failed"
            )
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response,
            ):
                asyncio.run(extractor.extract(description, use_cache=False, reporter=reporter))

        details = [call["details"] for call in reporter.report.llm.calls]
        assert {"skill_extraction": "cache_miss"} in details
        assert {
            "skill_extraction": "fallback",
            "from": "instructor",
            "to": "direct",
        } in details
        assert {
            "skill_extraction": "llm_call",
            "method": "direct",
            "fallback": True,
        } in details

    def test_reporter_records_error(self, extractor: LLMSkillExtractor) -> None:
        """Reporter records error event on LLM failure."""
        import asyncio

        from job_applicator.utils.verbose import VerboseReporter

        description = "We need Python."
        reporter = VerboseReporter(command="test", args={}, config={})

        with patch.object(extractor, "_call_llm", side_effect=RuntimeError("boom")):
            asyncio.run(extractor.extract(description, use_cache=False, reporter=reporter))

        details = [call["details"] for call in reporter.report.llm.calls]
        assert any(d.get("skill_extraction") == "error" for d in details)
        assert reporter.report.errors
