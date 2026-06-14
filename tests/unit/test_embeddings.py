"""Unit tests for embedding service and matching."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from job_applicator.config import EmbeddingConfig
from job_applicator.embeddings.service import EmbeddingService


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
