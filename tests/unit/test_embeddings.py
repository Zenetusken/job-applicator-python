"""Unit tests for embedding service and matching."""

from __future__ import annotations

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
        assert len(key1) == 16

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
