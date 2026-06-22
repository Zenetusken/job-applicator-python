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
