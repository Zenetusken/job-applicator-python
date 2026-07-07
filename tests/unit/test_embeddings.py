"""Unit tests for embedding service and matching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from instructor.core import InstructorError

from job_applicator.config import EmbeddingConfig, LLMConfig
from job_applicator.embeddings.matching import JobMatcher
from job_applicator.embeddings.service import EmbeddingService
from job_applicator.embeddings.skill_extraction import LLMSkillExtractor, _ExtractionResult
from job_applicator.exceptions import ConfigError
from job_applicator.models import ResumeData


class TestEmbeddingConfig:
    """Tests for embedding configuration."""

    def test_default_config(self) -> None:
        config = EmbeddingConfig()
        assert config.model_name == "mixedbread-ai/mxbai-embed-large-v1"
        assert config.device == "cuda"
        assert config.memory_limit_gb == 1.3
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

    def test_resolve_device_requires_cuda_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `cuda` request on a box without CUDA fails loudly instead of silently using CPU."""
        import sys
        from types import SimpleNamespace

        monkeypatch.setitem(
            sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        )
        svc = EmbeddingService(EmbeddingConfig(device="cuda", memory_limit_gb=0.5))
        with pytest.raises(ConfigError, match="CUDA is not available"):
            svc._resolve_device()

    def test_resolve_device_honors_cpu_and_available_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit `cpu` is honored without importing torch; available CUDA is kept."""
        import sys
        from types import SimpleNamespace

        cpu_svc = EmbeddingService(EmbeddingConfig(device="cpu", memory_limit_gb=0.5))
        assert cpu_svc._resolve_device() == "cpu"
        monkeypatch.setitem(
            sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        )
        cuda_svc = EmbeddingService(EmbeddingConfig(device="cuda", memory_limit_gb=0.5))
        assert cuda_svc._resolve_device() == "cuda"

    def test_configure_cpu_threads_caps_oversubscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        from types import SimpleNamespace

        state = {"threads": 16, "interop": 16}

        fake_torch = SimpleNamespace(
            get_num_threads=lambda: state["threads"],
            set_num_threads=lambda value: state.__setitem__("threads", value),
            get_num_interop_threads=lambda: state["interop"],
            set_num_interop_threads=lambda value: state.__setitem__("interop", value),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        EmbeddingService(
            EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        )._configure_cpu_threads()

        assert state == {"threads": 4, "interop": 1}

    def test_load_model_uses_local_only_when_snapshot_is_cached(
        self, monkeypatch: pytest.MonkeyPatch, config: EmbeddingConfig
    ) -> None:
        calls: list[dict[str, object]] = []

        class FakeSentenceTransformer:
            max_seq_length = 0

            def __init__(self, *args: object, **kwargs: object) -> None:
                calls.append({"args": args, **kwargs})

        monkeypatch.setattr(
            "job_applicator.embeddings.service.probe_hf_model_cache",
            lambda _model_name: (True, "/cache/model"),
        )
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            FakeSentenceTransformer,
        )

        model = EmbeddingService(config)._load_model()

        assert isinstance(model, FakeSentenceTransformer)
        assert calls[0]["device"] == "cpu"
        assert calls[0]["local_files_only"] is True
        assert calls[0]["model_kwargs"] == {"torch_dtype": "float32"}
        assert model.max_seq_length == config.max_seq_length

    def test_load_model_uses_fp16_only_on_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        from types import SimpleNamespace

        calls: list[dict[str, object]] = []

        class FakeSentenceTransformer:
            max_seq_length = 0

            def __init__(self, *args: object, **kwargs: object) -> None:
                calls.append({"args": args, **kwargs})

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    mem_get_info=lambda _idx: (2 * 1024**3, 12 * 1024**3),
                ),
                device=lambda _device: SimpleNamespace(index=0),
            ),
        )
        monkeypatch.setattr(
            "job_applicator.embeddings.service.probe_hf_model_cache",
            lambda _model_name: (True, "/cache/model"),
        )
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            FakeSentenceTransformer,
        )

        EmbeddingService(EmbeddingConfig(device="cuda", memory_limit_gb=0.5))._load_model()

        assert calls[0]["device"] == "cuda"
        assert calls[0]["model_kwargs"] == {"torch_dtype": "float16"}

    def test_load_model_fails_when_cuda_headroom_below_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        from types import SimpleNamespace

        class FakeSentenceTransformer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("model load should be skipped by VRAM preflight")

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    mem_get_info=lambda _idx: (512 * 1024**2, 12 * 1024**3),
                ),
                device=lambda _device: SimpleNamespace(index=0),
            ),
        )
        monkeypatch.setattr(
            "job_applicator.embeddings.service.probe_hf_model_cache",
            lambda _model_name: (True, "/cache/model"),
        )
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            FakeSentenceTransformer,
        )

        with pytest.raises(ConfigError, match="free VRAM"):
            EmbeddingService(EmbeddingConfig(device="cuda", memory_limit_gb=1.5))._load_model()

    def test_load_model_allows_download_when_snapshot_is_uncached(
        self, monkeypatch: pytest.MonkeyPatch, config: EmbeddingConfig
    ) -> None:
        calls: list[dict[str, object]] = []

        class FakeSentenceTransformer:
            max_seq_length = 0

            def __init__(self, *args: object, **kwargs: object) -> None:
                calls.append({"args": args, **kwargs})

        monkeypatch.setattr(
            "job_applicator.embeddings.service.probe_hf_model_cache",
            lambda _model_name: (False, None),
        )
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            FakeSentenceTransformer,
        )

        EmbeddingService(config)._load_model()

        assert calls[0]["local_files_only"] is False

    def test_load_model_wraps_runtime_failures(
        self, monkeypatch: pytest.MonkeyPatch, config: EmbeddingConfig
    ) -> None:
        class BrokenSentenceTransformer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise OSError("network unavailable")

        monkeypatch.setattr(
            "job_applicator.embeddings.service.probe_hf_model_cache",
            lambda _model_name: (False, None),
        )
        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            BrokenSentenceTransformer,
        )

        with pytest.raises(ConfigError, match="Embedding model could not be loaded"):
            EmbeddingService(config)._load_model()

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

    def test_cache_key_includes_max_seq_length(self) -> None:
        """Cache key must differ when max_seq_length changes — it changes truncation and thus the
        vector, so a different length must not return a stale cached embedding."""
        config1 = EmbeddingConfig(device="cpu", memory_limit_gb=0.5, max_seq_length=128)
        config2 = EmbeddingConfig(device="cpu", memory_limit_gb=0.5, max_seq_length=512)
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

    def test_embed_tolerates_cache_write_failure(
        self,
        service: EmbeddingService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        vector = np.array([1.0, 0.0], dtype=np.float32)

        class FakeModel:
            def encode(self, _text: str, **_kwargs: object) -> np.ndarray:
                return vector

        monkeypatch.setattr(service, "_load_model", lambda: FakeModel())

        def fail_save(*_args: object, **_kwargs: object) -> None:
            raise OSError("read-only cache")

        monkeypatch.setattr("job_applicator.embeddings.service.np.save", fail_save)

        result = service.embed("uncached text")

        np.testing.assert_array_equal(result, vector)

    def test_embed_batch_tolerates_cache_write_failure(
        self,
        service: EmbeddingService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        vectors = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        class FakeModel:
            def encode(self, _texts: list[str], **_kwargs: object) -> list[np.ndarray]:
                return vectors

        monkeypatch.setattr(service, "_load_model", lambda: FakeModel())

        def fail_save(*_args: object, **_kwargs: object) -> None:
            raise OSError("read-only cache")

        monkeypatch.setattr("job_applicator.embeddings.service.np.save", fail_save)

        results = service.embed_batch(["first", "second"])

        assert len(results) == 2
        np.testing.assert_array_equal(results[0], vectors[0])
        np.testing.assert_array_equal(results[1], vectors[1])

    def test_embed_batch_deduplicates_uncached_texts(
        self,
        service: EmbeddingService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[list[str]] = []

        class FakeModel:
            def encode(self, texts: list[str], **_kwargs: object) -> list[np.ndarray]:
                calls.append(list(texts))
                return [
                    np.array([float(index), 0.0], dtype=np.float32)
                    for index, _text in enumerate(texts, start=1)
                ]

        monkeypatch.setattr(service, "_load_model", lambda: FakeModel())

        results = service.embed_batch(["same", "other", "same"], use_cache=False)

        assert calls == [["same", "other"]]
        assert len(results) == 3
        np.testing.assert_array_equal(results[0], results[2])

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

    async def test_skill_matching_structure(self) -> None:
        """Test skill matching returns correct structure."""
        from job_applicator.embeddings.matching import JobMatcher

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config, LLMConfig())

        # Test with empty skills returns empty lists
        matched, missing = await matcher._match_skills([], ["Python", "FastAPI"])
        assert matched == []
        assert missing == ["Python", "FastAPI"]

        # Test with empty requirements
        matched, missing = await matcher._match_skills(["Python", "FastAPI"], [])
        assert matched == []
        assert missing == []

    async def test_skill_match_threshold_rejects_false_positive_band(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the 0.75 skill-match threshold. A best-cosine in the false-positive band
        (0.59-0.73 — e.g. React~Python measured 0.62) is correctly MISSING; a genuine-match
        score (>=0.78) is covered. At the old 0.55 the false-positive band was wrongly
        'covered' (a Python résumé reported missing_skills=[] for a React job)."""
        from job_applicator.embeddings.matching import JobMatcher

        matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5), LLMConfig())
        svc = matcher._service
        # Stub the model out — control the cosine directly (no model load).
        monkeypatch.setattr(svc, "embed_batch", lambda texts, **kw: [[1.0]] * len(texts))

        monkeypatch.setattr(svc, "similarity", lambda a, b: 0.62)  # false-positive band
        matched, missing = await matcher._match_skills(["Python"], ["React"])
        assert matched == [] and missing == ["React"]  # NOT covered at 0.75 (was at 0.55)

        monkeypatch.setattr(svc, "similarity", lambda a, b: 0.80)  # genuine match
        matched, missing = await matcher._match_skills(["Python"], ["React"])
        assert matched == ["Python"] and missing == []

    def test_embed_text_with_prefix(self) -> None:
        """embed_text should prepend the prefix when provided."""
        from job_applicator.embeddings.matching import JobMatcher

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config, LLMConfig())

        # Verify the method exists and accepts a prefix parameter
        import inspect

        sig = inspect.signature(matcher.embed_text)
        assert "prefix" in sig.parameters

    def test_compute_resume_embedding_uses_prefix(self) -> None:
        """Resume embedding should use the search prefix for asymmetric retrieval."""
        from job_applicator.embeddings.matching import JobMatcher
        from job_applicator.models import ResumeData

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config, LLMConfig())

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
        matcher = JobMatcher(config, LLMConfig())

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

    async def test_match_skills_shares_best_available_skill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two requirements that both prefer one skill must not falsely mark one missing."""
        from job_applicator.embeddings.matching import JobMatcher

        config = EmbeddingConfig(device="cpu", memory_limit_gb=0.5)
        matcher = JobMatcher(config, LLMConfig())

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

        matched, missing = await matcher._match_skills(
            ["Python", "Java"],
            ["Python programming", "Python development"],
        )

        # Both requirements are satisfied: the second falls back to its best
        # *available* skill (Java) instead of being marked missing.
        assert missing == []
        assert set(matched) == {"Python", "Java"}


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
    def extractor(
        self, llm_config: LLMConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> LLMSkillExtractor:
        # Keyword (legacy) path: these tests mock the plain `_call_llm` (string skills). The
        # default is now evidence_span, so pin keyword explicitly.
        extractor = LLMSkillExtractor(llm_config, grounding_mode="keyword")
        monkeypatch.setattr(extractor, "_cache_dir", tmp_path / "skill-extraction")
        extractor._cache_dir.mkdir(parents=True, exist_ok=True)
        return extractor

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

    def test_llm_failure_raises_not_empty_list(self, extractor: LLMSkillExtractor) -> None:
        """An LLM failure must RAISE LLMError — never return [] (a masked failure indistinguishable
        from a job genuinely listing no skills; the no-fabricated-fallback rule)."""
        import asyncio

        from job_applicator.exceptions import LLMError

        with patch.object(extractor, "_call_llm", side_effect=RuntimeError("boom")):
            with pytest.raises(LLMError):
                asyncio.run(extractor.extract("We need Python.", use_cache=False))

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

        description_lower = "we are hiring a React Native engineer."
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

    def test_common_word_after_skill_does_not_reject_it(self, extractor: LLMSkillExtractor) -> None:
        """A lowercase common word after a skill must not create a pseudo-compound."""
        import asyncio

        description = "Experience with Kubernetes is required."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(
                skills=["Kubernetes"], method="instructor", fallback=False
            ),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "Kubernetes" in result

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

    def test_v_prefixed_version_number_keeps_python(self, extractor: LLMSkillExtractor) -> None:
        """A 'v' prefix on a version number should not suppress the base skill."""
        import asyncio

        description = "Experience with Python v3.11."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(skills=["Python"], method="instructor", fallback=False),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "Python" in result

    def test_v_prefixed_version_number_keeps_pydantic(self, extractor: LLMSkillExtractor) -> None:
        """A 'v' prefix on a version number should not suppress the base skill."""
        import asyncio

        description = "Experience with Pydantic v2."
        with patch.object(
            extractor,
            "_call_llm",
            return_value=_ExtractionResult(
                skills=["Pydantic"], method="instructor", fallback=False
            ),
        ):
            result = asyncio.run(extractor.extract(description, use_cache=False))
            assert "Pydantic" in result

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

    def test_direct_fallback_raises_on_failed_response(self, extractor: LLMSkillExtractor) -> None:
        """The direct litellm fallback RAISES LLMError on a failed response (empty choices / None
        content) — never returns [] (a masked failure)."""
        import asyncio

        from job_applicator.exceptions import LLMError

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
                with pytest.raises(LLMError):
                    asyncio.run(extractor.extract(description, use_cache=False))

        # None content
        fake_response_none = AsyncMock()
        fake_response_none.choices = [AsyncMock()]
        fake_response_none.choices[0].message.content = None

        with patch("job_applicator.embeddings.skill_extraction.instructor", mock_instructor):
            with patch(
                "job_applicator.embeddings.skill_extraction.acompletion",
                return_value=fake_response_none,
            ):
                with pytest.raises(LLMError):
                    asyncio.run(extractor.extract(description, use_cache=False))

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

    def test_reporter_records_error_then_raises(self, extractor: LLMSkillExtractor) -> None:
        """On an LLM failure the reporter records the error AND extract RAISES — it records the
        failure for the verbose report, then still fails loudly (never silently returns [])."""
        import asyncio

        from job_applicator.exceptions import LLMError
        from job_applicator.utils.verbose import VerboseReporter

        description = "We need Python."
        reporter = VerboseReporter(command="test", args={}, config={})

        with patch.object(extractor, "_call_llm", side_effect=RuntimeError("boom")):
            with pytest.raises(LLMError):
                asyncio.run(extractor.extract(description, use_cache=False, reporter=reporter))

        details = [call["details"] for call in reporter.report.llm.calls]
        assert any(d.get("skill_extraction") == "error" for d in details)
        assert reporter.report.errors


class TestJobMatcherAsyncExtraction:
    """Tests for JobMatcher using LLMSkillExtractor for descriptions."""

    @pytest.fixture
    def matcher(self) -> JobMatcher:
        from job_applicator.embeddings.matching import JobMatcher

        return JobMatcher(
            EmbeddingConfig(device="cpu", memory_limit_gb=0.5),
            LLMConfig(model="test-model"),
        )

    async def test_description_only_job_uses_extractor(
        self, matcher: JobMatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from job_applicator.models import JobBoard, JobListing

        async def fake_extract(
            description: str,
            runtime: object = None,
            use_cache: bool = True,
            reporter: object = None,
        ) -> list[str]:
            return ["Python", "FastAPI"]

        monkeypatch.setattr(matcher._skill_extractor, "extract", fake_extract)

        resume = ResumeData(raw_text="Skills: Python", skills=["Python"])
        job = JobListing(
            title="Backend Dev",
            company="Acme",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
            description="We need Python and FastAPI.",
            requirements=[],
        )
        result = await matcher.match_resume_to_job(resume, job)
        assert "Python" in result.matched_skills
        assert "FastAPI" in result.missing_skills

    async def test_explicit_requirements_bypass_extractor(
        self, matcher: JobMatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from job_applicator.models import JobBoard, JobListing

        called = False

        async def fake_extract(*args: object, **kwargs: object) -> list[str]:
            nonlocal called
            called = True
            return ["Python"]

        monkeypatch.setattr(matcher._skill_extractor, "extract", fake_extract)

        resume = ResumeData(raw_text="Skills: Python", skills=["Python"])
        job = JobListing(
            title="Backend Dev",
            company="Acme",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
            description="...",
            requirements=["Python", "Django"],
        )
        result = await matcher.match_resume_to_job(resume, job)
        assert not called
        assert "Python" in result.matched_skills
        assert "Django" in result.missing_skills

    async def test_no_requirements_yields_semantic_only(
        self, matcher: JobMatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SUCCESSFUL extraction that finds no requirements → rank on semantic similarity ALONE
        (skill_score 0.0), not the old 0.5 neutral floor (L5: no fabricated neutral)."""
        from job_applicator.models import JobBoard, JobListing

        async def fake_extract(*args: object, **kwargs: object) -> list[str]:
            return []  # a genuine, successful "no skills listed" result

        monkeypatch.setattr(matcher._skill_extractor, "extract", fake_extract)

        resume = ResumeData(raw_text="Skills: Python", skills=["Python"])
        job = JobListing(
            title="Backend Dev",
            company="Acme",
            url="https://example.com/1",
            board=JobBoard.LINKEDIN,
            description="We need Python and FastAPI.",
            requirements=[],
        )
        result = await matcher.match_resume_to_job(resume, job)
        assert result.matched_skills == []
        assert result.missing_skills == []
        assert result.skill_score == 0.0  # no floor injected
        assert result.score == pytest.approx(result.semantic_score)  # semantic-only ranking

    async def test_extractor_failure_propagates_not_neutral(
        self, matcher: JobMatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An extraction FAILURE (LLM down) must PROPAGATE as an error — the match must never
        silently degrade to a semantic-only / neutral score (the no-masked-failure rule)."""
        from job_applicator.exceptions import LLMError
        from job_applicator.models import JobBoard, JobListing

        async def boom_extract(*args: object, **kwargs: object) -> list[str]:
            raise LLMError("LLM unreachable")

        monkeypatch.setattr(matcher._skill_extractor, "extract", boom_extract)

        resume = ResumeData(raw_text="Skills: Python", skills=["Python"])
        job = JobListing(
            title="Backend Dev",
            company="Acme",
            url="https://example.com/2",
            board=JobBoard.LINKEDIN,
            description="We need Python.",
            requirements=[],
        )
        with pytest.raises(LLMError):
            await matcher.match_resume_to_job(resume, job)


class TestJobMatcherRanking:
    """Tests for JobMatcher async ranking behavior."""

    @pytest.fixture
    def matcher(self) -> JobMatcher:
        return JobMatcher(
            EmbeddingConfig(device="cpu", memory_limit_gb=0.5),
            LLMConfig(model="test-model"),
        )

    async def test_rank_jobs_is_async_and_sorts_results(
        self,
        matcher: JobMatcher,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """rank_jobs must be awaited and return results sorted by score."""
        from job_applicator.models import JobBoard, JobListing

        # Avoid loading the real embedding model.
        monkeypatch.setattr(
            matcher._service,
            "embed",
            lambda _text: np.zeros(8, dtype=np.float32),
        )
        monkeypatch.setattr(
            matcher._service,
            "embed_batch",
            lambda texts: [np.zeros(8, dtype=np.float32) for _ in texts],
        )
        monkeypatch.setattr(
            matcher._service,
            "similarity",
            lambda _a, _b: 0.5,
        )

        async def fake_match_skills_for_jobs(
            _resume: ResumeData,
            ranked_jobs: list[JobListing],
        ) -> list[tuple[list[str], list[str]]]:
            return [
                (["Python"], ["Django"]) if "Python" in job.requirements else ([], ["Rust"])
                for job in ranked_jobs
            ]

        monkeypatch.setattr(matcher, "_match_skills_for_jobs", fake_match_skills_for_jobs)

        resume = ResumeData(raw_text="Skills: Python", skills=["Python"])
        jobs = [
            JobListing(
                title="Backend Dev",
                company="Acme",
                url="https://example.com/1",
                board=JobBoard.LINKEDIN,
                requirements=["Python", "Django"],
            ),
            JobListing(
                title="Systems Dev",
                company="Beta",
                url="https://example.com/2",
                board=JobBoard.LINKEDIN,
                requirements=["Rust"],
            ),
        ]

        results = await matcher.rank_jobs(resume, jobs, top_k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert results[0].job.company == "Acme"
        assert "Python" in results[0].matched_skills

    async def test_rank_jobs_batches_skill_embeddings_once(
        self,
        matcher: JobMatcher,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ranking must not re-embed the same résumé skills once per job.

        Live QA runs match several jobs through the real embedding model. If ranking sends the
        same résumé skills through sentence-transformers for every job, a CUDA fallback turns a
        small fixture into repeated blocking CPU encode work.
        """
        from job_applicator.models import JobBoard, JobListing

        embed_batch_calls: list[list[str]] = []

        def vector_for(text: str) -> np.ndarray:
            vectors = {
                "Python": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "FastAPI": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                "Django": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            }
            return vectors.get(text, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        monkeypatch.setattr(
            matcher._service,
            "embed",
            lambda _text: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

        def fake_embed_batch(texts: list[str]) -> list[np.ndarray]:
            embed_batch_calls.append(list(texts))
            return [vector_for(text) for text in texts]

        monkeypatch.setattr(matcher._service, "embed_batch", fake_embed_batch)
        monkeypatch.setattr(
            matcher._service,
            "similarity",
            lambda a, b: float(np.dot(a, b)),
        )

        resume = ResumeData(
            raw_text="Skills: Python, FastAPI, Django",
            skills=["Python", "FastAPI", "Django"],
        )
        jobs = [
            JobListing(
                title="Backend Dev",
                company="Acme",
                url="https://example.com/1",
                board=JobBoard.LINKEDIN,
                requirements=["Python", "FastAPI"],
            ),
            JobListing(
                title="Platform Dev",
                company="Beta",
                url="https://example.com/2",
                board=JobBoard.LINKEDIN,
                requirements=["Python", "Django"],
            ),
        ]

        results = await matcher.rank_jobs(resume, jobs, top_k=2)

        assert len(results) == 2
        assert embed_batch_calls == [
            [
                "Job: Backend Dev at Acme | Requirements: Python, FastAPI",
                "Job: Platform Dev at Beta | Requirements: Python, Django",
            ],
            ["Python", "FastAPI", "Django"],
            ["Python", "FastAPI", "Django"],
        ]
        assert results[0].matched_skills
        assert results[1].matched_skills


async def test_match_offloads_blocking_encode_off_event_loop(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The blocking sentence-transformers encode must run OFF the event loop,
    so matching never freezes the TUI / blocks concurrency (CLAUDE.md: async for I/O; CPU work
    offloaded, not run inline on the loop)."""
    from concurrent.futures import Future

    import numpy as np

    from job_applicator.config import EmbeddingConfig, LLMConfig
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.models import JobBoard, JobListing, ResumeData

    matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5), LLMConfig())
    monkeypatch.setattr(
        matcher._service, "embed", lambda *_a, **_k: np.zeros(1024, dtype=np.float32)
    )
    monkeypatch.setattr(
        matcher._service,
        "embed_batch",
        lambda texts, **_k: [np.zeros(1024, dtype=np.float32) for _ in texts],
    )
    monkeypatch.setattr(matcher._service, "_resolve_device", lambda: "cuda")

    calls: list[str] = []

    def _spy_submit(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(getattr(fn, "__name__", str(fn)))
        future: Future[object] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr(matcher._embedding_executor, "submit", _spy_submit)

    resume = ResumeData(raw_text="Python developer", skills=["Python"])
    job = JobListing(
        title="Dev",
        company="Co",
        url="https://x/1",
        board=JobBoard.LINKEDIN,
        requirements=["Python"],
    )
    result = await matcher.match_resume_to_job(resume, job)
    assert result is not None
    assert calls  # embeddings ran through the matcher-owned executor.


async def test_match_explicit_cpu_runs_embedding_inline(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Explicit CPU mode avoids worker-thread PyTorch inference, which can stall on this stack."""
    import numpy as np

    from job_applicator.config import EmbeddingConfig, LLMConfig
    from job_applicator.embeddings.matching import JobMatcher
    from job_applicator.models import JobBoard, JobListing, ResumeData

    matcher = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5), LLMConfig())
    monkeypatch.setattr(
        matcher._service, "embed", lambda *_a, **_k: np.zeros(1024, dtype=np.float32)
    )
    monkeypatch.setattr(
        matcher._service,
        "embed_batch",
        lambda texts, **_k: [np.zeros(1024, dtype=np.float32) for _ in texts],
    )

    def fail_submit(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("explicit CPU mode must not use worker-thread offload")

    monkeypatch.setattr(matcher._embedding_executor, "submit", fail_submit)

    resume = ResumeData(raw_text="Python developer", skills=["Python"])
    job = JobListing(
        title="Dev",
        company="Co",
        url="https://x/1",
        board=JobBoard.LINKEDIN,
        requirements=["Python"],
    )

    result = await matcher.match_resume_to_job(resume, job)

    assert result is not None


def test_combined_score_semantic_only_when_no_requirements() -> None:
    """L5: with no requirements to compare (skill coverage unknown) the score is semantic-ONLY —
    not the old 0.5 floor that added a uniform +0.2 to every such job."""
    import pytest

    from job_applicator.config import EmbeddingConfig, LLMConfig
    from job_applicator.embeddings.matching import JobMatcher

    m = JobMatcher(EmbeddingConfig(device="cpu", memory_limit_gb=0.5), LLMConfig())

    combined, skill = m._combined_score(0.7, [], [])  # unknown skill coverage
    assert combined == pytest.approx(0.7) and skill == 0.0  # semantic-only, not 0.62 (the floor)

    combined2, skill2 = m._combined_score(0.7, ["python"], ["rust"])  # 1/2 covered
    assert skill2 == pytest.approx(0.5)
    assert combined2 == pytest.approx(0.6 * 0.7 + 0.4 * 0.5)  # normal 60/40 blend
