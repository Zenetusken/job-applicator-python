"""Embedding service using sentence-transformers with mxbai-embed-large-v1.

Provides semantic embeddings for job matching, skill matching, and style similarity.
Allocates ~1.5GB VRAM alongside the main orchestrator (vLLM).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from job_applicator.config import EmbeddingConfig
from job_applicator.utils.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger("embeddings.service")

# Type alias for embedding vectors
EmbeddingVector = NDArray[np.float32]


class EmbeddingService:
    """Embedding service using sentence-transformers.

    Lazy-loads the model on first use. Caches embeddings to disk.
    Allocates controlled VRAM via model_kwargs.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._cache_dir = Path.home() / ".job-applicator" / "embeddings"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model with VRAM limits."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "Loading embedding model: %s (limit: %.1f GB VRAM)",
                    self._config.model_name,
                    self._config.memory_limit_gb,
                )

                # Load model with memory limit
                self._model = SentenceTransformer(
                    self._config.model_name,
                    device=self._config.device,
                    model_kwargs={
                        "torch_dtype": "float16",  # Use FP16 to save VRAM
                    },
                )

                # Set max sequence length
                if self._model is not None and hasattr(self._model, "max_seq_length"):
                    self._model.max_seq_length = self._config.max_seq_length

                logger.info("Embedding model loaded successfully")
            except ImportError as exc:
                from job_applicator.exceptions import ConfigError

                raise ConfigError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                ) from exc
        return self._model

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text, including model and config."""
        content = f"{self._config.model_name}:{self._config.normalize_embeddings}:{text}"
        return hashlib.md5(content.encode()).hexdigest()  # noqa: S324

    def _get_cache_path(self, text: str) -> Path:
        """Get cache file path for text embedding."""
        key = self._get_cache_key(text)
        return self._cache_dir / f"{key}.npy"

    def embed(self, text: str, use_cache: bool = True) -> EmbeddingVector:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to check/save to cache

        Returns:
            1024-dimensional embedding vector
        """
        # Check cache
        if use_cache:
            cache_path = self._get_cache_path(text)
            if cache_path.exists():
                try:
                    return np.load(str(cache_path))  # type: ignore[no-any-return]
                except Exception as e:
                    logger.debug("Cache miss: %s", e)

        # Generate embedding
        model = self._load_model()
        # cast (not type: ignore) so this type-checks whether or not the
        # optional sentence-transformers extra is installed: encode() returns
        # a union we narrow to EmbeddingVector, and with the extra absent it is
        # Any. A bare ignore would be flagged unused on a clean [dev] install.
        embedding = cast(
            EmbeddingVector,
            model.encode(
                text,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=False,
            ),
        )

        # Save to cache
        if use_cache:
            cache_path = self._get_cache_path(text)
            np.save(str(cache_path), embedding)

        return embedding

    def embed_batch(self, texts: list[str], use_cache: bool = True) -> list[EmbeddingVector]:
        """Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed
            use_cache: Whether to check/save to cache

        Returns:
            List of embedding vectors
        """
        # Check cache for all texts
        results: list[EmbeddingVector | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if use_cache:
                cache_path = self._get_cache_path(text)
                if cache_path.exists():
                    try:
                        results.append(np.load(str(cache_path)))
                        continue
                    except Exception as e:
                        logger.debug("Cache miss for index %d: %s", i, e)
            results.append(None)
            uncached_indices.append(i)
            uncached_texts.append(text)

        # Generate embeddings for uncached texts
        if uncached_texts:
            model = self._load_model()
            embeddings = cast(
                list[EmbeddingVector],
                model.encode(
                    uncached_texts,
                    batch_size=self._config.batch_size,
                    normalize_embeddings=self._config.normalize_embeddings,
                    show_progress_bar=len(uncached_texts) > 10,
                ),
            )

            # Save to cache and fill results
            for idx, embedding in zip(uncached_indices, embeddings, strict=False):
                if use_cache:
                    cache_path = self._get_cache_path(texts[idx])
                    np.save(str(cache_path), embedding)
                results[idx] = embedding

        return [r for r in results if r is not None]

    def similarity(self, vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
        """Compute cosine similarity between two vectors.

        Uses fast dot product when vectors are already normalized.
        Returns:
            Similarity score between -1 and 1
        """
        if self._config.normalize_embeddings:
            return float(np.dot(vec1, vec2))
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def find_most_similar(
        self,
        query: EmbeddingVector,
        candidates: list[EmbeddingVector],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find most similar candidates to query.

        Args:
            query: Query embedding vector
            candidates: List of candidate embedding vectors
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        similarities = [
            (i, self.similarity(query, candidate)) for i, candidate in enumerate(candidates)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def clear_cache(self) -> int:
        """Clear embedding cache. Returns number of files removed."""
        count = 0
        for f in self._cache_dir.glob("*.npy"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cached embeddings", count)
        return count
