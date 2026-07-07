"""Embedding service using sentence-transformers with mxbai-embed-large-v1.

Provides semantic embeddings for job matching, skill matching, and style similarity.
Allocates about 1.4GB VRAM alongside the main orchestrator (vLLM), with a lower
preflight budget so small desktop/VLLM VRAM fluctuations do not fail healthy loads.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from job_applicator.config import EmbeddingConfig
from job_applicator.embeddings.cache import probe_hf_model_cache
from job_applicator.exceptions import ConfigError
from job_applicator.utils.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger("embeddings.service")

# Type alias for embedding vectors
EmbeddingVector = NDArray[np.float32]
CPU_THREAD_CAP = 4
CPU_INTEROP_THREAD_CAP = 1


class EmbeddingService:
    """Embedding service using sentence-transformers.

    Lazy-loads the model on first use. Caches embeddings to disk.
    Allocates controlled VRAM via model_kwargs.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._resolved_device: str | None = None
        self._cache_dir = Path.home() / ".job-applicator" / "embeddings"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, EmbeddingVector] = {}
        self._memory_cache_lock = threading.Lock()

    def _resolve_device(self) -> str:
        """Resolve the embedding device.

        ``cuda`` is strict: if the user configured CUDA, matching must either run on CUDA
        or fail with an actionable error. CPU is still available when explicitly requested.
        """
        if self._resolved_device is not None:
            return self._resolved_device
        configured = self._config.device
        if not configured.lower().startswith("cuda"):
            self._resolved_device = configured
            return configured
        try:
            import torch
        except ImportError:
            raise ConfigError(
                "embedding.device is set to CUDA but torch is not installed. Install the "
                "CUDA-enabled embedding dependencies or set embedding.device='cpu' explicitly "
                "for a CPU-only environment."
            ) from None
        if not torch.cuda.is_available():
            raise ConfigError(
                "embedding.device is set to CUDA but CUDA is not available to torch. Expose the "
                "NVIDIA device to this process, install a CUDA-enabled torch wheel, or set "
                "embedding.device='cpu' explicitly for degraded CPU matching."
            )
        self._resolved_device = configured
        return configured

    def _configure_cpu_threads(self) -> None:
        """Cap Torch CPU threading for embedding fallback.

        The default 16x16 thread settings on the dev box made tiny CPU embedding batches stall
        long enough to trip live QA timeouts. A small cap is more predictable for CLI workloads.
        """
        try:
            import torch
        except ImportError:
            return

        current_threads = torch.get_num_threads()
        if current_threads > CPU_THREAD_CAP:
            torch.set_num_threads(CPU_THREAD_CAP)
            logger.info(
                "Capped Torch CPU embedding threads: intra-op %d -> %d",
                current_threads,
                CPU_THREAD_CAP,
            )

        try:
            current_interop = torch.get_num_interop_threads()
            if current_interop > CPU_INTEROP_THREAD_CAP:
                torch.set_num_interop_threads(CPU_INTEROP_THREAD_CAP)
                logger.info(
                    "Capped Torch CPU embedding threads: inter-op %d -> %d",
                    current_interop,
                    CPU_INTEROP_THREAD_CAP,
                )
        except RuntimeError as exc:
            logger.debug("Torch inter-op thread cap could not be changed: %s", exc)

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

                cached, cache_path = probe_hf_model_cache(self._config.model_name)
                if cached:
                    logger.info(
                        "Embedding model cache found at %s; loading without network probes",
                        cache_path,
                    )

                resolved_device = self._resolve_device()
                model_kwargs: dict[str, object] = {}
                if resolved_device.lower().startswith("cuda"):
                    self._check_cuda_memory_headroom(resolved_device)
                    model_kwargs["torch_dtype"] = "float16"  # Use FP16 to save VRAM.
                elif resolved_device == "cpu":
                    model_kwargs["torch_dtype"] = "float32"
                    self._configure_cpu_threads()

                # Load model with memory limit. When the snapshot is already cached, force
                # local-only mode so Hugging Face does not block on metadata probes in offline
                # or sandboxed environments. FP16 is only safe for CUDA; CPU fallback is forced
                # to FP32 because some embedding snapshots load half-precision weights by default.
                self._model = SentenceTransformer(
                    self._config.model_name,
                    device=resolved_device,
                    local_files_only=cached,
                    model_kwargs=model_kwargs,
                )

                # Set max sequence length
                if self._model is not None and hasattr(self._model, "max_seq_length"):
                    self._model.max_seq_length = self._config.max_seq_length

                logger.info("Embedding model loaded successfully on %s", resolved_device)
            except ImportError as exc:
                raise ConfigError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                ) from exc
            except ConfigError:
                raise
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    raise ConfigError(
                        "CUDA ran out of memory while loading the embedding model. "
                        "Reduce vLLM GPU_MEM / --gpu-memory-utilization, stop other GPU "
                        "processes, or explicitly set embedding.device='cpu' for degraded "
                        "CPU matching."
                    ) from exc
                raise ConfigError(
                    "Embedding model could not be loaded. If this machine is offline or running "
                    f"in a restricted sandbox, pre-download {self._config.model_name!r} into the "
                    "Hugging Face cache, or run once with network access so sentence-transformers "
                    "can fetch it."
                ) from exc
        return self._model

    def _check_cuda_memory_headroom(self, resolved_device: str) -> None:
        """Fail early when configured CUDA matching does not have enough free VRAM."""
        try:
            import torch

            index = torch.device(resolved_device).index or 0
            free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        except Exception as exc:
            logger.debug("Could not preflight CUDA embedding memory: %s", exc)
            return

        required_bytes = int(self._config.memory_limit_gb * 1024**3)
        if free_bytes < required_bytes:
            free_mib = free_bytes // (1024 * 1024)
            required_mib = required_bytes // (1024 * 1024)
            raise ConfigError(
                "embedding.device is CUDA, but free VRAM is below the configured embedding "
                f"budget ({free_mib} MiB free; {required_mib} MiB required by "
                "embedding.memory_limit_gb). Reduce vLLM GPU_MEM / --gpu-memory-utilization, "
                "stop other GPU processes, or explicitly set embedding.device='cpu' for "
                "degraded CPU matching."
            )

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text, including model and config."""
        # Include max_seq_length: it changes truncation and thus the vector, so a length change
        # must invalidate the cache rather than return a stale, differently-truncated embedding.
        content = (
            f"{self._config.model_name}:{self._config.normalize_embeddings}:"
            f"{self._config.max_seq_length}:{text}"
        )
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def _get_cache_path(self, text: str) -> Path:
        """Get cache file path for text embedding."""
        key = self._get_cache_key(text)
        return self._cache_dir / f"{key}.npy"

    def _get_memory_cached(self, text: str) -> EmbeddingVector | None:
        with self._memory_cache_lock:
            return self._memory_cache.get(self._get_cache_key(text))

    def _set_memory_cached(self, text: str, embedding: EmbeddingVector) -> None:
        with self._memory_cache_lock:
            self._memory_cache[self._get_cache_key(text)] = embedding

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
            memory_cached = self._get_memory_cached(text)
            if memory_cached is not None:
                return memory_cached
            cache_path = self._get_cache_path(text)
            if cache_path.exists():
                try:
                    cached = np.load(str(cache_path))
                    self._set_memory_cached(text, cached)
                    return cached  # type: ignore[no-any-return]
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
            self._set_memory_cached(text, embedding)
            cache_path = self._get_cache_path(text)
            try:
                np.save(str(cache_path), embedding)
            except OSError as exc:
                logger.debug("Could not write embedding cache %s: %s", cache_path, exc)

        return embedding

    def embed_batch(self, texts: list[str], use_cache: bool = True) -> list[EmbeddingVector]:
        """Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed
            use_cache: Whether to check/save to cache

        Returns:
            List of embedding vectors
        """
        # Check memory/disk cache for all texts and deduplicate model work for repeated texts.
        results: list[EmbeddingVector | None] = []
        unique_uncached_texts: list[str] = []
        pending_by_text: dict[str, list[int]] = {}

        for i, text in enumerate(texts):
            if use_cache:
                memory_cached = self._get_memory_cached(text)
                if memory_cached is not None:
                    results.append(memory_cached)
                    continue
                cache_path = self._get_cache_path(text)
                if cache_path.exists():
                    try:
                        cached = np.load(str(cache_path))
                        self._set_memory_cached(text, cached)
                        results.append(cached)
                        continue
                    except Exception as e:
                        logger.debug("Cache miss for index %d: %s", i, e)
            results.append(None)
            if text in pending_by_text:
                pending_by_text[text].append(i)
            else:
                pending_by_text[text] = [i]
                unique_uncached_texts.append(text)

        # Generate embeddings for uncached texts
        if unique_uncached_texts:
            model = self._load_model()
            embeddings = cast(
                list[EmbeddingVector],
                model.encode(
                    unique_uncached_texts,
                    batch_size=self._config.batch_size,
                    normalize_embeddings=self._config.normalize_embeddings,
                    show_progress_bar=len(unique_uncached_texts) > 10,
                ),
            )

            # Save to cache and fill results
            for text, embedding in zip(unique_uncached_texts, embeddings, strict=False):
                if use_cache:
                    self._set_memory_cached(text, embedding)
                    cache_path = self._get_cache_path(text)
                    try:
                        np.save(str(cache_path), embedding)
                    except OSError as exc:
                        logger.debug("Could not write embedding cache %s: %s", cache_path, exc)
                for idx in pending_by_text[text]:
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
