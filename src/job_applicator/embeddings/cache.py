"""Hugging Face cache probing for embedding models."""

from __future__ import annotations

import os
from pathlib import Path


def probe_hf_model_cache(model_name: str) -> tuple[bool, str | None]:
    """Return whether a Hugging Face model snapshot appears cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return _embedding_cache_fallback(model_name)
    hit = try_to_load_from_cache(model_name, "config.json")
    if isinstance(hit, str):
        # .../models--org--name/snapshots/<rev>/config.json -> repo dir is three up.
        return True, str(Path(hit).parent.parent.parent)
    return False, None


def _embedding_cache_fallback(model_name: str) -> tuple[bool, str | None]:
    """Filesystem fallback that honors the common Hugging Face cache env vars."""
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        root = Path(hub)
    else:
        hf_home = os.environ.get("HF_HOME")
        root = (Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface") / "hub"
    repo = root / ("models--" + model_name.replace("/", "--"))
    snapshots = repo / "snapshots"
    if snapshots.is_dir() and any(snapshots.iterdir()):
        return True, str(repo)
    return False, None
