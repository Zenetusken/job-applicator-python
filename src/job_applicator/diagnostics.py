"""Health checks for the AI backend — powers ``job-applicator doctor``.

job-applicator is a CLIENT of an OpenAI-compatible LLM endpoint (it never starts
one) and self-manages embeddings in-process. These checks tell a user — especially
on a clean install — whether the pieces are in place.

The probe is intentionally lightweight and side-effect-free: a single GET on
``/models`` to confirm the endpoint answers, NOT a real completion. Only that
reachability (an HTTP 200 from /models) is blocking; auth failures (401/403) are
surfaced distinctly — the endpoint is up, so the fix is the key, not starting a
server. Model-in-list and the embeddings cache are advisory: cloud/Ollama endpoints
name models differently than a local vLLM, and a fresh box downloads on first use.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import httpx

from job_applicator.config import AppSettings, EmbeddingConfig, LLMConfig
from job_applicator.models import (
    DoctorReport,
    EmbeddingsCheck,
    LLMEndpointCheck,
    SelfHostCheck,
)
from job_applicator.utils.logging import get_logger

logger = get_logger("diagnostics")

_PROBE_TIMEOUT_S = 5.0
# Use the same default that LLMConfig uses, so the two can never drift apart.
_LOCAL_API_KEY_PLACEHOLDER: str = LLMConfig.model_fields["api_key"].default


async def check_llm_endpoint(llm: LLMConfig) -> LLMEndpointCheck:
    """Probe the configured OpenAI-compatible endpoint via ``GET {api_base}/models``.

    ``reachable`` means an HTTP response came back at all (the server is up),
    regardless of status — so a 401/403 reads as "reachable but rejected", not
    "down". The blocking ``ok`` signal (on DoctorReport) additionally requires 200.
    ``model_available`` compares the bare configured id (the app only adds the
    ``openai/`` prefix when it calls litellm) and is advisory.
    """
    url = llm.api_base.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    # Only send auth when a real key is set (local vLLM uses a placeholder).
    if llm.api_key and llm.api_key != _LOCAL_API_KEY_PLACEHOLDER:
        headers["Authorization"] = f"Bearer {llm.api_key}"
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
    except (httpx.HTTPError, httpx.InvalidURL, OSError, ValueError) as exc:
        # Connection refused / DNS / timeout / malformed api_base → not reachable.
        logger.debug("LLM endpoint probe failed: %s", exc)
        return LLMEndpointCheck(
            api_base=llm.api_base,
            reachable=False,
            model_configured=llm.model,
            error=str(exc),
        )

    # An HTTP response came back → the server is reachable, whatever the status.
    configured = llm.model.removeprefix("openai/")
    models: list[str] = []
    if resp.status_code == 200:
        try:
            data = resp.json()
            models = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
        except (ValueError, AttributeError, TypeError):
            models = []
    return LLMEndpointCheck(
        api_base=llm.api_base,
        reachable=True,
        model_configured=llm.model,
        http_status=resp.status_code,
        model_available=configured in models,
        models_seen=models,
        error=None if resp.status_code == 200 else f"HTTP {resp.status_code}",
    )


def check_embeddings(emb: EmbeddingConfig) -> EmbeddingsCheck:
    """Report whether the embedding model is already cached (no model load).

    Prefers ``huggingface_hub``'s own cache resolution (honors ``HF_HUB_CACHE`` /
    ``HF_HOME`` and the real on-disk layout); otherwise falls back to a filesystem
    probe that honors the same env vars and requires a non-empty ``snapshots/`` so a
    partial/interrupted download isn't reported as cached.
    """
    cached, path = _probe_embedding_cache(emb.model_name)
    return EmbeddingsCheck(model_name=emb.model_name, cached=cached, cache_path=path)


def _probe_embedding_cache(model_name: str) -> tuple[bool, str | None]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return _embedding_cache_fallback(model_name)
    # A cached repo always has config.json; a real path back means it is present.
    hit = try_to_load_from_cache(model_name, "config.json")
    if isinstance(hit, str):
        # .../models--org--name/snapshots/<rev>/config.json → repo dir is three up.
        return True, str(Path(hit).parent.parent.parent)
    return False, None


def _embedding_cache_fallback(model_name: str) -> tuple[bool, str | None]:
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


def check_self_host() -> SelfHostCheck:
    """Optional prerequisites for self-hosting via scripts/serve-vllm.sh."""
    return SelfHostCheck(
        vllm_installed=shutil.which("vllm") is not None,
        hf_token_present=_hf_token_present(),
    )


def _hf_token_present() -> bool:
    """Whether an HF token is configured. Prefers ``huggingface_hub.get_token()``
    (honors HF_TOKEN, the HF_HOME-relative token, and the stored-tokens file);
    otherwise falls back to env vars + the common token files (honoring HF_HOME)."""
    if _hf_get_token_via_lib():
        return True
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    hf_home = os.environ.get("HF_HOME")
    candidates = [Path(hf_home) / "token"] if hf_home else []
    candidates += [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    return any(f.is_file() and f.stat().st_size > 0 for f in candidates)


def _hf_get_token_via_lib() -> str | None:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    token = get_token()
    return token if isinstance(token, str) and token else None


async def run_diagnostics(settings: AppSettings) -> DoctorReport:
    """Run every check and assemble the report (only an HTTP-200 /models is blocking)."""
    return DoctorReport(
        llm=await check_llm_endpoint(settings.llm),
        embeddings=check_embeddings(settings.embedding),
        self_host=check_self_host(),
    )
