#!/usr/bin/env bash
# Optionally self-host the OpenAI-compatible LLM endpoint that job-applicator's
# generation features (cover letters, résumé tailoring, style analysis) call.
#
# job-applicator is normally a CLIENT of an existing endpoint — see the [llm]
# section of config.toml (api_base, default http://localhost:8000/v1). Use this
# script ONLY for a standalone box with no shared/remote LLM. It needs the serve
# extra and a CUDA GPU:
#
#     pip install -e ".[serve]"
#     scripts/serve-vllm.sh
#
# The first run downloads the model from Hugging Face Hub (~4 GB for the default)
# to ~/.cache/huggingface; later runs reuse it. Gated models need HF_TOKEN.
#
# Leave it running in its own terminal (or wrap it in a process manager / systemd
# unit for always-on). Everything is env-overridable:
#
#   MODEL    model id            (default: cyankiwi/Qwen3.5-4B-AWQ-4bit)
#   HOST     bind address        (default: 127.0.0.1)
#   PORT     bind port           (default: 8000 — must match [llm] api_base)
#   GPU_MEM  GPU memory fraction (default: 0.60 — see the GPU layout in README)
set -euo pipefail

MODEL="${MODEL:-cyankiwi/Qwen3.5-4B-AWQ-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.60}"

if ! command -v vllm >/dev/null 2>&1; then
    echo "error: 'vllm' not found — install the serve extra:  pip install -e \".[serve]\"" >&2
    exit 1
fi

# Non-blocking HF-token status (gated models need it; public models don't). Honors
# HF_TOKEN / HUGGING_FACE_HUB_TOKEN and the cached login file under HF_HOME (or default).
hf_token_file="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] || [ -s "$hf_token_file" ]; then
    echo "HF token : detected" >&2
else
    echo "HF token : none — public models OK; for gated models run: huggingface-cli login" >&2
fi

echo "Starting vLLM  (model=$MODEL  host=$HOST  port=$PORT  gpu_mem=$GPU_MEM)" >&2
exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len 8192 \
    --enable-prefix-caching
