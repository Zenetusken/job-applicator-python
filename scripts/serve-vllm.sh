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
#   GPU_MEM  GPU memory fraction (default: 0.70 — tuned for 12 GB desktops;
#                                 raise/lower for your GPU)
#   MAX_MODEL_LEN  context length (default: 8192). Lowering this shrinks the KV
#                  cache and can let cudagraphs fit on tight GPUs.
#   VLLM_BIN path to vllm exec   (default: this project's .venv/bin/vllm; use
#                                 this to point at a shared distribution from
#                                 another venv without touching its config)
#   ENFORCE_EAGER  disable CUDA graphs (default: 1). Set to 0 if you have
#                  ample VRAM and want cudagraph throughput; leave at 1 on
#                  12 GB cards to avoid the cudagraph-profiling OOM with
#                  Qwen3.5-style hybrid models.
#
# Isolation rule: this script NEVER reads or writes Doc_Flo configuration. By
# default it runs job-applicator's own vLLM binary from .venv/bin/vllm and
# applies job-applicator's own parameters. If another app already has vLLM
# running on the target port, we check whether its command line is compatible;
# use RESTART=1 to stop it and start a fresh instance with this config.
set -euo pipefail

MODEL="${MODEL:-cyankiwi/Qwen3.5-4B-AWQ-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# vLLM 0.23's V1 engine + cudagraph profiling can OOM on 12 GB desktop GPUs with
# Qwen3.5-style hybrid models. 0.70 gives the engine enough headroom for the KV
# cache when CUDA graphs are disabled; raise/lower for your GPU.
GPU_MEM="${GPU_MEM:-0.70}"
# Context length. Lowering this shrinks the KV cache and can let cudagraphs fit
# on tight GPUs (try 4096 with ENFORCE_EAGER=0). Default 8192 matches the model.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
RESTART="${RESTART:-0}"
# Disable CUDA graphs/torch.compile by default to avoid the cudagraph-profiling
# OOM described above. Set ENFORCE_EAGER=0 if you have ample VRAM and want the
# extra throughput from cudagraphs.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# Default to job-applicator's own vLLM binary so we control the CUDA/runtime
# stack independently. Override VLLM_BIN to share a distribution from another
# venv without modifying that other project.
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Help vLLM's optional deep_gemm path find the CUDA 13.0 runtime libraries that
# ship with the vLLM cu130 wheel (e.g. libnvrtc.so.13). This is non-fatal if
# absent, but enables the fastest kernels on supported GPUs.
CU13_LIB="$PROJECT_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "$CU13_LIB" ]; then
    export LD_LIBRARY_PATH="${CU13_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

VLLM_BIN="${VLLM_BIN:-}"
if [ -z "$VLLM_BIN" ]; then
    if [ -x "$PROJECT_DIR/.venv/bin/vllm" ]; then
        VLLM_BIN="$PROJECT_DIR/.venv/bin/vllm"
    elif command -v vllm >/dev/null 2>&1; then
        VLLM_BIN="$(command -v vllm)"
    else
        echo "error: 'vllm' not found — install the serve extra:  pip install -e \".[serve]\"" >&2
        echo "       or point VLLM_BIN at a shared vLLM executable." >&2
        exit 1
    fi
fi

if [ ! -x "$VLLM_BIN" ]; then
    echo "error: VLLM_BIN is not an executable: $VLLM_BIN" >&2
    exit 1
fi

# Auto-select the vLLM tool-call parser for known model families so that
# structured-output / function-calling clients (e.g. instructor) work out of
# the box. Users can override or disable with TOOL_CALL_PARSER=<parser|none>.
_auto_tool_parser() {
    local model_tag="$1"
    # Lower-case, strip org/quants/suffixes for family detection.
    local norm
    norm=$(echo "$model_tag" | tr '[:upper:]' '[:lower:]')
    case "$norm" in
        *qwen3.5*|*qwen3*) echo "qwen3_xml" ;;
        *) echo "" ;;
    esac
}

if [ -z "$TOOL_CALL_PARSER" ]; then
    TOOL_CALL_PARSER=$(_auto_tool_parser "$MODEL")
fi

# Non-blocking HF-token status (gated models need it; public models don't). Honors
# HF_TOKEN / HUGGING_FACE_HUB_TOKEN and the cached login file under HF_HOME (or default).
hf_token_file="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ] || [ -s "$hf_token_file" ]; then
    echo "HF token : detected" >&2
else
    echo "HF token : none — public models OK; for gated models run: huggingface-cli login" >&2
fi

# Build the command line we want to run. We keep this in one place so the
# compatibility check below compares against exactly what we would launch.
_build_cmd() {
    local parser_arg=""
    if [ "$TOOL_CALL_PARSER" != "none" ] && [ -n "$TOOL_CALL_PARSER" ]; then
        parser_arg="--tool-call-parser $TOOL_CALL_PARSER --enable-auto-tool-choice"
    fi
    local eager_arg=""
    if [ "$ENFORCE_EAGER" = "1" ]; then
        eager_arg="--enforce-eager"
    fi
    echo "$VLLM_BIN serve $MODEL --host $HOST --port $PORT --gpu-memory-utilization $GPU_MEM --max-model-len $MAX_MODEL_LEN --enable-prefix-caching $parser_arg $eager_arg"
}

# Find an existing vLLM server process listening on the target port.
_find_running_vllm() {
    local pid
    pid=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ -z "$pid" ]; then
        # Fallback for systems where ss output differs.
        pid=$(netstat -tlnp 2>/dev/null | grep ":$PORT " | grep -oP '/[0-9]+' | head -1 | tr -d '/')
    fi
    echo "$pid"
}

# Inspect the command line of a running vLLM process.
_vllm_cmdline() {
    local pid="$1"
    if [ -z "$pid" ] || [ ! -d "/proc/$pid" ]; then
        echo ""
        return
    fi
    tr '\0' ' ' < "/proc/$pid/cmdline" | sed 's/ *$//'
}

# True when the running process already has the config we need.
_is_compatible_vllm() {
    local cmdline="$1"
    if [ -z "$cmdline" ]; then
        return 1
    fi
    # Must be serving the same model on the same port.
    if ! echo "$cmdline" | grep -q "vllm serve"; then
        return 1
    fi
    if ! echo "$cmdline" | grep -q -- "$MODEL"; then
        return 1
    fi
    if ! echo "$cmdline" | grep -q -- "--port $PORT"; then
        return 1
    fi
    # If we require a tool parser, the running instance must have it.
    if [ "$TOOL_CALL_PARSER" != "none" ] && [ -n "$TOOL_CALL_PARSER" ]; then
        if ! echo "$cmdline" | grep -q -- "--tool-call-parser $TOOL_CALL_PARSER"; then
            return 1
        fi
        if ! echo "$cmdline" | grep -q -- "--enable-auto-tool-choice"; then
            return 1
        fi
    fi
    return 0
}

RUNNING_PID=$(_find_running_vllm)

if [ -n "$RUNNING_PID" ]; then
    RUNNING_CMD=$(_vllm_cmdline "$RUNNING_PID")
    echo "Found existing vLLM process on port $PORT (pid $RUNNING_PID)" >&2
    if _is_compatible_vllm "$RUNNING_CMD"; then
        echo "Existing vLLM is already compatible with job-applicator config." >&2
        echo "Command: $RUNNING_CMD" >&2
        exit 0
    fi
    echo "Existing vLLM is NOT compatible with job-applicator config." >&2
    echo "Running: $RUNNING_CMD" >&2
    echo "Desired: $(_build_cmd)" >&2
    if [ "$RESTART" != "1" ]; then
        echo "Pass RESTART=1 (or --restart) to stop it and start a compatible instance." >&2
        exit 2
    fi
    echo "RESTART=1: stopping pid $RUNNING_PID..." >&2
    kill "$RUNNING_PID" 2>/dev/null || true
    # Wait for the port to be released (up to 15 s).
    for _ in $(seq 1 30); do
        if ! ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
            break
        fi
        sleep 0.5
    done
fi

# Announce what we are about to start.
parser_arg=""
if [ "$TOOL_CALL_PARSER" != "none" ] && [ -n "$TOOL_CALL_PARSER" ]; then
    parser_arg="--tool-call-parser $TOOL_CALL_PARSER"
fi
eager_arg=""
if [ "$ENFORCE_EAGER" = "1" ]; then
    eager_arg="--enforce-eager"
fi
parser_label="${TOOL_CALL_PARSER:-auto}"
if [ "$TOOL_CALL_PARSER" = "none" ]; then
    parser_label="disabled"
fi
echo "Starting vLLM  (model=$MODEL  host=$HOST  port=$PORT  gpu_mem=$GPU_MEM  max_model_len=$MAX_MODEL_LEN  tool_call_parser=$parser_label  enforce_eager=${ENFORCE_EAGER:-0}  binary=$VLLM_BIN)" >&2
exec "$VLLM_BIN" serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching \
    ${parser_arg:+--enable-auto-tool-choice $parser_arg} \
    ${eager_arg}
