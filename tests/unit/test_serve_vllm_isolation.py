"""Regression tests for the LLM distribution layer's serve script (audit F1, F4).

These pin the self-host isolation guarantees from the LLM-distribution audit:

  F1 — ``scripts/serve-vllm.sh`` must NOT silently adopt a ``vllm`` found on ``$PATH``
       (historically a sibling project's). It runs only the in-project
       ``.venv/bin/vllm`` or an explicit ``VLLM_BIN``; otherwise it errors.
  F4 — the bash ``_auto_tool_parser`` in ``serve-vllm.sh`` and the Python mirror in
       ``diagnostics.py`` must agree for the same model tag — they can't share code
       across the language boundary, so this guards against silent drift.

The serve script exits at binary discovery *before* it touches the GPU/port/exec,
so the failure paths run cleanly without vLLM or a GPU.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVE_SCRIPT = REPO_ROOT / "scripts" / "serve-vllm.sh"


def _run_serve(
    project_dir: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    """Run a COPY of serve-vllm.sh rooted at ``project_dir``.

    The copy lives at ``<project_dir>/scripts/serve-vllm.sh`` so the script's
    ``PROJECT_DIR=$(dirname "$0")/..`` resolves to ``project_dir`` — a throwaway dir
    with no ``.venv/bin/vllm`` — instead of the real repo's installed binary.
    """
    scripts = project_dir / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    dst = scripts / "serve-vllm.sh"
    shutil.copy2(SERVE_SCRIPT, dst)
    dst.chmod(0o755)
    return subprocess.run(  # noqa: S603
        ["bash", str(dst), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_serve_script_does_not_fall_back_to_path_vllm(tmp_path: Path) -> None:
    """F1 (the regression): no in-project binary + no VLLM_BIN + a vllm on $PATH →
    the script ERRORS and must NEVER exec the $PATH vllm."""
    project = tmp_path / "proj"
    project.mkdir()

    # A fake `vllm` on PATH that records (via a sentinel file) if it is ever executed.
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    sentinel = tmp_path / "EXECUTED"
    fake_vllm = fakebin / "vllm"
    fake_vllm.write_text(f"#!/usr/bin/env bash\ntouch '{sentinel}'\n", encoding="utf-8")
    fake_vllm.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"
    env.pop("VLLM_BIN", None)

    result = _run_serve(project, env)

    assert result.returncode == 1, result.stderr
    assert "no isolated vLLM" in result.stderr
    # The crux: the $PATH vllm was discovered (so the hint mentions it) but NEVER run.
    assert not sentinel.exists(), "serve script silently exec'd a $PATH vllm"
    assert "VLLM_BIN=" in result.stderr  # points the user at the explicit opt-in


def test_explicit_vllm_bin_is_honored(tmp_path: Path) -> None:
    """F1 (the opt-in still works): an explicit VLLM_BIN is consulted — a bad path
    reaches the executability check (proving the override is wired), not the
    'no isolated vLLM' branch."""
    project = tmp_path / "proj"
    project.mkdir()
    env = dict(os.environ)
    env["VLLM_BIN"] = str(tmp_path / "nope" / "vllm")  # set but not executable

    result = _run_serve(project, env)

    assert result.returncode == 1
    assert "not an executable" in result.stderr
    assert "no isolated vLLM" not in result.stderr


# --- F4: bash/Python parser-mapping agreement -------------------------------------


def _bash_auto_tool_parser(model_tag: str) -> str:
    """Invoke the REAL bash ``_auto_tool_parser`` from serve-vllm.sh in isolation.

    Extracts just the function body (so sourcing it doesn't launch a server) and
    calls it, returning its stdout (``""`` for no-match)."""
    text = SERVE_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^_auto_tool_parser\(\)\s*\{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert match, "could not extract _auto_tool_parser() from serve-vllm.sh"
    script = f'{match.group(0)}\n_auto_tool_parser "$1"\n'
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", script, "_", model_tag],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _bash_is_compatible_vllm(cmdline: str, *, gpu_mem: str = "0.65") -> bool:
    """Invoke the real bash ``_is_compatible_vllm`` without launching vLLM."""
    text = SERVE_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^_is_compatible_vllm\(\)\s*\{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert match, "could not extract _is_compatible_vllm() from serve-vllm.sh"
    script = (
        f"{match.group(0)}\n"
        "MODEL='Qwen/Qwen3-8B-AWQ'\n"
        "PORT=8000\n"
        f"GPU_MEM='{gpu_mem}'\n"
        "MAX_MODEL_LEN=8192\n"
        "TOOL_CALL_PARSER='qwen3_xml'\n"
        "ENFORCE_EAGER=1\n"
        '_is_compatible_vllm "$1"\n'
    )
    result = subprocess.run(  # noqa: S603
        ["bash", "-c", script, "_", cmdline],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    ("model_tag", "expected"),
    [
        ("cyankiwi/Qwen3.5-4B-AWQ-4bit", "qwen3_xml"),
        ("Qwen/Qwen3-8B-Instruct", "qwen3_xml"),
        ("qwen3.5-4b", "qwen3_xml"),
        ("meta-llama/Llama-3.1-8B-Instruct", ""),
        ("mistralai/Mistral-7B-Instruct-v0.3", ""),
        ("gpt-4o-mini", ""),
    ],
)
def test_bash_and_python_auto_tool_parser_agree(model_tag: str, expected: str) -> None:
    """F4: the bash mapping (serve-vllm.sh) and the Python mirror (diagnostics.py)
    must return the same tool-call parser for the same model tag."""
    from job_applicator import diagnostics

    bash_result = _bash_auto_tool_parser(model_tag)
    py_result = diagnostics._auto_tool_parser(model_tag) or ""  # Python returns None → ""
    assert bash_result == expected, f"bash mapping changed for {model_tag!r}"
    assert py_result == expected, f"python mapping changed for {model_tag!r}"
    assert bash_result == py_result, f"bash/python parser drift for {model_tag!r}"


def test_serve_script_compatibility_checks_gpu_memory_setting() -> None:
    """A different requested GPU_MEM must force RESTART=1 to replace the server."""
    cmdline = (
        "/repo/.venv/bin/vllm serve Qwen/Qwen3-8B-AWQ --host 127.0.0.1 --port 8000 "
        "--gpu-memory-utilization 0.65 --max-model-len 8192 --enable-prefix-caching "
        "--tool-call-parser qwen3_xml --enable-auto-tool-choice --enforce-eager"
    )

    assert _bash_is_compatible_vllm(cmdline, gpu_mem="0.65")
    assert not _bash_is_compatible_vllm(cmdline, gpu_mem="0.60")
