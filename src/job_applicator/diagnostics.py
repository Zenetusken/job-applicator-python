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

Browser, system-binary, and config checks are also advisory: a headless server may
use only the match/tailor pipeline and intentionally skip browser features.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Environment, PackageLoader

from job_applicator.config import (
    CONFIG_FILE_ENV_VAR,
    DEFAULT_CONFIG_FILE,
    AppSettings,
    EmbeddingConfig,
    LLMConfig,
)
from job_applicator.documents.formatted_models import (
    FormattedEducationEntry,
    FormattedExperienceEntry,
    FormattedResume,
    FormattedSkillGroup,
)
from job_applicator.documents.pdf_renderer import _typst_escape
from job_applicator.embeddings.cache import probe_hf_model_cache
from job_applicator.models import (
    BrowserCheck,
    CapabilityReadiness,
    ConfigCheck,
    DoctorReadiness,
    DoctorReport,
    EmbeddingsCheck,
    LLMEndpointCheck,
    PDFRenderingCheck,
    SelfHostCheck,
    SystemBinariesCheck,
    VLLMProcessCheck,
)
from job_applicator.utils.logging import get_logger


def _get_async_playwright() -> Any:
    try:
        from playwright.async_api import async_playwright
    except ImportError:  # pragma: no cover — core dep, but keep doctor robust.
        return None
    return async_playwright


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
    """Report embedding cache and lightweight runtime readiness (no model load).

    Prefers ``huggingface_hub``'s own cache resolution (honors ``HF_HUB_CACHE`` /
    ``HF_HOME`` and the real on-disk layout); otherwise falls back to a filesystem
    probe that honors the same env vars and requires a non-empty ``snapshots/`` so a
    partial/interrupted download isn't reported as cached.
    """
    cached, path = probe_hf_model_cache(emb.model_name)
    sentence_transformers_available = importlib.util.find_spec("sentence_transformers") is not None
    configured_device = emb.device
    requested_cuda = configured_device.lower().startswith("cuda")
    resolved_device: str | None = None
    device_ready = False
    runtime_error: str | None = None
    torch_available = False
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_available = False
    cuda_device_count = 0
    cuda_device_name: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None

    try:
        import torch

        torch_available = True
        torch_version = str(torch.__version__)
        torch_cuda_version = str(torch.version.cuda) if torch.version.cuda else None
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device_count = int(torch.cuda.device_count())
            if cuda_device_count:
                cuda_device_name = str(torch.cuda.get_device_name(0))
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                    vram_free_mb = int(free_bytes // (1024 * 1024))
                    vram_total_mb = int(total_bytes // (1024 * 1024))
                except Exception as exc:  # pragma: no cover - driver/runtime dependent.
                    logger.debug("Could not read CUDA memory info: %s", exc)
    except ImportError:
        runtime_error = "torch is not installed"
    except Exception as exc:
        runtime_error = f"torch/CUDA probe failed: {exc}"

    if not sentence_transformers_available:
        runtime_error = "sentence-transformers is not installed"
    elif not torch_available:
        runtime_error = runtime_error or "torch is not installed"
    elif requested_cuda:
        if not torch_available:
            runtime_error = runtime_error or "torch is not installed"
        elif not cuda_available:
            runtime_error = runtime_error or "CUDA is not available to torch"
        else:
            required_mib = int(emb.memory_limit_gb * 1024)
            if vram_free_mb is not None and vram_free_mb < required_mib:
                runtime_error = (
                    f"CUDA free VRAM is below embedding.memory_limit_gb "
                    f"({vram_free_mb} MiB free; {required_mib} MiB required)"
                )
            else:
                resolved_device = configured_device
                device_ready = True
    else:
        resolved_device = configured_device
        device_ready = True

    return EmbeddingsCheck(
        model_name=emb.model_name,
        cached=cached,
        cache_path=path,
        configured_device=configured_device,
        resolved_device=resolved_device,
        device_ready=device_ready,
        sentence_transformers_available=sentence_transformers_available,
        torch_available=torch_available,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb,
        runtime_error=runtime_error,
    )


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


async def check_browser(channel: str | None = None) -> BrowserCheck:
    """Check that Playwright is installed and can locate a Chromium executable.

    When ``channel="chrome"`` (the default) the scrape launches the host's REAL Chrome via
    Playwright's channel, not the bundled Chromium — so resolve that binary too, and the doctor
    reports the engine actually used (warning if it's requested but absent → silent fallback).
    """
    from job_applicator.utils.region import host_chrome_path

    host_chrome = host_chrome_path() if channel == "chrome" else None
    async_playwright = _get_async_playwright()
    if async_playwright is None:
        return BrowserCheck(
            playwright_installed=False,
            channel=channel,
            host_chrome=host_chrome,
            error="Playwright not installed; run: playwright install chromium",
        )

    try:
        async with async_playwright() as p:
            executable = p.chromium.executable_path
            return BrowserCheck(
                playwright_installed=True,
                chromium_executable=str(executable) if executable else None,
                channel=channel,
                host_chrome=host_chrome,
            )
    except Exception as exc:  # broad by design for install-state probing
        logger.debug("Browser check failed: %s", exc)
        return BrowserCheck(
            playwright_installed=True, channel=channel, host_chrome=host_chrome, error=str(exc)
        )


def check_system_binaries() -> SystemBinariesCheck:
    """Check for optional system binaries used by resume parsing and Indeed display."""
    pdftotext = shutil.which("pdftotext")
    # Xvfb is the display server; xvfb-run is a convenience wrapper. Either satisfies
    # the " headed windowless" requirement for the Indeed scraper.
    xvfb = shutil.which("Xvfb") or shutil.which("xvfb-run")
    return SystemBinariesCheck(
        pdftotext_available=pdftotext is not None,
        xvfb_available=xvfb is not None,
        pdftotext_path=pdftotext,
        xvfb_path=xvfb,
    )


def _config_file_path() -> Path:
    return Path(os.environ.get(CONFIG_FILE_ENV_VAR, DEFAULT_CONFIG_FILE))


def _has_plaintext_credentials(config_file: Path) -> bool:
    """Return True if the TOML config file sets board credentials directly.

    We inspect the raw file (not the merged settings) so env-var overrides are not
    falsely flagged. The goal is to warn users who still have passwords in
    ``config.toml`` even though the auth model is headed/manual.
    """
    if not config_file.is_file():
        return False
    try:
        with config_file.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return False

    target = data.get("target", {})
    credential_keys = ("linkedin_email", "linkedin_password", "indeed_email", "indeed_password")
    return any(target.get(key) for key in credential_keys)


def check_config(settings: AppSettings) -> ConfigCheck:
    """Check config file presence, parseability, and a few security/path hints."""
    config_file = _config_file_path()
    result = ConfigCheck(
        config_file_found=config_file.is_file(),
        config_file_path=str(config_file.resolve()) if config_file.is_file() else str(config_file),
    )

    if not result.config_file_found:
        result.config_file_parseable = False
        return result

    try:
        with config_file.open("rb") as fh:
            tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        result.config_file_parseable = False
        result.error = str(exc)
        return result

    result.plaintext_credentials = _has_plaintext_credentials(config_file)
    result.resume_path_set = bool(settings.resume_path)
    if result.resume_path_set:
        rp = Path(settings.resume_path)
        result.resume_path_exists = rp.is_file()
        if result.resume_path_exists:
            # Surface the résumé's IDENTITY + age + parsed-skill count so `doctor` makes the CV in
            # play VISIBLE (the real guard against a stale/wrong config.resume_path — a threshold
            # can't tell "Resume.docx, 2yr old" from the current one; a human eyeballing it can).
            result.resume_filename = rp.name
            try:
                result.resume_age_days = int((time.time() - rp.stat().st_mtime) / 86400)
            except OSError:
                pass
            # Parse WITHOUT OCR — a diagnostic must stay fast; if the file needs OCR, a 0-skill
            # parse is itself the signal. Best-effort: a parse failure becomes a soft note, never a
            # crash (the core config check must still return).
            try:
                from job_applicator.documents.resume import ResumeLoader

                result.resume_parsed_skills = len(
                    ResumeLoader().load(str(rp), ocr_mode="off").skills
                )
            except Exception as exc:  # advisory: any parse failure → a soft note, never a crash
                # Surface the actual message — load() raises DocumentError with an actionable one
                # ("password-protected…", "enable OCR…"); the type name alone throws it away.
                result.resume_sanity_note = f"could not parse résumé — {exc}"
            # Soft SECONDARY warnings (the surfaced facts above are the primary signal).
            notes: list[str] = []
            if result.resume_parsed_skills == 0:
                notes.append("parsed 0 skills (image-only / wrong format / mis-structured?)")
            if result.resume_age_days is not None and result.resume_age_days > 365:
                notes.append(f"~{result.resume_age_days // 30} months old — is it your current CV?")
            if notes and not result.resume_sanity_note:
                result.resume_sanity_note = "; ".join(notes)
    # A diagnostic must not create directories: probe the writability of the output
    # dir (or the nearest existing ancestor it would be created in) WITHOUT
    # ensure_output_dir()'s mkdir side-effect.
    probe = Path(settings.output_dir)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    result.output_dir_writable = probe.is_dir() and os.access(probe, os.W_OK)

    return result


def _pdf_smoke_resume() -> FormattedResume:
    """Return a minimal ``FormattedResume`` for the PDF doctor smoke test."""
    return FormattedResume(
        name="Smoke Test",
        title="PDF Rendering Check",
        email="smoke@example.com",
        phone="555-555-5555",
        location="City",
        summary="A minimal résumé used to verify the PDF rendering toolchain.",
        experience=[
            FormattedExperienceEntry(
                title="Engineer",
                company="Example Inc",
                start_date="2020",
                end_date="Present",
                bullets=["Built things."],
            )
        ],
        education=[
            FormattedEducationEntry(
                degree="BS",
                institution="University",
                start_date="2015",
                end_date="2019",
            )
        ],
        skills=[FormattedSkillGroup(category="Languages", skills=["Python"])],
    )


def _auto_tool_parser(model_tag: str) -> str | None:
    """Mirror of serve-vllm.sh: pick a parser for known local model families."""
    norm = model_tag.lower()
    if "qwen3.5" in norm or "qwen3" in norm:
        return "qwen3_xml"
    return None


def _parse_api_base_port(api_base: str) -> int:
    """Extract the port from an http://host:port/v1 style api_base."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(api_base)
        return parsed.port or 8000
    except Exception:
        return 8000


def check_vllm_process(settings: AppSettings) -> VLLMProcessCheck:
    """Check whether a local vLLM process is running and whether its command line
    matches the configuration job-applicator would start it with.

    This is intentionally separate from the endpoint HTTP probe: the endpoint may
    be a remote/cloud provider, in which case there is no local process to inspect.
    """
    from job_applicator.models import VLLMProcessCheck

    port = _parse_api_base_port(settings.llm.api_base)
    # Only inspect local endpoints.
    api_base = settings.llm.api_base.lower()
    if "localhost" not in api_base and "127.0.0.1" not in api_base:
        return VLLMProcessCheck(running=False)

    pid = _listening_pid(port)
    if not pid or not Path(f"/proc/{pid}").exists():
        return VLLMProcessCheck(running=False)

    cmdline = _proc_cmdline(pid)
    binary_path = cmdline.split()[0] if cmdline else None
    if "vllm serve" not in cmdline:
        return VLLMProcessCheck(
            running=True, pid=pid, command=cmdline, binary_path=binary_path, compatible=False
        )

    return _assess_vllm_command(pid=pid, cmdline=cmdline, settings=settings, port=port)


def _listening_pid(port: int) -> int | None:
    """Return the PID listening on a local TCP port, if visible."""
    for cmd in (
        f"ss -tlnp 2>/dev/null | grep ':{port} '",
        f"netstat -tlnp 2>/dev/null | grep ':{port} '",
        f"lsof -i :{port} 2>/dev/null | grep LISTEN",
    ):
        try:
            import subprocess  # nosec B404

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)  # noqa: S602
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().splitlines()[0]
                pid = _parse_listening_pid(line)
                if pid:
                    return pid
        except Exception as exc:
            logger.debug("vLLM process probe command failed: %s", exc)
            continue
    return None


def _parse_listening_pid(line: str) -> int | None:
    """Parse PID output from ss/netstat/lsof lines."""
    for token in line.replace(",", " ").replace("/", " ").split():
        if token.startswith("pid="):
            try:
                return int(token.split("=")[1])
            except ValueError:
                continue
        if token.isdigit():
            return int(token)
    return None


def _proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except Exception:
        return ""


def _assess_vllm_command(
    *, pid: int, cmdline: str, settings: AppSettings, port: int
) -> VLLMProcessCheck:
    binary_path = cmdline.split()[0] if cmdline else None
    model = settings.llm.model.removeprefix("openai/")
    desired_parser = _auto_tool_parser(model)
    reasons: list[str] = []

    if model not in cmdline:
        reasons.append(f"serving a different model than {model}")
    if f"--port {port}" not in cmdline:
        reasons.append(f"not listening on port {port}")
    if desired_parser:
        if f"--tool-call-parser {desired_parser}" not in cmdline:
            reasons.append(f"missing --tool-call-parser {desired_parser}")
        if "--enable-auto-tool-choice" not in cmdline:
            reasons.append("missing --enable-auto-tool-choice")

    compatible = not reasons
    return VLLMProcessCheck(
        running=True,
        pid=pid,
        command=cmdline,
        binary_path=binary_path,
        compatible=compatible,
        needs_restart_reason="; ".join(reasons) if reasons else None,
    )


def check_pdf_rendering() -> PDFRenderingCheck:
    """Smoke-test the PDF rendering toolchain (typst package + built-in template)."""
    try:
        import typst
    except ImportError:
        return PDFRenderingCheck(
            ok=False,
            message="typst package not installed; run pip install 'job-applicator[pdf]'",
        )

    env = Environment(
        loader=PackageLoader("job_applicator", "templates"),
        autoescape=False,  # noqa: S701
    )
    env.filters["typst_escape"] = _typst_escape

    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "smoke.typ"
        output_path = Path(tmp) / "smoke.pdf"
        try:
            template = env.get_template("cv/modern.typ")
        except Exception as exc:
            return PDFRenderingCheck(
                ok=False, message=f"could not load built-in CV template: {exc}"
            )
        source_path.write_text(template.render(resume=_pdf_smoke_resume()), encoding="utf-8")
        try:
            typst.compile(str(source_path), output=str(output_path), format="pdf")
            if output_path.exists() and output_path.stat().st_size > 0:
                return PDFRenderingCheck(ok=True, message="typst compile works")
            return PDFRenderingCheck(ok=False, message="typst produced empty PDF")
        except Exception as exc:  # pragma: no cover - typst runtime errors vary by host
            return PDFRenderingCheck(ok=False, message=f"typst compile failed: {exc}")


def build_readiness(
    *,
    llm: LLMEndpointCheck,
    embeddings: EmbeddingsCheck,
    browser: BrowserCheck,
    config: ConfigCheck,
    pdf_rendering: PDFRenderingCheck,
) -> DoctorReadiness:
    """Build capability-level readiness without changing DoctorReport.ok semantics."""
    llm_ready = llm.reachable and llm.http_status == 200
    ai_details = "LLM endpoint reachable" if llm_ready else "LLM endpoint is not ready"

    matching_ready = embeddings.cached and embeddings.device_ready
    matching_details_parts = [
        "embedding model cached locally"
        if embeddings.cached
        else "embedding model will need first-use network download"
    ]
    if embeddings.device_ready:
        matching_details_parts.append(
            f"embedding device ready ({embeddings.resolved_device or embeddings.configured_device})"
        )
    else:
        matching_details_parts.append(embeddings.runtime_error or "embedding runtime is not ready")
    matching_details = "; ".join(matching_details_parts)

    browser_ready = browser.playwright_installed and (
        bool(browser.chromium_executable) or bool(browser.host_chrome)
    )
    browser_details = (
        "browser engine available"
        if browser_ready
        else "Playwright/Chromium browser engine is not ready"
    )

    pdf_ready = pdf_rendering.ok
    pdf_details = pdf_rendering.message

    if not config.resume_path_set:
        matching_ready = False
        matching_details += "; resume_path is not configured"
    elif not config.resume_path_exists:
        matching_ready = False
        matching_details += "; configured resume_path does not exist"

    return DoctorReadiness(
        ai_generation=CapabilityReadiness(ready=llm_ready, details=ai_details),
        matching=CapabilityReadiness(ready=matching_ready, details=matching_details),
        browser_workflows=CapabilityReadiness(ready=browser_ready, details=browser_details),
        pdf_output=CapabilityReadiness(ready=pdf_ready, details=pdf_details),
    )


async def run_diagnostics(settings: AppSettings) -> DoctorReport:
    """Run every check and assemble the report (only an HTTP-200 /models is blocking).

    The two independent async probes (LLM endpoint + browser launch) run
    concurrently; the remaining checks are sync filesystem/shutil lookups.
    """
    llm, browser = await asyncio.gather(
        check_llm_endpoint(settings.llm), check_browser(settings.browser.channel)
    )
    embeddings = check_embeddings(settings.embedding)
    self_host = check_self_host()
    system = check_system_binaries()
    config = check_config(settings)
    vllm_process = check_vllm_process(settings)
    pdf_rendering = check_pdf_rendering()
    return DoctorReport(
        llm=llm,
        embeddings=embeddings,
        self_host=self_host,
        browser=browser,
        system=system,
        config=config,
        vllm_process=vllm_process,
        pdf_rendering=pdf_rendering,
        readiness=build_readiness(
            llm=llm,
            embeddings=embeddings,
            browser=browser,
            config=config,
            pdf_rendering=pdf_rendering,
        ),
    )
