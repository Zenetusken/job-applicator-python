"""Unit tests for the AI-backend diagnostics (`job-applicator doctor`).

The HTTP probe and huggingface_hub lookups are mocked, so these stay fast unit
tests (no network, no GPU, no model loads).
"""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from litellm.exceptions import APIConnectionError, Timeout

from job_applicator import cli, diagnostics
from job_applicator.config import AppSettings, EmbeddingConfig, LLMConfig
from job_applicator.models import (
    BrowserCheck,
    ConfigCheck,
    DoctorReport,
    EmbeddingsCheck,
    LLMEndpointCheck,
    PDFRenderingCheck,
    SelfHostCheck,
    SystemBinariesCheck,
)
from job_applicator.utils.llm import llm_call_error


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """httpx.AsyncClient stand-in that returns a canned response or raises, and
    records the request headers into a caller-owned dict (no shared class state)."""

    def __init__(
        self, *, resp: _FakeResponse | None, exc: Exception | None, captured: dict
    ) -> None:
        self._resp = resp
        self._exc = exc
        self._captured = captured

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def get(self, url: str, headers: dict | None = None) -> _FakeResponse:
        self._captured["headers"] = headers or {}
        if self._exc is not None:
            raise self._exc
        assert self._resp is not None
        return self._resp


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resp: _FakeResponse | None = None,
    exc: Exception | None = None,
) -> dict:
    """Patch httpx.AsyncClient; return a dict that captures the request headers."""
    captured: dict = {}

    def factory(*_args: object, **_kwargs: object) -> _FakeClient:
        return _FakeClient(resp=resp, exc=exc, captured=captured)

    monkeypatch.setattr(diagnostics.httpx, "AsyncClient", factory)
    return captured


def _models_payload(*ids: str) -> dict:
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


# --- check_llm_endpoint: reachability semantics ----------------------------


async def test_endpoint_200_model_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, resp=_FakeResponse(200, _models_payload("my-model")))
    res = await diagnostics.check_llm_endpoint(LLMConfig(model="my-model"))
    assert res.reachable
    assert res.http_status == 200
    assert res.model_available
    assert res.error is None


async def test_endpoint_200_model_absent_is_advisory(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, resp=_FakeResponse(200, _models_payload("other")))
    res = await diagnostics.check_llm_endpoint(LLMConfig(model="my-model"))
    assert res.reachable
    assert not res.model_available
    assert res.models_seen == ["other"]


async def test_endpoint_strips_openai_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    # Config may carry the litellm 'openai/' prefix; /models lists the bare id.
    _patch_client(monkeypatch, resp=_FakeResponse(200, _models_payload("my-model")))
    res = await diagnostics.check_llm_endpoint(LLMConfig(model="openai/my-model"))
    assert res.model_available


async def test_endpoint_401_is_reachable_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    # Auth failure: the server is UP (reachable), but not usable → distinct from "down".
    _patch_client(monkeypatch, resp=_FakeResponse(401))
    res = await diagnostics.check_llm_endpoint(LLMConfig())
    assert res.reachable
    assert res.http_status == 401
    assert res.error == "HTTP 401"


async def test_endpoint_503_is_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, resp=_FakeResponse(503))
    res = await diagnostics.check_llm_endpoint(LLMConfig())
    assert res.reachable  # an HTTP response came back
    assert res.http_status == 503
    assert res.error == "HTTP 503"


async def test_endpoint_connection_error_not_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, exc=httpx.ConnectError("connection refused"))
    res = await diagnostics.check_llm_endpoint(LLMConfig())
    assert not res.reachable
    assert res.http_status is None
    assert res.error


async def test_endpoint_invalid_url_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # httpx.InvalidURL is NOT a subclass of httpx.HTTPError — must be caught explicitly.
    _patch_client(monkeypatch, exc=httpx.InvalidURL("malformed"))
    res = await diagnostics.check_llm_endpoint(LLMConfig())
    assert not res.reachable
    assert res.error


async def test_endpoint_auth_header_only_with_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _patch_client(monkeypatch, resp=_FakeResponse(200, _models_payload()))
    await diagnostics.check_llm_endpoint(LLMConfig(api_key="not-needed-for-local"))
    assert "Authorization" not in cap["headers"]

    cap = _patch_client(monkeypatch, resp=_FakeResponse(200, _models_payload()))
    await diagnostics.check_llm_endpoint(LLMConfig(api_key="real-key"))
    assert cap["headers"].get("Authorization") == "Bearer real-key"


# --- check_embeddings: huggingface_hub resolution + fallback ---------------


def test_embeddings_via_hub_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = "/cache/hub/models--org--name/snapshots/abc/config.json"
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: cfg)
    res = diagnostics.check_embeddings(EmbeddingConfig(model_name="org/name"))
    assert res.cached
    assert res.cache_path == "/cache/hub/models--org--name"


def test_embeddings_via_hub_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    res = diagnostics.check_embeddings(EmbeddingConfig(model_name="org/name"))
    assert not res.cached
    assert res.cache_path is None


def test_embeddings_fallback_honors_hf_hub_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Fallback path (library absent): must honor HF_HUB_CACHE and require a real snapshot.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--org--name" / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    cached, path = diagnostics._embedding_cache_fallback("org/name")
    assert cached
    assert path is not None


def test_embeddings_fallback_empty_snapshots_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A partial/interrupted download (no snapshot contents) must NOT read as cached.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    (tmp_path / "models--org--name" / "snapshots").mkdir(parents=True)
    cached, path = diagnostics._embedding_cache_fallback("org/name")
    assert not cached
    assert path is None


# --- check_self_host: token via library + fallback -------------------------


def test_self_host_token_via_lib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _: "/usr/bin/vllm")
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "tok")
    res = diagnostics.check_self_host()
    assert res.vllm_installed
    assert res.hf_token_present


def test_self_host_token_fallback_honors_hf_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _: None)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    empty_home = tmp_path / "home"  # isolate from the real ~/.cache/huggingface/token
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "token").write_text("secret")
    res = diagnostics.check_self_host()
    assert not res.vllm_installed
    assert res.hf_token_present  # found $HF_HOME/token, not a real one


def test_self_host_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _: None)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # empty home → no token files anywhere
    res = diagnostics.check_self_host()
    assert not res.hf_token_present


# --- DoctorReport.ok (HTTP 200 is the only blocking signal) ----------------


def _report(*, reachable: bool, http_status: int | None, model_available: bool) -> DoctorReport:
    return DoctorReport(
        llm=LLMEndpointCheck(
            api_base="x",
            reachable=reachable,
            model_configured="m",
            http_status=http_status,
            model_available=model_available,
        ),
        embeddings=EmbeddingsCheck(model_name="m", cached=False),
        self_host=SelfHostCheck(vllm_installed=False, hf_token_present=False),
        browser=BrowserCheck(playwright_installed=True, chromium_executable="/bin/chromium"),
        system=SystemBinariesCheck(
            pdftotext_available=True, xvfb_available=True, pdftotext_path="/bin/pdftotext"
        ),
        config=ConfigCheck(config_file_found=True, config_file_path="config.toml"),
        pdf_rendering=PDFRenderingCheck(ok=False, message="not checked"),
    )


def test_ok_requires_http_200() -> None:
    assert _report(reachable=True, http_status=200, model_available=True).ok
    # model mismatch is advisory — still ok (green) for cloud/Ollama.
    assert _report(reachable=True, http_status=200, model_available=False).ok
    # reachable but auth-rejected → NOT ok.
    assert not _report(reachable=True, http_status=401, model_available=True).ok
    # unreachable → NOT ok.
    assert not _report(reachable=False, http_status=None, model_available=True).ok


def test_doctor_json_emits_report_and_preserves_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """`doctor --json` emits the DoctorReport as JSON (no Rich table) and preserves the
    exit code (non-zero when not ok)."""
    import json
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    report = _report(reachable=False, http_status=None, model_available=True)  # not ok
    monkeypatch.setattr(
        "job_applicator.diagnostics.run_diagnostics", AsyncMock(return_value=report)
    )
    result = CliRunner().invoke(cli.app, ["doctor", "--json"])
    assert result.exit_code == 1  # not ok → exit 1, preserved under --json
    parsed = json.loads(result.stdout)  # raises if stdout isn't pure JSON
    assert parsed["ok"] is False  # headline verdict included (ok is a @property)
    assert parsed["llm"]["reachable"] is False
    assert "job-applicator doctor" not in result.stdout  # the Rich health view is suppressed


def test_doctor_report_round_trips() -> None:
    # computed-free `ok` + extra='forbid' must survive dump → validate.
    rep = _report(reachable=True, http_status=401, model_available=False)
    again = DoctorReport.model_validate(rep.model_dump())
    assert again.ok == rep.ok


# --- _render_doctor never crashes on markup-shaped dynamic values ----------


@pytest.mark.parametrize("status", [200, 401, 503, None])
def test_render_doctor_escapes_dynamic_values(status: int | None) -> None:
    rep = DoctorReport(
        llm=LLMEndpointCheck(
            api_base="http://[x]/v1",
            reachable=status is not None,
            model_configured="bad-[red]-id",
            http_status=status,
            model_available=False,
            models_seen=["a [/] b"],
            error="boom [/] [red] boom" if status != 200 else None,
        ),
        embeddings=EmbeddingsCheck(model_name="m-[/]", cached=False),
        self_host=SelfHostCheck(vllm_installed=False, hf_token_present=False),
        browser=BrowserCheck(playwright_installed=False, error="[red]fail[/red]"),
        system=SystemBinariesCheck(
            pdftotext_available=False,
            xvfb_available=False,
            pdftotext_path="/[/]bin/pdftotext",
        ),
        config=ConfigCheck(
            config_file_found=True,
            config_file_path="[x]config.toml",
            plaintext_credentials=True,
            error="boom [/]",
        ),
        pdf_rendering=PDFRenderingCheck(ok=False, message="[red]fail[/red]"),
    )
    cli._render_doctor(rep)  # must not raise rich.markup.MarkupError


# --- llm_call_error: typed classification (REAL litellm exceptions) --------


def test_llm_call_error_timeout_is_reachable_but_slow() -> None:
    exc = Timeout(message="read timed out", model="m", llm_provider="openai")
    msg = str(llm_call_error(exc, "http://x/v1"))
    assert "timed out" in msg
    assert "Start one" not in msg  # a timeout must NOT say "start a server"


def test_llm_call_error_connection_is_unreachable() -> None:
    exc = APIConnectionError(message="refused", llm_provider="openai", model="m")
    msg = str(llm_call_error(exc, "http://x/v1"))
    assert "Can't reach the LLM endpoint at http://x/v1" in msg
    assert "doctor" in msg


def test_llm_call_error_string_fallback_for_untyped_connection() -> None:
    msg = str(llm_call_error(OSError("Connection refused"), "http://x/v1"))
    assert "Can't reach" in msg


def test_llm_call_error_other_errors_verbatim() -> None:
    msg = str(llm_call_error(ValueError("bad prompt"), "http://x/v1"))
    assert "LLM call failed" in msg
    assert "doctor" not in msg


async def test_llm_call_error_classifies_real_litellm_failure() -> None:
    """The wiring that runs in production: a real litellm call to a dead endpoint must
    classify as unreachable. Locks in _CONNECTION_MARKERS against future edits — litellm
    wraps a refused connection as InternalServerError, which only the string fallback
    catches (a typed-only check would silently miss it). Offline + fast (refused port)."""
    import litellm

    try:
        await litellm.acompletion(
            model="openai/m",
            api_base="http://127.0.0.1:9999/v1",
            api_key="x",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            num_retries=0,
        )
    except Exception as exc:
        assert "Can't reach the LLM endpoint" in str(
            llm_call_error(exc, "http://127.0.0.1:9999/v1")
        )
    else:
        pytest.fail("expected a connection failure from the dead endpoint")


# --- config-init sources the model from LLMConfig (no drift) ---------------


def test_config_init_uses_llmconfig_defaults(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    out = tmp_path / "config.toml"
    result = CliRunner().invoke(cli.app, ["config-init", "-o", str(out)])
    assert result.exit_code == 0
    text = out.read_text()
    assert f'model = "{LLMConfig.model_fields["model"].default}"' in text
    assert f'api_base = "{LLMConfig.model_fields["api_base"].default}"' in text


def test_config_init_unwritable_path_is_clean_error(tmp_path: Path) -> None:
    """config-init to an unwritable path → clean message + exit 1, not a raw traceback."""
    from typer.testing import CliRunner

    bad = tmp_path / "missing-dir" / "config.toml"  # parent dir absent → write raises OSError
    result = CliRunner().invoke(cli.app, ["config-init", "-o", str(bad)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "Could not write config" in result.output


# --- check_browser: Playwright + Chromium presence -------------------------


async def test_browser_check_detects_missing_playwright(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "_get_async_playwright", lambda: None)
    res = await diagnostics.check_browser()
    assert not res.playwright_installed
    assert res.error


async def test_browser_check_reports_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeBrowserType:
        executable_path = "/fake/chromium"

    class _FakePlaywright:
        chromium = _FakeBrowserType()

    class _FakeContext:
        async def __aenter__(self) -> _FakePlaywright:
            return _FakePlaywright()

        async def __aexit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(diagnostics, "_get_async_playwright", lambda: lambda: _FakeContext())
    res = await diagnostics.check_browser()
    assert res.playwright_installed
    assert res.chromium_executable == "/fake/chromium"


# --- check_system_binaries: pdftotext + Xvfb -------------------------------


def test_system_binaries_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostics.shutil,
        "which",
        lambda name: "/usr/bin/" + name if name in ("pdftotext", "Xvfb") else None,
    )
    res = diagnostics.check_system_binaries()
    assert res.pdftotext_available
    assert res.xvfb_available
    assert res.pdftotext_path == "/usr/bin/pdftotext"
    assert res.xvfb_path == "/usr/bin/Xvfb"


def test_system_binaries_xvfb_run_also_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostics.shutil,
        "which",
        lambda name: "/usr/bin/xvfb-run" if name == "xvfb-run" else None,
    )
    res = diagnostics.check_system_binaries()
    assert not res.pdftotext_available
    assert res.xvfb_available
    assert res.xvfb_path == "/usr/bin/xvfb-run"


# --- check_pdf_rendering: typst package + built-in template ------------------


def test_check_pdf_rendering_success(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "typst":
            mod = MagicMock()

            def fake_compile(source: str, output: str, format: str | None = None) -> None:
                Path(output).write_bytes(b"%PDF-1.4 fake")

            mod.compile = fake_compile
            return mod
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = diagnostics.check_pdf_rendering()
    assert res.ok
    assert "works" in res.message


def test_check_pdf_rendering_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "typst":
            raise ImportError("No module named 'typst'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = diagnostics.check_pdf_rendering()
    assert not res.ok
    assert "typst package not installed" in res.message


# --- check_config: file presence, parseability, credentials ------------------


def test_config_check_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(tmp_path / "nonexistent.toml"))
    settings = AppSettings()
    res = diagnostics.check_config(settings)
    assert not res.config_file_found
    assert not res.config_file_parseable


def test_config_check_parse_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # AppSettings itself will reject a malformed TOML file, so we construct settings
    # against a valid file and then point check_config at the broken one.
    valid = tmp_path / "valid.toml"
    valid.write_text("")
    bad = tmp_path / "bad.toml"
    bad.write_text('[target\nlinkedin_email = "x"')
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(valid))
    settings = AppSettings()
    monkeypatch.setattr(diagnostics, "_config_file_path", lambda: bad)
    res = diagnostics.check_config(settings)
    assert res.config_file_found
    assert not res.config_file_parseable
    assert res.error


def test_config_check_plaintext_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "cred.toml"
    cfg.write_text('[target]\nlinkedin_email = "a@b.com"\nlinkedin_password = "secret"\n')
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    settings = AppSettings()
    res = diagnostics.check_config(settings)
    assert res.config_file_found
    assert res.config_file_parseable
    assert res.plaintext_credentials


def test_config_check_resume_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "cfg.toml"
    resume = tmp_path / "resume.pdf"
    resume.write_text("fake pdf")
    cfg.write_text(f'resume_path = "{resume}"\noutput_dir = "{tmp_path / "out"}"\n')
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    settings = AppSettings()
    res = diagnostics.check_config(settings)
    assert res.resume_path_set
    assert res.resume_path_exists
    assert res.output_dir_writable


def test_config_check_does_not_create_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check_config is a diagnostic — it must NOT create the output dir (no mkdir
    side-effect); writability is reported via the nearest existing ancestor."""
    cfg = tmp_path / "cfg.toml"
    out = tmp_path / "nested" / "out"
    cfg.write_text(f'output_dir = "{out}"\n')
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", str(cfg))
    settings = AppSettings()
    res = diagnostics.check_config(settings)
    assert not out.exists()  # the diagnostic created nothing
    assert res.output_dir_writable  # the existing ancestor (tmp_path) is writable


# --- run_diagnostics wires every new check ---------------------------------


async def test_run_diagnostics_includes_browser_system_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch, resp=_FakeResponse(200, _models_payload("m")))
    monkeypatch.setenv("JOB_APPLICATOR_CONFIG_FILE", "nonexistent-config-unit-test.toml")
    report = await diagnostics.run_diagnostics(AppSettings(llm=LLMConfig(model="m")))
    assert report.ok
    assert isinstance(report.browser, BrowserCheck)
    assert isinstance(report.system, SystemBinariesCheck)
    assert isinstance(report.config, ConfigCheck)
