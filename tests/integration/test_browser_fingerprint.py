"""Offline browser-fingerprint self-consistency gate (scraper anti-detection audit follow-up).

Launches the REAL browser stack (``BrowserManager``, default ``channel="chrome"``) against a
LOOPBACK server (``http://127.0.0.1`` — a secure/trustworthy context, so ``navigator.userAgentData``
+ client-hints populate) that captures the request headers, then asserts the presented fingerprint
is self-consistent and doesn't leak automation/headless. **Zero external/LinkedIn traffic.**

Each assertion pins one finding from the audit's risk register: R2 (HeadlessChrome in Sec-CH-UA),
R3 (UA vs client-hints version skew), R4 (a cross-OS WebGL renderer under a Linux UA), and the
``navigator.webdriver`` / platform-agreement invariants. Skipped when no host Chrome is installed —
the guarantees are for the ``channel="chrome"`` path; the bundled-Chromium fallback is weaker by
design (it can't avoid the HeadlessChrome client-hint in headless mode).
"""

from __future__ import annotations

import asyncio
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, ClassVar

import pytest

from job_applicator.browser.manager import BrowserManager
from job_applicator.config import BrowserConfig
from job_applicator.utils.region import host_chrome_path

pytestmark = pytest.mark.skipif(
    host_chrome_path() is None,
    reason="channel='chrome' fingerprint gate needs a host Chrome/Chromium installed",
)

_FP_JS = """() => {
  let webgl = '';
  try {
    const gl = document.createElement('canvas').getContext('webgl');
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    webgl = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : '';
  } catch (_) { /* headless without GL — leave blank */ }
  const uaData = navigator.userAgentData;
  return {
    ua: navigator.userAgent,
    webdriver: navigator.webdriver,
    platform: navigator.platform,
    uaDataPlatform: uaData ? uaData.platform : null,
    webgl: webgl,
  };
}"""


class _CapturingHandler(BaseHTTPRequestHandler):
    last_headers: ClassVar[dict[str, str]] = {}

    def do_GET(self) -> None:
        _CapturingHandler.last_headers = {k.lower(): v for k, v in self.headers.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body>fingerprint probe</body></html>")

    def log_message(self, *args: object) -> None:  # silence the server
        return


async def _launch_and_probe(url: str) -> dict[str, Any]:
    async with BrowserManager(BrowserConfig(headless=True)) as manager:  # channel="chrome" default
        async with manager.new_page() as page:
            await page.goto(url, wait_until="load")
            fp: dict[str, Any] = await page.evaluate(_FP_JS)
    return fp


@pytest.fixture
def fingerprint() -> tuple[dict[str, Any], dict[str, str]]:
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        fp = asyncio.run(_launch_and_probe(f"http://127.0.0.1:{port}/"))
    finally:
        server.shutdown()
    return fp, _CapturingHandler.last_headers


def _major(text: str) -> str | None:
    m = re.search(r"Chrome/(\d{2,})\.", text) or re.search(
        r'"(?:Google Chrome|Chromium)";v="(\d{2,})"', text
    )
    return m.group(1) if m else None


def test_no_headless_chrome_leak(fingerprint: tuple[dict[str, Any], dict[str, str]]) -> None:
    """R2: `HeadlessChrome` must not appear in the UA or the Sec-CH-UA client-hint header — the
    strongest fingerprint tell, sent on every request and invisible to JS-level stealth."""
    fp, headers = fingerprint
    assert "HeadlessChrome" not in fp["ua"]
    assert "HeadlessChrome" not in headers.get("sec-ch-ua", "")
    assert "HeadlessChrome" not in headers.get("sec-ch-ua-full-version-list", "")


def test_ua_and_client_hint_versions_agree(
    fingerprint: tuple[dict[str, Any], dict[str, str]],
) -> None:
    """R3: the UA major version must equal the Sec-CH-UA major (channel='chrome' → both the real
    engine's version; the old host-UA-over-bundled-engine skew is the regression this guards)."""
    fp, headers = fingerprint
    ua_major = _major(fp["ua"])
    hint_major = _major(headers.get("sec-ch-ua", ""))
    assert ua_major is not None and hint_major is not None
    assert ua_major == hint_major, f"UA {ua_major} != Sec-CH-UA {hint_major}"


def test_webdriver_flag_absent(fingerprint: tuple[dict[str, Any], dict[str, str]]) -> None:
    fp, _ = fingerprint
    assert fp["webdriver"] in (False, None)


def test_platform_is_self_consistent(fingerprint: tuple[dict[str, Any], dict[str, str]]) -> None:
    """navigator.platform / Sec-CH-UA-Platform / userAgentData.platform must all agree with the UA
    OS token — the alignment `_aligned_stealth` exists to enforce."""
    fp, headers = fingerprint
    assert "Linux" in fp["ua"]  # this dev/CI host
    assert "Linux" in fp["platform"]
    assert "Linux" in headers.get("sec-ch-ua-platform", "")
    if fp["uaDataPlatform"] is not None:
        assert fp["uaDataPlatform"] == "Linux"


def test_webgl_renderer_not_cross_os(fingerprint: tuple[dict[str, Any], dict[str, str]]) -> None:
    """R4: the WebGL renderer must not be an OS-foreign string under a Linux UA (the playwright-
    stealth default `Intel Iris OpenGL Engine` is a macOS renderer — an impossible combination and
    a blocklisted signature). SwiftShader / ANGLE / Mesa are all Linux-plausible and allowed."""
    fp, _ = fingerprint
    webgl = fp["webgl"] or ""
    for foreign in ("Iris OpenGL Engine", "Apple", "Metal", "Direct3D", "D3D11"):
        assert foreign not in webgl, f"cross-OS WebGL tell under a Linux UA: {webgl!r}"
