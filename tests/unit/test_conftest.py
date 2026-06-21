"""Tests for shared test configuration helpers."""

from __future__ import annotations

from unittest.mock import patch

from tests.conftest import _vllm_endpoint_reachable


def test_vllm_endpoint_reachable_detects_closed_port() -> None:
    with patch("socket.create_connection", side_effect=OSError("refused")):
        assert not _vllm_endpoint_reachable()


def test_vllm_endpoint_reachable_detects_open_port() -> None:
    class _DummyConn:
        def __enter__(self) -> "_DummyConn":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    with patch("socket.create_connection", return_value=_DummyConn()):
        assert _vllm_endpoint_reachable()
