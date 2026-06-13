"""Tests for retry utility."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from job_applicator.utils.retry import async_retry


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        mock = AsyncMock(return_value="ok")

        @async_retry(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        async def func():
            return await mock()

        result = await func()
        assert result == "ok"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        mock = AsyncMock(side_effect=[ValueError("fail"), "ok"])

        @async_retry(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        async def func():
            return await mock()

        result = await func()
        assert result == "ok"
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        mock = AsyncMock(side_effect=ValueError("fail"))

        @async_retry(max_attempts=2, base_delay=0.01, exceptions=(ValueError,))
        async def func():
            return await mock()

        with pytest.raises(ValueError):
            await func()
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_other_exceptions(self):
        mock = AsyncMock(side_effect=TypeError("wrong type"))

        @async_retry(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        async def func():
            return await mock()

        with pytest.raises(TypeError):
            await func()
        assert mock.call_count == 1
