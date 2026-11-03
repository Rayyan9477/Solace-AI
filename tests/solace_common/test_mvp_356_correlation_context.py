"""tests/solace_common/test_mvp_356_correlation_context.py
MVP-356 (supporting; WS-11 closes it): correlation ids live in one request-scoped context that WS-04.3 sets."""
from __future__ import annotations

import asyncio

import pytest

from solace_common.request_context import clear_correlation_ids, get_correlation_headers, set_correlation_ids


def test_headers_reflect_only_the_ids_that_are_set():
    clear_correlation_ids()
    assert get_correlation_headers() == {}
    set_correlation_ids(request_id="r-1", correlation_id="c-1", turn_id="t-1")
    assert get_correlation_headers() == {"X-Request-ID": "r-1", "X-Correlation-ID": "c-1", "X-Turn-ID": "t-1"}
    clear_correlation_ids()
    assert get_correlation_headers() == {}


def test_empty_values_are_omitted():
    clear_correlation_ids()
    set_correlation_ids(request_id="r-2", correlation_id="", turn_id="")
    assert get_correlation_headers() == {"X-Request-ID": "r-2"}
    clear_correlation_ids()


@pytest.mark.asyncio
async def test_ids_are_isolated_between_concurrent_requests():
    clear_correlation_ids()
    seen = {}

    async def request(label):
        set_correlation_ids(request_id=f"r-{label}", correlation_id=f"c-{label}", turn_id=f"t-{label}")
        await asyncio.sleep(0.01)
        seen[label] = get_correlation_headers()["X-Request-ID"]

    await asyncio.gather(asyncio.create_task(request("a")), asyncio.create_task(request("b")))
    assert seen == {"a": "r-a", "b": "r-b"}
