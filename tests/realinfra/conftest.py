"""Real-infra component tests (IF-19): one directory per workstream, ``tests/realinfra/<ws-slug>/``.

* Re-exports the five session fixtures of :mod:`solace_testing.realinfra`: ``pg_migrated``, ``redis_url``,
  ``kafka_bootstrap``, ``smtp_sink`` and ``fake_llm``.
* Marks every item under this directory ``realinfra`` and ``integration``.
* Runs coroutine tests on the session event loop (``asyncio(loop_scope="session")``, pytest-asyncio 1.3), the
  loop the async session fixtures and their pools are bound to. A test that sets its own loop scope keeps it.
* Requires Docker for the whole directory: skipped without it, failed when SOLACE_REQUIRE_REALINFRA is set.
* Isolates every test (``_realinfra_isolate``, autouse here): the mail sink, the fake LLM and Redis start clean,
  and a test that leaves the kafka broker paused fails.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from solace_testing.realinfra import (  # noqa: F401 - fixtures are found by name
    _realinfra_isolate,
    fake_llm,
    kafka_bootstrap,
    pg_migrated,
    redis_url,
    require_docker,
    smtp_sink,
)

_HERE = Path(__file__).resolve().parent


def _is_here(item: pytest.Item) -> bool:
    path = Path(item.path).resolve()
    return path == _HERE or _HERE in path.parents


@pytest.hookimpl(tryfirst=True)  # before -m selection reads the markers
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if not _is_here(item):
            continue
        item.add_marker(pytest.mark.realinfra)
        item.add_marker(pytest.mark.integration)
        if inspect.iscoroutinefunction(getattr(item, "obj", None)):
            marker = item.get_closest_marker("asyncio")
            if marker is None or not (marker.kwargs.get("loop_scope") or marker.kwargs.get("scope")):
                # Prepended, so it is the closest marker even after auto mode's bare asyncio marker.
                item.add_marker(pytest.mark.asyncio(loop_scope="session"), append=False)


@pytest.fixture(scope="session", autouse=True)
def _realinfra_requires_docker() -> None:
    require_docker()
