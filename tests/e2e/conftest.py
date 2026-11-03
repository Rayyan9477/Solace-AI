"""E2E tests against the compose test stack (IF-19 E2E part; MVP-039, WS-05.4).

* ``e2e_stack`` (session): ``require_docker()``, then ``start_stack(StackConfig.from_env())`` (hop ``stack_up``),
  and ``stop_stack`` at the end of the session (kept with ``SOLACE_E2E_KEEP=1``). See solace_testing.e2e.
* ``@pytest.mark.profile("dev")`` / ``("strict")`` / ``("dev", "strict")``: an item whose profiles do not include
  ``SOLACE_E2E_PROFILE`` (default dev) is deselected, not skipped, so ``SOLACE_REQUIRE_REALINFRA=1`` still means
  that every selected test ran. Unmarked items run in both profiles. Deselection runs after the expected-failure
  manifest has indexed every item (so a profile-specific entry is never reported stale).
* Every item here is marked ``e2e`` and ``integration``, runs its coroutine on the session event loop
  (pytest-asyncio ``loop_scope="session"``) and gets ``pytest.mark.timeout(900, func_only=True)`` unless it sets
  its own: the test function is bounded, its setup (the stack build, which has its own timeouts) is not.
"""
from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from solace_testing.e2e import Stack, StackConfig, require_docker, start_stack, stop_stack

_HERE = Path(__file__).resolve().parent
_PROFILES = ("dev", "strict")
_DEFAULT_TIMEOUT_S = 900
_SELECTOR = "solace-e2e-profile-selection"


def _is_here(item: pytest.Item) -> bool:
    path = Path(item.path).resolve()
    return path == _HERE or _HERE in path.parents


def _current_profile() -> str:
    profile = os.environ.get("SOLACE_E2E_PROFILE", "dev").strip()
    if profile not in _PROFILES:
        raise pytest.UsageError(f"SOLACE_E2E_PROFILE must be dev or strict, got {profile!r}")
    return profile


def _item_profiles(item: pytest.Item) -> set[str] | None:
    """The profiles an item's ``profile`` markers name, or None when it has none (both profiles)."""
    marks = list(item.iter_markers("profile"))
    if not marks:
        return None
    names = {name for mark in marks for name in mark.args}
    if not names or not names <= set(_PROFILES) or any(mark.kwargs for mark in marks):
        raise pytest.UsageError(f"{item.nodeid}: use pytest.mark.profile('dev'), ('strict') or ('dev', 'strict')")
    return names


class _ProfileSelection:
    """Deselects the items of the other profile (trylast: after the manifest plugin saw every item)."""

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        profile = _current_profile()
        keep, drop = [], []
        for item in items:
            names = _item_profiles(item) if _is_here(item) else None
            (drop if names is not None and profile not in names else keep).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep


def pytest_configure(config: pytest.Config) -> None:
    if not config.pluginmanager.has_plugin(_SELECTOR):
        config.pluginmanager.register(_ProfileSelection(), _SELECTOR)


@pytest.hookimpl(tryfirst=True)  # before -m selection reads the markers
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_timeout = config.pluginmanager.hasplugin("timeout")
    for item in items:
        if not _is_here(item):
            continue
        item.add_marker(pytest.mark.e2e)
        item.add_marker(pytest.mark.integration)
        if has_timeout and item.get_closest_marker("timeout") is None:
            # func_only: the test function only. The session fixture builds the stack in the first test's setup
            # and is bounded by start_stack's own timeouts (a cold `up --build` may take up to UP_TIMEOUT_S).
            item.add_marker(pytest.mark.timeout(_DEFAULT_TIMEOUT_S, func_only=True))
        if inspect.iscoroutinefunction(getattr(item, "obj", None)):
            marker = item.get_closest_marker("asyncio")
            if marker is None or not (marker.kwargs.get("loop_scope") or marker.kwargs.get("scope")):
                # Prepended, so it is the closest marker even after auto mode's bare asyncio marker.
                item.add_marker(pytest.mark.asyncio(loop_scope="session"), append=False)


@pytest.fixture(scope="session")
def e2e_stack() -> Iterator[Stack]:
    """The running compose stack of this session's profile and lane."""
    require_docker()
    stack = start_stack(StackConfig.from_env())
    try:
        yield stack
    finally:
        stop_stack(stack)
