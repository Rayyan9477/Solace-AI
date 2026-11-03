"""MVP-180 / WS-09.F1: the PHI log sanitizer never recurses into the logging chain.

``PHIDetector.detect`` used to log its ``phi_detected`` counter through structlog from inside
the chain. ``merge_contextvars`` put the bound request context back into that line, so a bound
value that looks like PHI (a 9-digit id, ``a1234567``, about 1 in 61 uuid4 strings) sent every
log call of the request into ``RecursionError``, the crisis path included.

All values below are synthetic.
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import logging
import random
import re
import sys
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import structlog
from pydantic import ValidationError

from solace_infrastructure.runtime import ServiceRuntimeSpec
from solace_infrastructure.runtime.logging import configure_service_logging
from solace_security import phi_protection
from solace_security.phi_protection import (
    PHIDetector,
    PHIMasker,
    PHISettings,
    PHIType,
    mask_phi,
    phi_sanitizer_processor,
)

# Synthetic ids the sanitizer flags: "SSN without dashes" and "driver's licence".
_PHI_LOOKING_IDS = {"request_id": "123456789", "user_id": "a1234567"}
_MASKED_IDS = {"request_id": "***-**-6789", "user_id": "********"}
_CANARY = "jane.doe@example.com 555-867-5309"
_CANARY_MASKED = "j*******@example.com ******5309"
_COUNTER = re.compile(r"^phi_detected count=\d+ types=[a-z_]+(,[a-z_]+)*$")

# Loggers that configure_service_logging (or a test) changes; restored after every test.
_TOUCHED_LOGGERS = (
    "",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "aiokafka",
    "asyncpg",
    "httpx",
    "httpcore",
    "solace_security.phi_protection",
)


def _spec(env: str = "production", **overrides: Any) -> ServiceRuntimeSpec:
    return ServiceRuntimeSpec(service_name="safety-service", environment=env, **overrides)


def _frame_depth() -> int:
    depth, frame = 0, sys._getframe()
    while frame is not None:
        depth, frame = depth + 1, frame.f_back
    return depth


@contextlib.contextmanager
def _recursion_guard(headroom: int = 300) -> Iterator[None]:
    """Allow ``headroom`` frames beyond the caller, so a recursing log call fails fast."""
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(_frame_depth() + headroom)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


def _json_lines(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _assert_no_raw_ids(text: str) -> None:
    for name, raw in _PHI_LOOKING_IDS.items():
        assert raw not in text, f"unmasked {name} in log output"


def _uuid4_ids(count: int, seed: int) -> list[str]:
    """uuid4 strings from a seeded generator, so the flagged share is the same on every run."""
    rng = random.Random(seed)  # noqa: S311 - reproducible test ids, not a secret
    return [str(uuid.UUID(int=rng.getrandbits(128), version=4)) for _ in range(count)]


@pytest.fixture(autouse=True)
def _isolated_logging() -> Iterator[None]:
    """Start with no bound context; put structlog and stdlib logging back afterwards."""
    saved_structlog = structlog.get_config()
    saved_loggers = {}
    for name in _TOUCHED_LOGGERS:
        lg = logging.getLogger(name)
        saved_loggers[name] = (list(lg.handlers), lg.level, lg.propagate, lg.disabled)
    structlog.contextvars.clear_contextvars()
    try:
        yield
    finally:
        structlog.contextvars.clear_contextvars()
        structlog.configure(**saved_structlog)
        for name, (handlers, level, propagate, disabled) in saved_loggers.items():
            lg = logging.getLogger(name)
            for handler in list(lg.handlers):
                if handler not in handlers:
                    lg.removeHandler(handler)
                    handler.close()
            for handler in handlers:
                if handler not in lg.handlers:
                    lg.addHandler(handler)
            lg.setLevel(level)
            lg.propagate = propagate
            lg.disabled = disabled


def test_bound_phi_looking_context_is_logged_once_masked_and_never_recurses(capsys):
    configure_service_logging(_spec())
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)

    with _recursion_guard():
        structlog.get_logger("t").info("hello")

    captured = capsys.readouterr()
    lines = _json_lines(captured.out)
    assert len(lines) == 1
    assert lines[0]["event"] == "hello"
    assert {k: lines[0][k] for k in _MASKED_IDS} == _MASKED_IDS
    _assert_no_raw_ids(captured.out + captured.err)


def test_two_hundred_uuid4_request_ids_all_log(capsys):
    ids = _uuid4_ids(200, seed=0)
    quiet = PHIDetector(PHISettings(log_detections=False))
    expected = [quiet.detect(request_id).masked_text for request_id in ids]
    assert any(m != r for m, r in zip(expected, ids, strict=True)), "the seed must give a flagged uuid4"
    configure_service_logging(_spec())
    capsys.readouterr()
    log = structlog.get_logger("t")

    with _recursion_guard():
        for request_id in ids:
            structlog.contextvars.bind_contextvars(request_id=request_id)
            log.info("hello")

    lines = _json_lines(capsys.readouterr().out)
    assert len(lines) == 200
    assert all(line["event"] == "hello" for line in lines)
    assert [line["request_id"] for line in lines] == expected


def test_detect_called_by_service_code_logs_its_counter_once(capsys):
    """Outside the chain the counter is still a structlog line, and that line is sanitized once."""
    configure_service_logging(_spec())
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)

    with _recursion_guard():
        masked = mask_phi("call back on 555-867-5309")

    assert masked == "call back on ******5309"
    captured = capsys.readouterr()
    lines = _json_lines(captured.out)
    assert [(line["event"], line["count"], line["types"]) for line in lines] == [("phi_detected", 1, ["phone"])]
    assert {k: lines[0][k] for k in _MASKED_IDS} == _MASKED_IDS
    _assert_no_raw_ids(captured.out + captured.err)
    assert "555-867-5309" not in captured.out + captured.err


@pytest.mark.parametrize(
    ("log_level", "expected_events"),
    [
        ("INFO", ["escalation for ***-**-6789"]),
        # At DEBUG the counter is written by the same root handler, inside the outer format.
        ("DEBUG", ["phi_detected count=1 types=ssn", "escalation for ***-**-6789"]),
    ],
)
def test_stdlib_record_with_phi_is_emitted_once_while_context_is_bound(capsys, log_level, expected_events):
    configure_service_logging(_spec(log_level=log_level))
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)

    with _recursion_guard():
        logging.getLogger("svc.crisis").error("escalation for %s", "123456789")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert [line["event"] for line in _json_lines(captured.out)] == expected_events
    _assert_no_raw_ids(captured.out)


def test_counter_inside_the_chain_is_stdlib_debug_with_counts_and_types_only(capsys):
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    configure_service_logging(_spec(log_level="DEBUG"))
    phi_logger = logging.getLogger("solace_security.phi_protection")
    phi_logger.addHandler(_Collect(level=logging.DEBUG))
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)

    with _recursion_guard():
        structlog.get_logger("t").info("hello")

    assert {(r.levelno, r.getMessage()) for r in records} == {
        (logging.DEBUG, "phi_detected count=1 types=ssn"),
        (logging.DEBUG, "phi_detected count=1 types=license_number"),
    }
    assert all(_COUNTER.match(r.getMessage()) for r in records)
    # At DEBUG the counter lines also reach stdout through the root handler. They carry no
    # bound context, so nothing unmasked gets out, and the triggering line is still emitted once.
    captured = capsys.readouterr()
    events = [line["event"] for line in _json_lines(captured.out)]
    assert events.count("hello") == 1
    assert sorted(e for e in events if e != "hello") == [
        "phi_detected count=1 types=license_number",
        "phi_detected count=1 types=ssn",
    ]
    _assert_no_raw_ids(captured.out + captured.err)


def test_sanitizer_canary_is_masked_exactly_once_with_phi_looking_context_bound(capsys):
    configure_service_logging(_spec())
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)

    with _recursion_guard():
        out = phi_sanitizer_processor(None, "info", {"event": _CANARY})

    assert out == {"event": _CANARY_MASKED}
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_a_nested_call_while_sanitizing_is_masked_and_writes_nothing(capsys):
    """Fix round 2: a nested event is masked by a detector that never logs, not passed through."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    configure_service_logging(_spec(log_level="DEBUG"))
    logging.getLogger("solace_security.phi_protection").addHandler(_Collect(level=logging.DEBUG))
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)

    token = phi_protection._phi_sanitizing.set(True)
    try:
        with _recursion_guard():
            out = phi_sanitizer_processor(None, "info", {"event": _CANARY})
    finally:
        phi_protection._phi_sanitizing.reset(token)

    assert out == {"event": _CANARY_MASKED}
    assert records == []
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert phi_protection._phi_sanitizing.get() is False


def _log_once_from_inside_sanitize_dict(
    monkeypatch: pytest.MonkeyPatch, action: Any
) -> list[bool]:
    """Run ``action`` once, from inside the first ``sanitize_dict`` call (the guard is set)."""
    original = phi_protection.PHISanitizer.sanitize_dict
    fired: list[bool] = []

    def _wrapped(self: Any, data: dict[str, Any], sensitive_keys: Any = None) -> dict[str, Any]:
        if not fired:
            fired.append(phi_protection._phi_sanitizing.get())
            action()
        return original(self, data, sensitive_keys)

    monkeypatch.setattr(phi_protection.PHISanitizer, "sanitize_dict", _wrapped)
    return fired


def test_a_line_logged_from_inside_sanitize_dict_is_masked(capsys, monkeypatch):
    configure_service_logging(_spec())
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)
    fired = _log_once_from_inside_sanitize_dict(
        monkeypatch, lambda: structlog.get_logger("inner").info("inner_line", phone="555-867-5309")
    )

    with _recursion_guard():
        structlog.get_logger("t").info("hello")

    assert fired == [True]
    captured = capsys.readouterr()
    lines = _json_lines(captured.out)
    assert [line["event"] for line in lines] == ["inner_line", "hello"]
    assert lines[0]["phone"] == "******5309"
    assert all({k: line[k] for k in _MASKED_IDS} == _MASKED_IDS for line in lines)
    _assert_no_raw_ids(captured.out + captured.err)
    assert "555-867-5309" not in captured.out + captured.err


def test_a_finalizer_run_while_sanitizing_logs_masked(capsys, monkeypatch):
    class _Cycle:
        def __init__(self) -> None:
            self.me = self

        def __del__(self) -> None:
            structlog.get_logger("gc").info("finalizer_line", phone="555-867-5309")

    configure_service_logging(_spec())
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)
    fired = _log_once_from_inside_sanitize_dict(monkeypatch, gc.collect)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        _Cycle()  # unreachable at once; only the forced collection below can finalize it
        with _recursion_guard():
            structlog.get_logger("t").info("hello")
    finally:
        if was_enabled:
            gc.enable()

    assert fired == [True]
    captured = capsys.readouterr()
    lines = _json_lines(captured.out)
    assert sorted(line["event"] for line in lines) == ["finalizer_line", "hello"]
    assert captured.err == ""
    _assert_no_raw_ids(captured.out)
    assert "555-867-5309" not in captured.out


def test_a_callback_scheduled_while_sanitizing_logs_masked(capsys, monkeypatch):
    """call_soon copies the context, guard included, so the callback's lines are nested too."""
    configure_service_logging(_spec())
    capsys.readouterr()
    structlog.contextvars.bind_contextvars(**_PHI_LOOKING_IDS)
    loop = asyncio.new_event_loop()
    try:
        fired = _log_once_from_inside_sanitize_dict(
            monkeypatch,
            lambda: loop.call_soon(
                lambda: structlog.get_logger("cb").info("callback_line", phone="555-867-5309")
            ),
        )
        with _recursion_guard():
            structlog.get_logger("t").info("hello")
            loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()

    assert fired == [True]
    captured = capsys.readouterr()
    lines = _json_lines(captured.out)
    assert [line["event"] for line in lines] == ["hello", "callback_line"]
    _assert_no_raw_ids(captured.out + captured.err)
    assert "555-867-5309" not in captured.out + captured.err


def test_the_guard_is_reset_when_sanitizing_raises(monkeypatch):
    class _BoomError(Exception):
        pass

    def _explode(self: Any, data: dict[str, Any], sensitive_keys: Any = None) -> dict[str, Any]:
        raise _BoomError

    monkeypatch.setattr(phi_protection.PHISanitizer, "sanitize_dict", _explode)
    with pytest.raises(_BoomError):
        phi_sanitizer_processor(None, "info", {"event": _CANARY})
    monkeypatch.undo()

    assert phi_sanitizer_processor(None, "info", {"event": _CANARY}) == {"event": _CANARY_MASKED}


def test_concurrent_requests_never_share_the_guard(capsys):
    configure_service_logging(_spec())
    capsys.readouterr()

    async def _request(request_id: str) -> None:
        structlog.contextvars.bind_contextvars(request_id=request_id)
        log = structlog.get_logger("t")
        for _ in range(20):
            log.info("step")
            await asyncio.sleep(0)

    async def _main() -> None:
        await asyncio.gather(*(_request(r) for r in ("123456789", "a1234567", "probe-1")))

    with _recursion_guard():
        asyncio.run(_main())

    captured = capsys.readouterr()
    lines = _json_lines(captured.out)
    assert len(lines) == 60
    assert sorted({line["request_id"] for line in lines}) == ["********", "***-**-6789", "probe-1"]
    _assert_no_raw_ids(captured.out + captured.err)


# --- WS-09.F1 add-on: PHI_PARTIAL_MASK_VISIBLE=0 or below used to reveal the whole phone number,
# because digits[-0:] is the full string. The setting is now 0..4, and 0 masks every digit.


def _settings_from(source: str, monkeypatch: pytest.MonkeyPatch, value: int) -> PHISettings:
    if source == "env":
        monkeypatch.setenv("PHI_PARTIAL_MASK_VISIBLE", str(value))
        return PHISettings(log_detections=False)
    return PHISettings(partial_mask_visible=value, log_detections=False)


@pytest.mark.parametrize("source", ["kwarg", "env"])
def test_partial_mask_visible_zero_masks_every_phone_digit(monkeypatch, source):
    settings = _settings_from(source, monkeypatch, 0)
    assert settings.partial_mask_visible == 0

    assert PHIMasker(settings).mask("555-867-5309", PHIType.PHONE) == "*" * 10
    masked = PHIDetector(settings).detect("call back on 555-867-5309").masked_text
    assert masked == "call back on **********"
    assert not re.search(r"\d", masked)


def test_mask_phone_masks_every_digit_when_visible_is_below_zero():
    # Built without validation, so a negative value reaches the masker itself.
    settings = PHISettings.model_construct(partial_mask_visible=-1)
    assert PHIMasker(settings).mask("555-867-5309", PHIType.PHONE) == "*" * 10


@pytest.mark.parametrize("source", ["kwarg", "env"])
@pytest.mark.parametrize(("value", "error_type"), [(-1, "greater_than_equal"), (5, "less_than_equal")])
def test_partial_mask_visible_out_of_range_is_rejected_without_echoing_it(
    monkeypatch, source, value, error_type
):
    with pytest.raises(ValidationError) as exc:
        _settings_from(source, monkeypatch, value)

    errors = exc.value.errors(include_input=False)
    assert [(e["loc"], e["type"]) for e in errors] == [(("partial_mask_visible",), error_type)]
    echoed = re.compile(rf"(?<![\w.-]){re.escape(str(value))}(?![\w.])")
    for text in (str(exc.value), repr(exc.value)):
        assert "input_value" not in text
        assert not echoed.search(text), "the rejected value is echoed in the error text"


def test_partial_mask_visible_keeps_its_default_and_range_ends():
    assert PHISettings().partial_mask_visible == 4
    assert PHIMasker(PHISettings(partial_mask_visible=4)).mask("555-867-5309", PHIType.PHONE) == "******5309"
    assert PHIMasker(PHISettings(partial_mask_visible=1)).mask("555-867-5309", PHIType.PHONE) == "*********9"


# --- WS-09.F1 fix round 2: the e-mail pattern was quadratic (64 KB of "a." took about 17 s,
# synchronously on the event loop), and its [A-Z|a-z] class also matched a literal "|".

_DOTTED_64KB = "a." * 32768  # 64 KB with no address in it


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_DOTTED_64KB, _DOTTED_64KB),
        (_DOTTED_64KB + " jane.doe@example.com", _DOTTED_64KB + " j*******@example.com"),
    ],
    ids=["no-address", "address-at-the-end"],
)
def test_email_pattern_sanitizes_64kb_of_dotted_text_quickly(text, expected):
    started = time.perf_counter()
    out = phi_sanitizer_processor(None, "info", {"event": text})
    elapsed = time.perf_counter() - started

    assert out == {"event": expected}
    assert elapsed < 0.5, f"sanitizing 64 KB took {elapsed:.2f} s"


def test_email_pattern_still_masks_addresses():
    quiet = PHIDetector(PHISettings(log_detections=False))
    assert quiet.detect("contact jane.doe@example.com today").masked_text == (
        "contact j*******@example.com today"
    )
    long_local = "a" * 70
    masked = quiet.detect(f"to {long_local}@example.com").masked_text
    assert f"{long_local}@example.com" not in masked
    assert masked.endswith("*" * 63 + "@example.com")
    # "|" is not a top-level-domain character.
    assert quiet.detect("ops@host.a|b").contains_phi is False
