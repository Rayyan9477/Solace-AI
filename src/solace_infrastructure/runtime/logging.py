"""PHI-safe structured logging for every service (IF-01, MVP-180).

``configure_service_logging(spec)`` installs one processor chain for structlog and a matching
one for stdlib ``logging`` records (uvicorn, aiokafka, asyncpg, Python warnings and the rest).
Both render one JSON object per line on stdout:

    merge_contextvars, add_log_level, TimeStamper, StackInfoRenderer, format_exc_info,
    redact_credentials, phi_sanitizer_processor, JSONRenderer

``format_exc_info`` turns ``exc_info`` into the ``exception`` traceback text before the
redaction and PHI processors run. So ``logger.exception(...)`` logs the traceback, and both
processors see that text (MVP-180). The runtime's ``format_exc_info`` step never renders an
exception that carries validation input: when a pydantic ``ValidationError`` or one of FastAPI's
``ValidationException`` family (request, response, WebSocket) is the exception or is linked to
it (cause, context, ExceptionGroup member), the line gets the error types, masked field names
and raise site instead, because that message quotes every input value and the regex PHI
sanitizer cannot recognise names or free text (WS-04.F1).

``redact_credentials`` first turns every value into JSON-native data: tuples and sets become
lists, an exception linked to a validation error becomes its error type and validation errors,
any other object becomes the text JSONRenderer would print for it, dict keys become strings
and a PHI-shaped key becomes ``[key]``. Any string that quotes validation input (the event text
included, for example asyncio's "Task exception was never retrieved" with the task's repr) is
cut to its first line, or to a fixed text. So both scrubbers see everything that will be
rendered. It runs in every environment. In strict environments the PHI sanitizer is mandatory
and must pass a self-test; otherwise ``StartupDependencyError("logging", ...)`` is raised and
nothing is reconfigured. Outside strict environments a missing sanitizer is left out and one
warning says so.

``warnings.warn`` output is captured (``logging.captureWarnings``) and goes through the same
chain on the ``py.warnings`` logger instead of raw text on stderr; a warning that quotes pydantic
input (``input_value=``) becomes a fixed event naming only its category.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sys
import traceback
from typing import Any

import structlog
from structlog.typing import EventDict, Processor

from solace_infrastructure.runtime.spec import ServiceRuntimeSpec, StartupDependencyError
from solace_security.production_guards import is_strict_environment

# Python warnings are logged here once logging.captureWarnings(True) is in effect.
WARNINGS_LOGGER = "py.warnings"

# Library loggers whose records go through the runtime handler on the root logger.
ROUTED_STDLIB_LOGGERS: tuple[str, ...] = (
    "uvicorn",
    "uvicorn.error",
    "aiokafka",
    "asyncpg",
    WARNINGS_LOGGER,
)

# uvicorn.error logs WebSocket handshakes ('"WebSocket /path?token=..." [accepted]' and the
# 403 lines) at INFO, query string included. It is kept at WARNING or above, so routing it
# through this chain never brings those lines back after ``--log-level warning`` (R-01.2).
UVICORN_ERROR_MIN_LEVEL: int = logging.WARNING

# httpx logs "HTTP Request: GET <full URL with query string>" at INFO, and httpcore logs
# request internals at DEBUG. Both are kept at WARNING or above in every process, including
# workers and CLIs that never build an app.
QUIETED_STDLIB_LOGGERS: tuple[str, ...] = ("httpx", "httpcore")
OUTBOUND_HTTP_MIN_LEVEL: int = logging.WARNING

# The runtime writes its own access line per request. uvicorn's access log carries raw query
# strings, so it is disabled.
SILENCED_STDLIB_LOGGERS: tuple[str, ...] = ("uvicorn.access",)

# uvicorn logs "Exception in ASGI application" with exc_info for every exception that escapes
# the app (Starlette re-raises after its 500 handler). str(exc) can carry request data, for
# example a pydantic ValidationError's input_value holding patient free text that the regex
# sanitizer cannot mask. A filter on these loggers rewrites that record to a fixed event that
# carries only the exception type; the record is kept, so the fault stays visible.
# A lifespan failure reaches uvicorn.error in one of two forms:
# - "Exception in 'lifespan' protocol" with exc_info (a plain ASGI lifespan app);
# - Starlette's traceback.format_exc() text, sent as lifespan.startup.failed (or
#   shutdown.failed) and logged by uvicorn as the record's message, without exc_info. This is
#   the form every FastAPI service produces.
# When the failure involves a pydantic ValidationError (for example settings read from the
# environment), its input_value can be a secret, so the record becomes a fixed event listing
# error types and field names only. Any other lifespan failure keeps its traceback for
# diagnosis; credential redaction and the PHI sanitizer still run on it.
ASGI_EXCEPTION_LOGGERS: tuple[str, ...] = ("uvicorn", "uvicorn.error")
# These prefixes are uvicorn's and Starlette's own wording. A uvicorn upgrade must re-run
# WS-04.3's probe (D:\tmp\mvp-exec\ws04\ws043_uvicorn_probe.py) to confirm they still match.
_ASGI_EXCEPTION_PREFIX = "Exception in ASGI application"
_LIFESPAN_EXCEPTION_PREFIX = "Exception in 'lifespan' protocol"
_LIFESPAN_FAILURE_TEXT_PREFIX = "Traceback (most recent call last):"
_ASGI_EXCEPTION_EVENT = "asgi_unhandled_exception"
_LIFESPAN_VALIDATION_EVENT = "lifespan_startup_failed"
# pydantic v2 renders each error's input as "input_value=..."; text carrying it is dropped.
_PYDANTIC_INPUT_MARKER = "input_value="
# Every rendering of a validation error's input: pydantic's input_value=, and an error dict's
# 'input' key as FastAPI's ValidationException family and errors() print it, in Python or JSON
# quotes. The quotes may arrive escaped, once or more: repr() of text that holds both quote
# kinds turns 'input': into \'input\':, and JSON re-encoding turns "input": into \"input\":
# (or a " escape). Text that carries one of these never reaches a log line as is
# (WS-04.F1 fix rounds 1 and 2).
# The backslash runs are bounded (16 covers four levels of re-encoding): an unbounded run made the
# search quadratic, so one long body of backslashes could block a service's event loop while its
# log line was checked (WS04F1F2-PHI-1).
_QUOTE = r"(?:\\{0,16}['\"]|\\{1,16}u00(?:22|27))"
_VALIDATION_INPUT_PATTERN = re.compile(
    re.escape(_PYDANTIC_INPUT_MARKER) + "|" + _QUOTE + "input" + _QUOTE + r"\s*:"
)
# What a string that quotes validation input in its first line is replaced with.
_WITHHELD_TEXT = "[withheld: text quoting validation input]"
# The keys of a pydantic or FastAPI errors() dict that hold the input or can quote it. A dict
# with "type", "loc" and any of these is logged as {"type", "field"} only (fix round 2).
_ERROR_DICT_DETAIL_KEYS = frozenset({"input", "ctx", "url", "msg"})
# The class name on an exception line of traceback text, e.g. "pkg.mod.ValidationError: ...".
_EXCEPTION_LINE = re.compile(r"(?m)^(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*(?:Error|Exception))(?::|$)")
# How many linked exceptions (cause, context, group members) are searched for a ValidationError.
_MAX_LINKED_EXCEPTIONS = 32

# A captured warning whose text quotes pydantic input becomes this event, which carries only
# ``warning_category``. warnings.formatwarning writes "<file>:<line>: <Category>: <message>".
_WARNING_REDACTED_EVENT = "python_warning_redacted"
_WARNING_CATEGORY = re.compile(r":\d+: ([A-Za-z_][\w.]*): ")
_UNKNOWN_WARNING_CATEGORY = "Warning"

# Validation error summaries (the 500 log line, the background-task and lifespan lines): a
# location part that is not an identifier (a dict key the sender chose, which could be an
# e-mail address) is replaced, and so is the key an extra_forbidden error names, which can be
# any text even when it is shaped like an identifier. The 422 body uses the same rule.
_FIELD_PART_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]{0,63}")
_MASKED_FIELD_PART = "[key]"
_MAX_FIELD_LENGTH = 200
_MAX_VALIDATION_ERRORS = 50

# A dict key in a log event that matches one of the PHI log sanitizer's patterns is replaced
# with this. The patterns are set by configure_service_logging; without a sanitizer, none.
_MASKED_KEY = "[key]"
_phi_key_patterns: tuple[re.Pattern[str], ...] = ()
# Values JSONRenderer serialises itself; anything else it would print with repr().
_JSON_NATIVE = (str, int, float, dict, list, tuple, set, frozenset)

_HANDLER_NAME = "solace-runtime-stdout"
_DEFAULT_LEVEL = logging.INFO

# Credential shapes rewritten in every string value, in this order, before the PHI sanitizer:
# URL userinfo (DSNs), Authorization schemes, credential query parameters, password pairs.
# The URL scheme is bounded to 32 characters; an unbounded scheme makes the first pattern
# quadratic on long scheme-like text (64 KB took seconds).
_CREDENTIAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]{0,31}://)[^\s/@:]*:[^\s/@]*@"), r"\1[REDACTED]@"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 [REDACTED]"),
    (
        re.compile(
            r"(?i)([?&;](?:token|access_token|refresh_token|id_token|api_key|apikey|code"
            r"|password|secret|sig|signature)=)[^&\s\"'#]+"
        ),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)\b(password|passwd|pwd)\s*[=:]\s*\S+"), r"\1=[REDACTED]"),
)

# Synthetic values for the strict self-test of the PHI sanitizer. They are not real PHI.
_CANARY_EMAIL = "jane.doe@example.com"
_CANARY_PHONE = "555-867-5309"
_CANARY_PHONE_DIGITS = "5558675309"
# A masked phone keeps at most its last four digits. PHI_PARTIAL_MASK_VISIBLE=0 masks every
# digit and values outside 0..4 are rejected by PHISettings. Any five consecutive canary digits
# that survive therefore mean the number is not masked: defence in depth against settings that
# skip validation (model_construct) and show five or more digits.
_CANARY_DIGIT_WINDOW = 5
_NON_DIGITS = re.compile(r"\D")


class _CurrentStdout:
    """Stream for the root handler that writes to whatever ``sys.stdout`` is at write time.

    The output is the same as ``StreamHandler(sys.stdout)``. Looking up ``sys.stdout`` on
    every write means a replaced stdout (test capture, a re-opened stream) never leaves the
    root logger writing to a stale or closed file.
    """

    def write(self, text: str) -> int:
        return sys.stdout.write(text)

    def flush(self) -> None:
        sys.stdout.flush()


class _RuntimeHandler(logging.StreamHandler[_CurrentStdout]):
    """The root handler. A record that cannot be formatted becomes one fixed JSON line.

    The stock ``Handler.handleError`` prints the raw message and arguments to stderr, PHI
    included. This one writes only the logger name, the level and the error type, and never
    writes ``record.msg`` or ``record.args``.
    """

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802 - stdlib name
        with contextlib.suppress(Exception):
            line = json.dumps(
                {
                    "event": "log_record_dropped",
                    "logger": record.name,
                    "level": record.levelname.lower(),
                    "error_type": type(sys.exc_info()[1]).__name__,
                }
            )
            self.stream.write(line + self.terminator)
            self.flush()


class _AsgiExceptionFilter(logging.Filter):
    """Rewrite uvicorn's exception records that can carry request data or secrets.

    - "Exception in ASGI application" becomes ``asgi_unhandled_exception`` with
      ``error_type`` (the exception class name) only.
    - A lifespan failure ("Exception in 'lifespan' protocol" with exc_info, or Starlette's
      traceback text) that involves a pydantic ValidationError becomes
      ``lifespan_startup_failed`` with ``error_type`` and ``validation_errors`` (error type
      and field name per error, never the input). Traceback text that carries pydantic's
      ``input_value=`` rendering is rewritten the same way even when the exception object is
      not available. Other lifespan failures pass unchanged.

    A rewritten record keeps its level; its traceback, message and arguments are dropped
    before any handler sees them. The filter never raises and never drops a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if message.startswith(_ASGI_EXCEPTION_PREFIX):
            exc = _record_exception(record)
            _rewrite_exception_record(record, _ASGI_EXCEPTION_EVENT, _type_name(exc))
        elif message.startswith(_LIFESPAN_EXCEPTION_PREFIX):
            exc = _record_exception(record)
            validation = _find_validation_error(exc)
            if validation is not None:
                _rewrite_lifespan_record(record, _type_name(exc), validation)
        elif message.startswith(_LIFESPAN_FAILURE_TEXT_PREFIX):
            # uvicorn logs Starlette's text from inside Starlette's except block, so the
            # exception that failed the lifespan is still the active one.
            active = sys.exc_info()[1]
            validation = _find_validation_error(active)
            if validation is not None or _quotes_validation_input(message):
                name = _type_name(active) if active is not None else _last_exception_name(message)
                _rewrite_lifespan_record(record, name, validation)
        return True


def _record_exception(record: logging.LogRecord) -> BaseException | None:
    return record.exc_info[1] if isinstance(record.exc_info, tuple) else None


def _type_name(exc: BaseException | None) -> str:
    return type(exc).__name__ if exc is not None else "unknown"


def _last_exception_name(text: str) -> str:
    """The class name on the last exception line of traceback text, else ``unknown``."""
    names = _EXCEPTION_LINE.findall(text)
    return names[-1] if names else "unknown"


def _replace_record_message(record: logging.LogRecord, event: str) -> None:
    """Make ``event`` the record's whole message; drop its arguments, traceback and stack."""
    record.msg = event
    record.args = ()
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None


def _rewrite_exception_record(record: logging.LogRecord, event: str, error_type: str) -> None:
    record.__dict__["error_type"] = error_type
    _replace_record_message(record, event)


class _WarningFilter(logging.Filter):
    """Rewrite a captured warning that quotes pydantic input (``input_value=...``).

    Pydantic's serializer warnings put the offending value into the warning text verbatim, and
    the regex PHI sanitizer cannot recognise free text. Such a record becomes
    ``python_warning_redacted`` with ``warning_category`` (the warning class name) only; its
    level is kept. Any other warning passes unchanged (both scrubbers still run on it). The
    filter never raises and never drops a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if _quotes_validation_input(message):
            match = _WARNING_CATEGORY.search(message)
            record.__dict__["warning_category"] = (
                match.group(1) if match else _UNKNOWN_WARNING_CATEGORY
            )
            _replace_record_message(record, _WARNING_REDACTED_EVENT)
        return True


def _rewrite_lifespan_record(
    record: logging.LogRecord, error_type: str, validation: BaseException | None
) -> None:
    _rewrite_exception_record(record, _LIFESPAN_VALIDATION_EVENT, error_type)
    record.__dict__["validation_errors"] = (
        _validation_error_fields(validation) if validation is not None else []
    )


def _quotes_validation_input(text: str) -> bool:
    """Whether ``text`` carries a validation error's input (pydantic's or FastAPI's rendering).

    Escaped quotes count too (see ``_VALIDATION_INPUT_PATTERN``), so text that went through
    ``repr()`` or JSON encoding, once or more, is still recognised.
    """
    return _VALIDATION_INPUT_PATTERN.search(text) is not None


def _withhold_validation_input(text: str) -> str:
    """``text``, unless it quotes validation input: then its first line, or a fixed text.

    The first line of a pydantic or FastAPI validation message is its title ("1 validation error
    for Age", "2 validation errors:"), and asyncio's "Task exception was never retrieved" puts
    the task's repr on the lines after it, so the first line is kept when it is clean.
    """
    if not _quotes_validation_input(text):
        return text
    lines = text.splitlines()
    first = lines[0] if lines else ""
    if first.strip() and not _quotes_validation_input(first):
        return first
    return _WITHHELD_TEXT


def _find_validation_error(exc: BaseException | None) -> BaseException | None:
    """The first validation error in ``exc`` or its cause, context or group members.

    A validation error is a pydantic ``ValidationError`` or one of FastAPI's
    ``ValidationException`` family (``RequestValidationError``, ``ResponseValidationError``,
    ``WebSocketRequestValidationError``): both render every input value in their message.
    """
    types = _validation_error_types()
    if exc is None or not types:
        return None
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending and len(seen) < _MAX_LINKED_EXCEPTIONS:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, types):
            return current
        linked = [current.__cause__, current.__context__]
        if isinstance(current, BaseExceptionGroup):
            linked.extend(current.exceptions)
        pending.extend(item for item in linked if item is not None)
    return None


def _pydantic_validation_error_type() -> type[BaseException] | None:
    """The pydantic v2 ValidationError class (pydantic and pydantic_settings raise it)."""
    try:
        from pydantic import ValidationError  # re-export of pydantic_core's class; pydantic is the locked dependency
    except ImportError:
        return None
    return ValidationError


def _fastapi_validation_error_type() -> type[BaseException] | None:
    """FastAPI's ValidationException, the base of its request, response and WebSocket errors."""
    try:
        from fastapi.exceptions import ValidationException
    except ImportError:
        return None
    return ValidationException


def _validation_error_types() -> tuple[type[BaseException], ...]:
    """Every exception class whose message renders validation input (see _find_validation_error)."""
    found = (_pydantic_validation_error_type(), _fastapi_validation_error_type())
    return tuple(cls for cls in found if cls is not None)


def _validation_error_fields(exc: BaseException) -> list[dict[str, str]]:
    """Error type and masked field path per validation error (at most 50); never the input."""
    try:
        pydantic_error = _pydantic_validation_error_type()
        if pydantic_error is not None and isinstance(exc, pydantic_error):
            errors = list(
                exc.errors(  # type: ignore[attr-defined]
                    include_input=False, include_url=False, include_context=False
                )
            )
        else:
            # FastAPI's ValidationException family: errors() takes no arguments and its dicts
            # still hold the input; only "type" and the masked "loc" are read from them.
            errors = list(exc.errors())  # type: ignore[attr-defined]
        return [
            {"type": str(error.get("type", "")), "field": _masked_field_path(error)}
            for error in errors[:_MAX_VALIDATION_ERRORS]
            if isinstance(error, dict)
        ]
    except Exception:
        return []


def _field_path(loc: Any) -> str:
    """A location as a dotted path; any part that is not an int or an identifier is ``[key]``."""
    parts: list[str] = []
    for part in loc if isinstance(loc, list | tuple) else (loc,):
        if isinstance(part, int):
            parts.append(str(part))
        elif isinstance(part, str) and _FIELD_PART_PATTERN.fullmatch(part):
            parts.append(part)
        else:
            parts.append(_MASKED_FIELD_PART)
    return ".".join(parts)[:_MAX_FIELD_LENGTH]


def _masked_field_path(error: Any) -> str:
    """The masked field path of one pydantic error, for a log line or a 422 body.

    An ``extra_forbidden`` error's last location part is the sender's own key name, which can be
    any text (a name, say) even when it is shaped like an identifier, so it is always masked.
    """
    loc = error.get("loc", ())
    if str(error.get("type", "")) == "extra_forbidden" and isinstance(loc, list | tuple) and loc:
        loc = (*loc[:-1], _MASKED_FIELD_PART)
    return _field_path(loc)


def _raise_site(exc: BaseException) -> str | None:
    """``file:line:function`` of the frame that raised ``exc``. Never its message."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return None
    frame = frames[-1]
    return f"{frame.filename}:{frame.lineno}:{frame.name}"


def _exception_from_exc_info(value: Any) -> BaseException | None:
    """The exception an ``exc_info`` value names (as structlog resolves it), else None."""
    if isinstance(value, BaseException):
        return value
    if isinstance(value, tuple) and len(value) == 3 and isinstance(value[1], BaseException):
        return value[1]
    if value:
        return sys.exc_info()[1]
    return None


def _add_exception_summary(
    event_dict: EventDict, exc: BaseException | None, validation: BaseException | None
) -> None:
    """Describe ``exc`` without its text: class name, validation errors and raise site."""
    event_dict.setdefault("error_type", _type_name(exc))
    event_dict.setdefault(
        "validation_errors",
        _validation_error_fields(validation) if validation is not None else [],
    )
    origin = _raise_site(exc) if exc is not None else None
    if origin is not None:
        event_dict.setdefault("origin", origin)


class _ValidationSafeExceptionRenderer(structlog.processors.ExceptionRenderer):
    """The chain's ``format_exc_info`` step: structlog's, except for pydantic input (PHI-1).

    It renders ``exc_info`` into ``exception`` as ``structlog.processors.format_exc_info`` does,
    with two exceptions:

    - when the exception is a validation error (pydantic's ``ValidationError`` or FastAPI's
      ``ValidationException`` family), or one is linked to it (``__cause__``, ``__context__`` or
      an ExceptionGroup member), nothing is rendered: the line gets ``error_type`` (the logged
      exception's class), ``validation_errors`` (type and masked field per error, never the
      input) and ``origin`` (the raise site) instead;
    - rendered traceback text that still quotes validation input (pydantic's ``input_value=``,
      or FastAPI's ``'input':`` error dicts, copied into another exception's message) is dropped
      the same way, with empty ``validation_errors``.

    Text outside ``exception`` (the event, other values) is handled by ``redact_credentials``.

    Fields the caller already set (``error_type``, for one) are kept. It fills the slot of
    ``format_exc_info`` in both chains and ``logging_chain_names()`` reports it under that name.
    """

    def __call__(self, logger: Any, name: str, event_dict: EventDict) -> EventDict:
        if "exc_info" not in event_dict:
            return event_dict
        exc = _exception_from_exc_info(event_dict.get("exc_info"))
        validation = _find_validation_error(exc)
        if validation is not None:
            del event_dict["exc_info"]
            _add_exception_summary(event_dict, exc, validation)
            return event_dict
        event_dict = super().__call__(logger, name, event_dict)
        rendered = event_dict.get("exception")
        if isinstance(rendered, str) and _quotes_validation_input(rendered):
            del event_dict["exception"]
            _add_exception_summary(event_dict, exc, None)
        return event_dict


_FORMAT_EXC_INFO = _ValidationSafeExceptionRenderer()


def configure_service_logging(spec: ServiceRuntimeSpec) -> None:
    """Install the runtime logging chain for structlog and for the stdlib root logger.

    Safe to call more than once. Each call replaces the structlog configuration and the root
    handler that an earlier call installed.

    Ordering contract: uvicorn applies its log_config when it starts (CLI or uvicorn.run),
    before the lifespan; call this function from the lifespan (start_service_runtime), never
    at import time. log_config=None is optional.

    Raises:
        StartupDependencyError: in a strict environment, when the PHI log sanitizer cannot be
            imported or fails its self-test (for example ``PHI_ENABLE_MASKING=false``); in
            any environment, when the sanitizer is importable but its ``PHI_*`` settings are
            invalid (for example ``PHI_PARTIAL_MASK_VISIBLE=5``).
    """
    global _phi_key_patterns
    strict = _is_strict(spec)
    sanitizer, sanitizer_error = _load_phi_sanitizer(strict)
    level, level_known = _resolve_level(spec.log_level)
    scrubbers: list[Processor] = [redact_credentials]
    if sanitizer is not None:
        scrubbers.append(sanitizer)
    # Dict keys are checked against the sanitizer's own patterns (none without a sanitizer).
    _phi_key_patterns = _load_phi_key_patterns() if sanitizer is not None else ()

    renderer: Processor
    if spec.debug and not strict:
        # The traceback is already text (format_exc_info), so print it as is.
        renderer = structlog.dev.ConsoleRenderer(exception_formatter=structlog.dev.plain_traceback)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            _FORMAT_EXC_INFO,
            *scrubbers,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        # Not cached, so a later configure call reaches every logger and none keeps an older
        # chain or a replaced stdout.
        cache_logger_on_first_use=False,
    )
    _route_stdlib_logging(level, scrubbers)
    _capture_warnings()

    log = structlog.get_logger(__name__)
    if sanitizer_error is not None:
        log.warning(
            "phi_log_sanitizer_unavailable",
            service=spec.service_name,
            environment=spec.environment,
            error_type=sanitizer_error,
        )
    if not level_known:
        log.warning(
            "log_level_unrecognised",
            service=spec.service_name,
            log_level=spec.log_level,
            using=logging.getLevelName(_DEFAULT_LEVEL),
        )


def logging_chain_names() -> list[str]:
    """Return the names of the processors in the active structlog configuration, in order."""
    return [_processor_name(p) for p in structlog.get_config()["processors"]]


def redact_credentials(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """structlog processor: normalise every value to JSON-native data, then rewrite credentials.

    Normalising first means the PHI sanitizer that follows (which reads strings, dicts and
    lists only) sees everything JSONRenderer will print (PHI-3): tuples, sets and frozensets
    become lists; any other non-JSON value (bytes, an arbitrary object) becomes the text
    JSONRenderer would print for it (``__structlog__()`` or ``repr``); dict keys become strings,
    and a key that matches one of the PHI sanitizer's patterns becomes ``[key]``.

    Validation input never reaches a rendered string (WS-04.F1 fix rounds 1 and 2). An
    exception value that is, or is linked to, a validation error becomes ``{"error_type",
    "validation_errors"}`` instead of its repr (any other exception keeps its repr). An error
    dict of pydantic's or FastAPI's ``errors()`` (``type`` and ``loc`` plus ``input``, ``ctx``,
    ``url`` or ``msg``), at any depth, becomes ``{"type", "field"}`` with the field masked. A
    string that quotes validation input (``input_value=``, ``'input':``, ``"input":``, with the
    quotes escaped or not), the event text included, is cut to its first line, or to a fixed
    text when the first line quotes it too.

    Credential shapes are rewritten in every string, keys included: DSN and URL userinfo
    (``scheme://user:pass@``), ``Bearer``/``Basic`` credentials, credential query parameters
    (``?token=``, ``&code=``, ...) and ``password=`` pairs.
    """
    return _redact_mapping(event_dict)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        value = _withhold_validation_input(value)
        for pattern, replacement in _CREDENTIAL_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if value is None or isinstance(value, int | float):
        return value
    if isinstance(value, dict):
        if _is_validation_error_dict(value):
            return _reduced_error_dict(value)
        return _redact_mapping(value)
    if isinstance(value, list | tuple | set | frozenset):
        return [_redact(item) for item in value]
    if isinstance(value, BaseException):
        validation = _find_validation_error(value)
        if validation is not None:
            return {
                "error_type": type(value).__name__,
                "validation_errors": _validation_error_fields(validation),
            }
    return _redact(_renderable(value))


def _is_validation_error_dict(value: dict[Any, Any]) -> bool:
    """Whether ``value`` is one error dict of pydantic's or FastAPI's ``errors()``.

    Such a dict has ``type`` and ``loc`` plus one of the keys that hold the input or text that
    can quote it: ``input``, ``ctx`` (``ctx.error`` is the validator's own exception), ``url``
    or ``msg`` (a ``value_error`` message is the validator author's text).
    """
    return "type" in value and "loc" in value and not _ERROR_DICT_DETAIL_KEYS.isdisjoint(value)


def _reduced_error_dict(error: dict[Any, Any]) -> dict[str, Any]:
    """An error dict as its type and masked field path, as every validation summary logs it."""
    return {"type": _redact(str(error.get("type", ""))), "field": _masked_field_path(error)}


def _redact_mapping(mapping: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in mapping.items():
        name = _safe_key(key)
        if name in result:  # two keys that render alike (1 and "1", or two masked keys)
            suffix = 2
            while f"{name}#{suffix}" in result:
                suffix += 1
            name = f"{name}#{suffix}"
        result[name] = _redact(item)
    return result


def _safe_key(key: Any) -> str:
    """A dict key as JSONRenderer would print it, credential-redacted, ``[key]`` if PHI-shaped."""
    if isinstance(key, str):
        text = key
    elif key is None or isinstance(key, int | float):
        text = json.dumps(key)  # 1 -> "1", True -> "true", None -> "null", as json.dumps does
    else:
        rendered = _renderable(key)
        text = rendered if isinstance(rendered, str) else repr(rendered)
    text = _redact(text)
    if any(pattern.search(text) for pattern in _phi_key_patterns):
        return _MASKED_KEY
    return text


def _renderable(value: Any) -> Any:
    """What JSONRenderer prints for a value it cannot serialise: ``__structlog__()``, else repr."""
    method = getattr(type(value), "__structlog__", None)
    if method is not None:
        with contextlib.suppress(Exception):
            result = method(value)
            if result is None or isinstance(result, _JSON_NATIVE):
                return result
            value = result
    try:
        return repr(value)
    except Exception:  # a broken __repr__ must not fail the log call
        return f"<{type(value).__name__}>"


def _load_phi_key_patterns() -> tuple[re.Pattern[str], ...]:
    """The PHI log sanitizer's patterns, compiled as its detector compiles them, else none."""
    try:
        from solace_security.phi_protection import DEFAULT_PATTERNS
    except ImportError:
        return ()
    return tuple(re.compile(p.pattern, re.IGNORECASE) for p in DEFAULT_PATTERNS)


def _is_strict(spec: ServiceRuntimeSpec) -> bool:
    """Strict when the service's own environment or the deployment's ENVIRONMENT is strict."""
    return is_strict_environment(spec.environment) or is_strict_environment()


def _load_phi_sanitizer(strict: bool) -> tuple[Processor | None, str | None]:
    """Return ``(sanitizer, None)``, or ``(None, error type)`` when it is unavailable outside strict."""
    try:
        from solace_security.phi_protection import phi_sanitizer_processor
    except ImportError as exc:
        if strict:
            raise StartupDependencyError("logging", "PHI log sanitizer unavailable") from exc
        return None, type(exc).__name__
    if strict:
        _assert_sanitizer_active(phi_sanitizer_processor)
    else:
        _prime_sanitizer(phi_sanitizer_processor)
    return phi_sanitizer_processor, None


def _prime_sanitizer(sanitizer: Processor) -> None:
    """Build the lazily created sanitizer once, outside strict environments.

    The sanitizer reads its ``PHI_*`` settings on first use. Invalid settings (for example
    ``PHI_PARTIAL_MASK_VISIBLE=5``) would otherwise make every later log call raise.
    """
    try:
        sanitizer(None, "info", {"event": "phi_sanitizer_ready"})
    except Exception as exc:
        raise StartupDependencyError("logging", "PHI log sanitizer settings are invalid") from exc


def _assert_sanitizer_active(sanitizer: Processor) -> None:
    """Prove the sanitizer masks a synthetic e-mail address and phone number.

    ``PHI_ENABLE_MASKING=false`` or ``PHI_ENABLE_DETECTION=false`` turn the sanitizer into a
    pass-through while it still appears in the chain; a strict service must not start so.
    ``PHI_PARTIAL_MASK_VISIBLE=0`` masks every digit and values outside 0..4 are rejected by
    PHISettings (which fails this self-test). The digit-window check is defence in depth
    against settings that skip validation (model_construct) and show five or more digits.
    """
    try:
        probe = sanitizer(None, "info", {"event": f"canary {_CANARY_EMAIL} {_CANARY_PHONE}"})
    except Exception as exc:
        raise StartupDependencyError("logging", "PHI log sanitizer failed its self-test") from exc
    rendered = str(probe)
    digits = _NON_DIGITS.sub("", rendered)
    leaked_digits = any(
        _CANARY_PHONE_DIGITS[start : start + _CANARY_DIGIT_WINDOW] in digits
        for start in range(len(_CANARY_PHONE_DIGITS) - _CANARY_DIGIT_WINDOW + 1)
    )
    if _CANARY_EMAIL in rendered or _CANARY_PHONE in rendered or leaked_digits:
        raise StartupDependencyError("logging", "PHI log sanitizer inactive")


def _resolve_level(name: str) -> tuple[int, bool]:
    """Map a level name to its number; an unknown name gives INFO and ``False``."""
    level = logging.getLevelNamesMapping().get(name.strip().upper())
    if level is None:
        return _DEFAULT_LEVEL, False
    return level, True


def _route_stdlib_logging(level: int, scrubbers: list[Processor]) -> None:
    """Give the root logger one JSON handler on stdout and route the library loggers to it."""
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            _FORMAT_EXC_INFO,
            # Carries the fields that _AsgiExceptionFilter, _WarningFilter (or an explicit
            # extra) set.
            structlog.stdlib.ExtraAdder(
                allow=["error_type", "validation_errors", "warning_category"]
            ),
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *scrubbers,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = _RuntimeHandler(_CurrentStdout())
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
            existing.close()
    root.addHandler(handler)
    root.setLevel(level)

    for name in ROUTED_STDLIB_LOGGERS:
        routed = logging.getLogger(name)
        _remove_handlers(routed)
        routed.setLevel(logging.NOTSET)
        routed.propagate = True
        routed.disabled = False
    logging.getLogger("uvicorn.error").setLevel(max(level, UVICORN_ERROR_MIN_LEVEL))

    for name in ASGI_EXCEPTION_LOGGERS:
        _replace_filter(logging.getLogger(name), _AsgiExceptionFilter())
    _replace_filter(logging.getLogger(WARNINGS_LOGGER), _WarningFilter())

    for name in QUIETED_STDLIB_LOGGERS:
        logging.getLogger(name).setLevel(max(level, OUTBOUND_HTTP_MIN_LEVEL))

    for name in SILENCED_STDLIB_LOGGERS:
        silenced = logging.getLogger(name)
        _remove_handlers(silenced)
        silenced.propagate = False
        silenced.disabled = True


def _remove_handlers(logger: logging.Logger) -> None:
    for existing in list(logger.handlers):
        logger.removeHandler(existing)


def _replace_filter(logger: logging.Logger, new: logging.Filter) -> None:
    """Install ``new`` on ``logger``, removing any filter of the same class an earlier call added."""
    for existing in list(logger.filters):
        if type(existing) is type(new):
            logger.removeFilter(existing)
    logger.addFilter(new)


def _capture_warnings() -> None:
    """Send ``warnings.warn`` output to the ``py.warnings`` logger, so it gets the runtime chain.

    ``logging.captureWarnings(True)`` does nothing once it has run, even when
    ``warnings.showwarning`` has been put back since (a ``warnings.catch_warnings()`` block
    restores it on exit), so the capture is reset first.
    """
    logging.captureWarnings(False)
    logging.captureWarnings(True)


def _processor_name(processor: Any) -> str:
    # format_exc_info is a module-level ExceptionRenderer instance, so it has no __name__. The
    # runtime's own exception step fills that slot and is reported under the same name.
    if processor is structlog.processors.format_exc_info or isinstance(
        processor, _ValidationSafeExceptionRenderer
    ):
        return "format_exc_info"
    name = getattr(processor, "__name__", None)
    return name if isinstance(name, str) else type(processor).__name__


__all__ = ["configure_service_logging", "logging_chain_names", "redact_credentials"]
