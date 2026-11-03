"""The IF-02 deployment env registry (``deploy/env/env-contract.yaml``) and synthetic values.

Each registry row describes one container variable for one set of services:

- ``name``: the variable the container sees; ``services``: the compose services that get it;
- ``source``: the env-file / secrets-store key it is interpolated from (None: a compose literal);
- ``profiles``: ``dev`` (docker-compose.yml) and/or ``strict`` (+ docker-compose.prod.yml);
- ``secret``, ``format`` (one of ``FORMATS``), ``required_in_strict`` (the strict overlay uses
  ``${SOURCE:?...}``), ``dev_default`` (the committed dev fallback or literal, None when the value
  is empty unless supplied), ``strict_default`` (the strict value when the source is unset),
  ``example`` (placeholder for operator-supplied non-secrets), ``group`` (``.env.example``
  heading; ``legacy`` for aliases awaiting retirement), ``notes`` and ``retired``.

The same ``name`` may appear in several rows with different services or sources.
"""
from __future__ import annotations

import os
import re
import secrets
import string
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import yaml

REGISTRY_PATH = Path(__file__).with_name("env-contract.yaml")

APP_SERVICES: tuple[str, ...] = (
    "orchestrator-service",
    "user-service",
    "safety-service",
    "notification-service",
    "diagnosis-service",
    "memory-service",
    "therapy-service",
    "personality-service",
    "analytics-service",
)

PROFILES: tuple[str, ...] = ("dev", "strict")

FORMATS: frozenset[str] = frozenset({
    "text", "int", "bool", "url", "email", "hostport", "csv", "json",
    "urlsafe24", "urlsafe48", "hex64", "fernet", "bcrypt", "pem",
})

# Formats generate_value can produce without an example.
GENERATED_FORMATS: frozenset[str] = frozenset({"urlsafe24", "urlsafe48", "hex64", "fernet", "bcrypt", "pem"})

# Extra sources (names the compose files still interpolate but the registry no longer lists)
# get this format when synthetic_values is asked for them.
EXTRA_SOURCE_FORMAT = "urlsafe48"

_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_BCRYPT_ALPHABET = "./" + string.ascii_uppercase + string.ascii_lowercase + string.digits


class RegistryError(ValueError):
    """The registry file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class EnvVar:
    name: str
    services: tuple[str, ...]
    source: str | None = None
    profiles: tuple[str, ...] = PROFILES
    secret: bool = False
    format: str = "text"
    required_in_strict: bool = False
    dev_default: str | None = None
    strict_default: str | None = None
    example: str | None = None
    group: str = ""
    notes: str = ""
    retired: bool = False


_FIELD_NAMES = frozenset(f.name for f in fields(EnvVar))
_TEXT_FIELDS = ("dev_default", "strict_default", "example")


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise TypeError(f"expected a scalar, got {type(value).__name__}")


def _as_tuple(value: Any, what: str, where: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise RegistryError(f"{where}: {what} must be a list")
    if not all(isinstance(v, str) and v for v in value):
        raise RegistryError(f"{where}: {what} must hold non-empty strings")
    return tuple(value)


def _row(raw: Any, index: int) -> EnvVar:
    where = f"variables[{index}]"
    if not isinstance(raw, dict):
        raise RegistryError(f"{where}: each row must be a mapping")
    unknown = set(raw) - _FIELD_NAMES
    if unknown:
        raise RegistryError(f"{where}: unknown keys {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise RegistryError(f"{where}: name must be an upper-case env name")
    where = f"{where} ({name})"
    source = raw.get("source")
    if source is not None and (not isinstance(source, str) or not _NAME.match(source)):
        raise RegistryError(f"{where}: source must be an upper-case env name or null")
    kwargs: dict[str, Any] = {"name": name, "source": source,
                              "services": _as_tuple(raw.get("services"), "services", where)}
    if "profiles" in raw:
        kwargs["profiles"] = _as_tuple(raw["profiles"], "profiles", where)
        if not set(kwargs["profiles"]) <= set(PROFILES):
            raise RegistryError(f"{where}: profiles must be a subset of {list(PROFILES)}")
    for flag in ("secret", "required_in_strict", "retired"):
        if flag in raw:
            if not isinstance(raw[flag], bool):
                raise RegistryError(f"{where}: {flag} must be true or false")
            kwargs[flag] = raw[flag]
    for key in _TEXT_FIELDS:
        try:
            kwargs[key] = _as_text(raw.get(key))
        except TypeError as exc:
            raise RegistryError(f"{where}: {key}: {exc}") from None
    for key in ("format", "group", "notes"):
        if key in raw:
            if not isinstance(raw[key], str):
                raise RegistryError(f"{where}: {key} must be a string")
            kwargs[key] = raw[key]
    row = EnvVar(**kwargs)
    if row.format not in FORMATS:
        raise RegistryError(f"{where}: format {row.format!r} is not one of {sorted(FORMATS)}")
    if not row.group:
        raise RegistryError(f"{where}: group is required")
    if row.secret and (row.strict_default is not None or row.example is not None):
        raise RegistryError(f"{where}: a secret has no committed strict_default or example")
    if row.required_in_strict and row.source is None:
        raise RegistryError(f"{where}: a compose literal cannot be required_in_strict")
    return row


def _check_consistency(rows: list[EnvVar]) -> None:
    seen: set[tuple[str, str]] = set()
    by_source: dict[str, EnvVar] = {}
    for row in rows:
        for service in row.services:
            if (row.name, service) in seen:
                raise RegistryError(f"{row.name} is registered twice for {service}")
            seen.add((row.name, service))
        if row.source is None or row.retired:
            continue
        first = by_source.setdefault(row.source, row)
        if (first.secret, first.format) != (row.secret, row.format):
            raise RegistryError(f"rows sourced from {row.source} disagree on secret/format "
                                f"({first.name} vs {row.name})")
    for source, rows_for_source in sources(rows, "strict").items():
        required = any(r.required_in_strict for r in rows_for_source)
        if required and source != "ENVIRONMENT" and not rows_for_source[0].secret:
            if not any(r.example for r in rows_for_source):
                raise RegistryError(f"{source} is required in strict but no row gives an example")


def load_registry(path: Path = REGISTRY_PATH) -> list[EnvVar]:
    """Parse and validate the registry file; rows keep file order."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("variables"), list):
        raise RegistryError(f"{path}: expected a mapping with a 'variables' list")
    rows = [_row(raw, i) for i, raw in enumerate(data["variables"])]
    _check_consistency(rows)
    return rows


def sources(registry: Iterable[EnvVar], profile: str | None = None) -> dict[str, list[EnvVar]]:
    """Non-retired rows grouped by source (file order), for one profile or for any profile."""
    out: dict[str, list[EnvVar]] = {}
    for row in registry:
        if row.source and not row.retired and (profile is None or profile in row.profiles):
            out.setdefault(row.source, []).append(row)
    return out


def _pem_private_key() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode("ascii")
    # Single-lined with literal \n escapes (solace_security.keys._normalize_pem restores them).
    return pem.strip().replace("\n", "\\n")


def generate_value(fmt: str, *, example: str | None = None) -> str:
    """A fresh random value for a secret format, or ``example`` for any other format.

    ``bcrypt`` yields a ``$2a$14$``-prefixed 60-character string with the bcrypt alphabet. It is
    shaped like a hash but is not one; it exists for render tests only.
    """
    if fmt == "urlsafe24":
        return secrets.token_urlsafe(24)
    if fmt == "urlsafe48":
        return secrets.token_urlsafe(48)
    if fmt == "hex64":
        return secrets.token_hex(32)
    if fmt == "fernet":
        from cryptography.fernet import Fernet

        return Fernet.generate_key().decode("ascii")
    if fmt == "bcrypt":
        return "$2a$14$" + "".join(secrets.choice(_BCRYPT_ALPHABET) for _ in range(53))
    if fmt == "pem":
        return _pem_private_key()
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    if example is None:
        raise ValueError(f"format {fmt!r} has no generator; the registry row needs an example")
    return example


def synthetic_values(
    registry: Iterable[EnvVar],
    *,
    profile: Literal["dev", "strict"],
    environment: str,
    extra_sources: Iterable[str] = (),
) -> dict[str, str]:
    """Interpolation values for rendering or booting ``profile`` with ``ENVIRONMENT=environment``.

    dev: only ``ENVIRONMENT``, so every committed dev default applies (as on a fresh checkout).
    strict: ``ENVIRONMENT``, a fresh value for every source a row requires in strict (secrets
    generated, non-secrets from their ``example``) and every non-secret source with an
    ``example`` (operator-supplied settings). Optional secrets stay unset.
    ``extra_sources`` get fresh ``urlsafe48`` values in both profiles.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    values: dict[str, str] = {"ENVIRONMENT": environment}
    if profile == "strict":
        for source, rows in sources(registry, profile).items():
            if source in values:
                continue
            head = rows[0]
            example = next((r.example for r in rows if r.example is not None), None)
            required = any(r.required_in_strict for r in rows)
            if head.secret:
                if required:
                    values[source] = generate_value(head.format)
            elif example is not None:
                values[source] = example
            elif required:
                raise RegistryError(f"{source} is required in strict but has no example")
    for name in extra_sources:
        values.setdefault(name, generate_value(EXTRA_SOURCE_FORMAT))
    return values


def write_synthetic_env(
    path: Path,
    *,
    profile: Literal["dev", "strict"],
    environment: str,
    registry: Iterable[EnvVar] | None = None,
) -> None:
    """Write ``synthetic_values`` as a compose ``--env-file`` (single-quoted lines, mode 0600).

    The values are written, never printed.
    """
    from deploy.env.compose_render import format_env_file

    rows = list(registry) if registry is not None else load_registry()
    text = format_env_file(synthetic_values(rows, profile=profile, environment=environment))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
