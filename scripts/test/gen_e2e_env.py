"""Per-run secrets for the E2E compose stack (MVP-039, WS-05.4). TEST ONLY.

    python -m scripts.test.gen_e2e_env --profile dev|strict --environment development|staging|production --out DIR
        [--registry deploy/env/env-contract.yaml] [--rotate] [--certs-cmd CMD]

Writes, for docker-compose.test.yml and the e2e_stack fixture (src/solace_testing/e2e.py):

* ``DIR/e2e.env``: the compose interpolation values, one ``NAME="value"`` line each with every ``$`` written ``$$``
  (compose reads ``$$`` as a literal ``$``), for ``docker compose --env-file``. It holds:
  - ``ENVIRONMENT`` (``development`` in dev; ``staging`` or ``production`` in strict);
  - ``E2E_SMTP_TLS_MODE`` (``none`` in dev, ``starttls`` in strict) and, in strict only,
    ``E2E_SSL_CERT_FILE=/run/secrets/e2e-ca.crt``, the CA the services trust for mailpit's STARTTLS;
  - ``SOLACE_SECRETS_DIR`` (``DIR/secrets``, forward slashes);
  - a fresh value for every source the registry marks ``secret`` (``source``, or ``name`` when a row has no
    source), by the format rules below, plus ``E2E_BASIC_AUTH_PASSWORD`` (the plaintext of ``BASIC_AUTH_HASH``,
    for harness use) and ``REDIS_URL`` (derived from ``REDIS_PASSWORD``);
  - in strict, the registry ``example`` of every non-secret source that is required in strict and has no
    ``strict_default`` (for example ``DOMAIN``), so the strict render's ``${SOURCE:?...}`` checks pass.
* ``DIR/secrets/``: ``e2e-ca.crt``, ``mailpit.crt`` and ``mailpit.key`` (a fresh CA and the certificate for host
  ``mailpit``; written in both profiles because the one overlay mounts the CA on every app service and mailpit's
  pair on mailpit in both, but only strict uses them) and, in strict, the IF-22 Postgres TLS files
  ``postgres-ca.crt``, ``server.crt`` and
  ``server.key`` (SAN ``postgres``). Both sets come from ``python -m scripts.db.gen_certs`` (``--certs-cmd``
  replaces that command in tests: it is run as ``CMD --out D --san NAME`` and must write the three gen_certs
  file names into D).

Format rules (a registry secret without one stops the run with ``no format rule for <NAME>``, exit 2):

    hex32      ENCRYPTION_MASTER_KEY, ENCRYPTION_SEARCH_HASH_SALT, AUDIT_HMAC_KEY        secrets.token_hex(32)
    urlsafe32  AUTH_SECRET_KEY, USER_SERVICE_JWT_SECRET_KEY, every *_PASSWORD, WEAVIATE_API_KEY,
               LLM_PORTKEY_API_KEY, GOOGLE_OAUTH_CLIENT_SECRET, BACKUP_ENCRYPTION_KEY,
               USER_SERVICE_FIELD_ENCRYPTION_KEY                                         secrets.token_urlsafe(32)
    fernet     FERNET_TOKEN_KEY, FERNET_FIELD_KEY                                        32 random bytes, urlsafe b64
    bcrypt     BASIC_AUTH_HASH                                                           bcrypt of a fresh password
    derived    REDIS_URL from REDIS_PASSWORD                                             redis://:<pw>@redis:6379/0
    literal    ENCRYPTION_KEY_ID="e2e-k1", ENCRYPTION_RETIRED_KEYS_JSON="{}"
    unset      LLM_ANTHROPIC_API_KEY (D-04: the overlay pins it empty, so nothing reaches a provider),
               AUTH_PRIVATE_KEY_PEM (unused while AUTH_ALGORITHM=HS256; the W4 RS256 cut-over adds a rule)

DIR must be new, empty, or an earlier run: the generator writes ``DIR/.solace-e2e-run`` (RUN_MARKER) and refuses a
non-empty DIR without it, so it never restricts or writes into a foreign directory, and the harness deletes only
marked run dirs. Every check (rules, arguments, DIR) runs before anything is written.

Access: DIR is restricted to the current user before anything is written into it.
* Windows: ``icacls DIR /inheritance:r /grant:r "<user>:(OI)(CI)F"`` (the user given by SID); files inherit it.
* POSIX, owner the generating user:
  - ``DIR`` and ``DIR/secrets`` 0700, and ``e2e.env`` 0600 (read on the host only, by compose);
  - ``e2e-ca.crt``, ``mailpit.crt``, ``postgres-ca.crt`` and ``server.crt`` 0644 (public material). Compose file
    secrets are bind mounts that keep the host owner and mode, and their readers include non-root users: the app
    services and migrate run as appuser (uid 1000) and read the CAs;
  - ``mailpit.key`` and ``server.key`` 0600. Their readers run as root: mailpit (the image sets no USER) and
    postgres, whose TLS wrapper copies the key, as root, to a postgres-owned 0600 file before stepping down. A
    rootless or user-namespaced Docker would need another arrangement.
  A mounted file is reached through its mount point, not through the 0700 host directories.
Only variable and file names are printed, never a value.

Exit codes: 0 written; 1 refused (DIR already holds a run and --rotate is not given, or DIR is a foreign
non-empty directory) or failed; 2 bad arguments or a registry entry without a format rule. ``main(argv)`` returns
the code.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import secrets
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import yaml

ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY: Final = ROOT / "deploy" / "env" / "env-contract.yaml"
ENV_FILE: Final = "e2e.env"
SECRETS_DIR: Final = "secrets"
RUN_MARKER: Final = ".solace-e2e-run"  # solace_testing.e2e.RUN_MARKER is the same name

PROFILES: Final = {"dev": ("development",), "strict": ("staging", "production")}
CONTAINER_SECRETS: Final = "/run/secrets"
E2E_CA_FILE: Final = "e2e-ca.crt"
MAILPIT_FILES: Final = {"postgres-ca.crt": E2E_CA_FILE, "server.crt": "mailpit.crt", "server.key": "mailpit.key"}
POSTGRES_FILES: Final = ("postgres-ca.crt", "server.crt", "server.key")
# POSIX modes of the secrets files (see the module docstring for who reads each one).
FILE_MODES: Final = {E2E_CA_FILE: 0o644, "mailpit.crt": 0o644, "postgres-ca.crt": 0o644, "server.crt": 0o644,
                     "mailpit.key": 0o600, "server.key": 0o600}
PRIVATE_MODE: Final = 0o600

HEX32: Final = frozenset({"ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT", "AUDIT_HMAC_KEY"})
URLSAFE32: Final = frozenset({
    "AUTH_SECRET_KEY", "USER_SERVICE_JWT_SECRET_KEY", "WEAVIATE_API_KEY", "LLM_PORTKEY_API_KEY",
    "GOOGLE_OAUTH_CLIENT_SECRET", "BACKUP_ENCRYPTION_KEY", "USER_SERVICE_FIELD_ENCRYPTION_KEY",
})
FERNET: Final = frozenset({"FERNET_TOKEN_KEY", "FERNET_FIELD_KEY"})
BCRYPT: Final = frozenset({"BASIC_AUTH_HASH"})
DERIVED: Final = {"REDIS_URL": ("REDIS_PASSWORD", "redis://:{}@redis:6379/0")}
LITERALS: Final = {"ENCRYPTION_KEY_ID": "e2e-k1", "ENCRYPTION_RETIRED_KEYS_JSON": "{}"}
UNSET: Final = frozenset({"LLM_ANTHROPIC_API_KEY", "AUTH_PRIVATE_KEY_PEM"})
BASIC_AUTH_PASSWORD: Final = "E2E_BASIC_AUTH_PASSWORD"
BCRYPT_ROUNDS: Final = 10  # caddy verifies it on every basic-auth request; the test value needs no more

CERTS_TIMEOUT_S: Final = 120


class _Exit(Exception):  # noqa: N818 - control flow: a message for stderr and an exit code
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _note(message: str) -> None:
    print(f"gen_e2e_env: {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------- rules


def rule_for(name: str) -> str | None:
    """The format rule for a secret source name, or None when there is none."""
    if name in HEX32:
        return "hex32"
    if name in FERNET:
        return "fernet"
    if name in BCRYPT:
        return "bcrypt"
    if name in LITERALS:
        return "literal"
    if name in UNSET:
        return "unset"
    if name in DERIVED:
        return "derived"
    if name in URLSAFE32 or name.endswith("_PASSWORD"):
        return "urlsafe32"
    return None


def _generate(rule: str, name: str, values: dict[str, str]) -> str | None:
    if rule == "hex32":
        return secrets.token_hex(32)
    if rule == "urlsafe32":
        return secrets.token_urlsafe(32)
    if rule == "fernet":
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
    if rule == "bcrypt":
        import bcrypt

        password = values.setdefault(BASIC_AUTH_PASSWORD, secrets.token_urlsafe(24))
        return bcrypt.hashpw(password.encode("ascii"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")
    if rule == "derived":
        source, template = DERIVED[name]
        return template.format(values.setdefault(source, secrets.token_urlsafe(32)))
    if rule == "literal":
        return LITERALS[name]
    return None  # unset


# --------------------------------------------------------------------------------------------- registry


def _rows(path: Path) -> list[dict[str, Any]]:
    """The registry rows: deploy/env/env-contract.yaml's ``variables`` list, or a bare list of rows."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise _Exit(2, f"error: cannot read the registry {path}: {type(exc).__name__}") from None
    rows = data.get("variables") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("name"), str) for r in rows):
        raise _Exit(2, f"error: {path} is not a registry (a list of rows with a name, or a mapping with 'variables')")
    return [r for r in rows if not r.get("retired")]


def _secret_sources(rows: list[dict[str, Any]]) -> list[str]:
    """Every secret source, once, in registry order (a row without a source is its own source)."""
    seen: dict[str, None] = {}
    for row in rows:
        if row.get("secret"):
            seen.setdefault(row.get("source") or row["name"], None)
    return list(seen)


def _strict_operator_values(rows: list[dict[str, Any]], values: dict[str, str]) -> dict[str, str]:
    """Strict: the example of every non-secret source required in strict that has no strict default."""
    out: dict[str, str] = {}
    missing: list[str] = []
    for row in rows:
        source = row.get("source")
        if row.get("secret") or not row.get("required_in_strict") or not source or source in values:
            continue
        if row.get("strict_default") is not None or source in out:
            continue
        example = row.get("example")
        if example is None:
            missing.append(source)
        else:
            out[source] = str(example)
    if missing:
        raise _Exit(2, "\n".join(f"no format rule for {name}" for name in dict.fromkeys(missing)))
    return out


# --------------------------------------------------------------------------------------------- files and access


def _current_user_sid() -> str:
    proc = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True,  # noqa: S603, S607
                          timeout=30, check=False)
    row = next(csv.reader(io.StringIO(proc.stdout.strip())), [])
    if proc.returncode != 0 or len(row) < 2 or not row[1].startswith("S-1-"):
        raise _Exit(1, "error: could not determine the current user's SID (whoami /user)")
    return row[1]


def _restrict_dir(path: Path) -> None:
    """Only the current user may use ``path`` (and, on Windows, everything later created in it)."""
    if os.name == "nt":
        grant = f"*{_current_user_sid()}:(OI)(CI)F"
        proc = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", grant],  # noqa: S603, S607
                              capture_output=True, text=True, timeout=60, check=False)
        if proc.returncode != 0:
            raise _Exit(1, f"error: icacls could not restrict {path} (exit {proc.returncode})")
    else:
        os.chmod(path, 0o700)


def _set_mode(path: Path, mode: int) -> None:
    if os.name != "nt":  # Windows: the file inherits the directory's single-user ACL
        os.chmod(path, mode)


def _restrict_file(path: Path) -> None:
    _set_mode(path, PRIVATE_MODE)


def _quote(value: str) -> str:
    if any(c in value for c in '"\\\n\r'):
        raise ValueError("a generated value contains a character e2e.env cannot hold")
    return '"' + value.replace("$", "$$") + '"'


def _write_env(path: Path, values: dict[str, str]) -> None:
    """Write ``path`` atomically (a temporary name in the same directory, then a rename)."""
    lines = ["# E2E compose interpolation values from python -m scripts.test.gen_e2e_env",
             "# TEST ONLY: fresh per run, never committed and never printed"]
    lines += [f"{name}={_quote(value)}" for name, value in sorted(values.items())]
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(("\n".join(lines) + "\n").encode("utf-8"))
        _restrict_file(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------- certificates


def _certs_command(value: str | None) -> list[str]:
    if value is None:
        return [sys.executable, "-m", "scripts.db.gen_certs"]
    tokens = shlex.split(value, posix=os.name != "nt")
    if os.name == "nt":  # non-POSIX shlex keeps the quotes that protect a path with spaces
        tokens = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]
    if not tokens:
        raise _Exit(2, "error: --certs-cmd is empty")
    return tokens


def _run_certs(command: list[str], out: Path, san: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH")) if p)
    try:
        proc = subprocess.run([*command, "--out", str(out), "--san", san], cwd=ROOT, env=env,  # noqa: S603
                              capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=CERTS_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _Exit(1, f"error: the certificate command failed for --san {san}: {type(exc).__name__}") from None
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-5:])  # gen_certs prints file names only
        raise _Exit(1, f"error: the certificate command exited {proc.returncode} for --san {san}:\n{tail}")


def _write_certs(command: list[str], secrets_dir: Path, *, strict: bool) -> list[str]:
    """The mailpit trio (both profiles) and, in strict, the Postgres TLS files; returns the file names."""
    stage = secrets_dir / f".mailpit-{secrets.token_hex(4)}"
    try:
        _run_certs(command, stage, "mailpit")
        for produced, final in MAILPIT_FILES.items():
            if not (stage / produced).is_file():
                raise _Exit(1, f"error: the certificate command wrote no {produced} for --san mailpit")
            os.replace(stage / produced, secrets_dir / final)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    names = list(MAILPIT_FILES.values())
    if strict:
        _run_certs(command, secrets_dir, "postgres")
        absent = [name for name in POSTGRES_FILES if not (secrets_dir / name).is_file()]
        if absent:
            raise _Exit(1, f"error: the certificate command wrote no {', '.join(absent)} for --san postgres")
        names += POSTGRES_FILES
    for name in names:
        _set_mode(secrets_dir / name, FILE_MODES[name])
    return sorted(names)


# --------------------------------------------------------------------------------------------- main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.test.gen_e2e_env",
        description="Write per-run E2E secrets: DIR/e2e.env (compose --env-file) and DIR/secrets/. TEST ONLY.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--environment", required=True, choices=("development", "staging", "production"))
    parser.add_argument("--out", required=True, type=Path, metavar="DIR")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, metavar="PATH",
                        help="the IF-02 registry (default: deploy/env/env-contract.yaml)")
    parser.add_argument("--rotate", action="store_true", help="replace an existing run in DIR with fresh values")
    parser.add_argument("--certs-cmd", default=None, metavar="CMD",
                        help="test seam: run CMD --out D --san NAME instead of python -m scripts.db.gen_certs")
    return parser


def _run(args: argparse.Namespace) -> int:
    strict = args.profile == "strict"
    rows = _rows(args.registry)
    sources = _secret_sources(rows)
    missing = [name for name in sources if rule_for(name) is None]
    if missing:
        raise _Exit(2, "\n".join(f"no format rule for {name} (add one to scripts/test/gen_e2e_env.py)"
                                 for name in missing))
    command = _certs_command(args.certs_cmd)

    out: Path = args.out.resolve()
    env_path, secrets_dir = out / ENV_FILE, out / SECRETS_DIR
    values: dict[str, str] = {
        "ENVIRONMENT": args.environment,
        "E2E_SMTP_TLS_MODE": "starttls" if strict else "none",
        "SOLACE_SECRETS_DIR": secrets_dir.as_posix(),
        **LITERALS,
    }
    if strict:
        values["E2E_SSL_CERT_FILE"] = f"{CONTAINER_SECRETS}/{E2E_CA_FILE}"
        # before any write: a strict source without a value stops the run with nothing on disk
        operator = _strict_operator_values(rows, {**values, **dict.fromkeys(sources, "")})
    else:
        operator = {}

    if out.exists() and not out.is_dir():
        raise _Exit(1, f"error: --out {out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()) and not (out / RUN_MARKER).is_file():
        raise _Exit(1, f"refusing to use {out}: it is not empty and not an e2e run dir (no {RUN_MARKER}); "
                       "--out must be a new, empty or earlier run dir")
    held = env_path.exists() or (secrets_dir.is_dir() and any(secrets_dir.iterdir()))
    if held and not args.rotate:
        raise _Exit(1, f"refusing to overwrite the run in {out}; pass --rotate to replace it")

    out.mkdir(parents=True, exist_ok=True)
    _restrict_dir(out)
    marker = out / RUN_MARKER
    marker.write_text("solace-e2e run dir (python -m scripts.test.gen_e2e_env); the e2e harness deletes it\n",
                      encoding="utf-8")
    _restrict_file(marker)
    if args.rotate and secrets_dir.exists():
        shutil.rmtree(secrets_dir)
    secrets_dir.mkdir(exist_ok=True)
    if os.name != "nt":
        os.chmod(secrets_dir, 0o700)
    files = _write_certs(command, secrets_dir, strict=strict)

    for name in sources:
        if name in values:
            continue
        value = _generate(rule_for(name) or "unset", name, values)
        if value is not None:
            values[name] = value
    if "REDIS_PASSWORD" in values:
        values.setdefault("REDIS_URL", DERIVED["REDIS_URL"][1].format(values["REDIS_PASSWORD"]))
    values.update(operator)

    _write_env(env_path, values)
    print(f"gen_e2e_env: wrote {ENV_FILE} ({len(values)} variables): {', '.join(sorted(values))}", flush=True)
    print(f"gen_e2e_env: wrote {SECRETS_DIR}/: {', '.join(files)}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help (0) or a usage error (2), already reported on stderr
        return exc.code if isinstance(exc.code, int) else 2
    if args.environment not in PROFILES[args.profile]:
        _note(f"error: profile {args.profile} runs with --environment {' or '.join(PROFILES[args.profile])}, "
              f"not {args.environment}")
        return 2
    try:
        return _run(args)
    except _Exit as exc:
        _note(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
