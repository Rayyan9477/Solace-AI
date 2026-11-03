"""python -m scripts.test.gen_e2e_env: per-run E2E secrets (MVP-039, WS-05.4; extended by WS-05.13)."""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

REGISTRY = """
- {name: AUTH_SECRET_KEY, services: [user-service], secret: true, required_in_strict: true, dev_default: null, notes: "", retired: false}
- {name: ENCRYPTION_MASTER_KEY, services: [user-service], secret: true, required_in_strict: true, dev_default: null, notes: "", retired: false}
- {name: FERNET_FIELD_KEY, services: [user-service], secret: true, required_in_strict: true, dev_default: null, notes: "", retired: false}
- {name: BASIC_AUTH_HASH, services: [caddy], secret: true, required_in_strict: true, dev_default: null, notes: "", retired: false}
- {name: ORCHESTRATOR_TURN_TIMEOUT_SECONDS, services: [orchestrator-service], secret: false, required_in_strict: false, dev_default: "20", notes: "", retired: false}
"""

# A stand-in for scripts.db.gen_certs: writes the three IF-22 file names into --out, no key material.
FAKE_CERTS = """
import pathlib, sys
args = sys.argv[1:]
out = pathlib.Path(args[args.index("--out") + 1])
san = args[args.index("--san") + 1]
out.mkdir(parents=True, exist_ok=True)
for name in ("postgres-ca.crt", "server.crt", "server.key"):
    (out / name).write_text(f"fake {name} for {san}\\n", encoding="utf-8")
"""


def _gen(tmp_path, *extra, registry=REGISTRY, profile="dev", environment="development"):
    reg = tmp_path / "registry.yaml"; reg.write_text(registry)
    return subprocess.run([sys.executable, "-m", "scripts.test.gen_e2e_env", "--profile", profile, "--environment", environment,  # noqa: S603
                           "--out", str(tmp_path / "run"), "--registry", str(reg), *extra], capture_output=True, text=True, cwd=REPO)


def _env(tmp_path): return dict(l.split("=", 1) for l in (tmp_path / "run" / "e2e.env").read_text().splitlines() if "=" in l)


def test_mvp_039_generates_every_registry_secret_without_printing_values(tmp_path):
    r = _gen(tmp_path); env = _env(tmp_path)
    assert r.returncode == 0 and {"AUTH_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "FERNET_FIELD_KEY", "BASIC_AUTH_HASH"} <= env.keys()
    assert re.fullmatch(r"[0-9a-f]{64}", env["ENCRYPTION_MASTER_KEY"].strip('"'))
    assert len(base64.urlsafe_b64decode(env["FERNET_FIELD_KEY"].strip('"'))) == 32
    assert "$$2" in env["BASIC_AUTH_HASH"]                                            # bcrypt '$' escaped for compose
    assert all(v.strip('"') not in r.stdout for k, v in env.items() if k != "ORCHESTRATOR_TURN_TIMEOUT_SECONDS")
    # ...nor in their real form: the file escapes '$' as '$$', so also look for the value compose reads
    assert all(v.strip('"').replace("$$", "$") not in r.stdout + r.stderr
               for k, v in env.items() if k != "ORCHESTRATOR_TURN_TIMEOUT_SECONDS")


def test_mvp_039_unknown_secret_name_fails_loud(tmp_path):
    r = _gen(tmp_path, registry=REGISTRY + "- {name: NEW_VENDOR_TOKEN, services: [x], secret: true, required_in_strict: true, dev_default: null, notes: '', retired: false}\n")
    assert r.returncode == 2 and "no format rule for NEW_VENDOR_TOKEN" in r.stderr


def test_mvp_039_refuses_to_overwrite_without_rotate(tmp_path):
    assert _gen(tmp_path).returncode == 0
    assert _gen(tmp_path).returncode != 0
    first = _env(tmp_path)["AUTH_SECRET_KEY"]; assert _gen(tmp_path, "--rotate").returncode == 0
    assert _env(tmp_path)["AUTH_SECRET_KEY"] != first


def test_mvp_039_profiles_set_the_smtp_tls_mode(tmp_path):
    # dev -> E2E_SMTP_TLS_MODE=none; strict (gen_certs replaced through --certs-cmd) -> starttls and
    # E2E_SSL_CERT_FILE=/run/secrets/e2e-ca.crt
    fake = tmp_path / "fake_certs.py"
    fake.write_text(FAKE_CERTS, encoding="utf-8")
    certs_cmd = f'"{sys.executable}" "{fake}"'

    dev = tmp_path / "dev"
    dev.mkdir()
    assert _gen(dev, "--certs-cmd", certs_cmd).returncode == 0
    dev_env = _env(dev)
    assert dev_env["E2E_SMTP_TLS_MODE"].strip('"') == "none" and "E2E_SSL_CERT_FILE" not in dev_env
    assert dev_env["ENVIRONMENT"].strip('"') == "development"

    strict = tmp_path / "strict"
    strict.mkdir()
    r = _gen(strict, "--certs-cmd", certs_cmd, profile="strict", environment="production")
    assert r.returncode == 0, r.stderr
    env = _env(strict)
    assert env["E2E_SMTP_TLS_MODE"].strip('"') == "starttls"
    assert env["E2E_SSL_CERT_FILE"].strip('"') == "/run/secrets/e2e-ca.crt"
    assert env["ENVIRONMENT"].strip('"') == "production"
    secrets_dir = strict / "run" / "secrets"
    assert Path(env["SOLACE_SECRETS_DIR"].strip('"')) == secrets_dir
    # the Postgres TLS files (IF-22 names, SAN postgres) and the mailpit CA, certificate and key (SAN mailpit)
    assert {p.name for p in secrets_dir.iterdir()} == {"postgres-ca.crt", "server.crt", "server.key",
                                                       "e2e-ca.crt", "mailpit.crt", "mailpit.key"}
    assert "for postgres" in (secrets_dir / "server.crt").read_text() and "for mailpit" in (secrets_dir / "mailpit.crt").read_text()
    assert "for mailpit" in (secrets_dir / "e2e-ca.crt").read_text()
    assert (strict / "run" / RUN_MARKER).is_file()          # the marker the harness checks before deleting a run dir

    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    assert _gen(mismatch, "--certs-cmd", certs_cmd, profile="strict", environment="development").returncode == 2


RUN_MARKER = ".solace-e2e-run"


def _fake_certs_cmd(tmp_path) -> str:
    fake = tmp_path / "fake_certs.py"
    fake.write_text(FAKE_CERTS, encoding="utf-8")
    return f'"{sys.executable}" "{fake}"'


def test_mvp_039_a_strict_value_without_a_rule_stops_before_anything_is_written(tmp_path):
    # a required non-secret source with neither strict_default nor example: exit 2 and no key material on disk
    registry = REGISTRY + ("- {name: NEW_OPERATOR_SETTING, source: NEW_OPERATOR_SETTING, services: [x], secret: false, "
                           "required_in_strict: true, dev_default: null, notes: '', retired: false}\n")
    r = _gen(tmp_path, "--certs-cmd", _fake_certs_cmd(tmp_path), registry=registry, profile="strict",
             environment="production")
    assert r.returncode == 2 and "no format rule for NEW_OPERATOR_SETTING" in r.stderr
    assert not (tmp_path / "run").exists()


def test_mvp_039_refuses_a_foreign_non_empty_directory(tmp_path):
    # --out must be new, empty, or an earlier run (it carries the run marker); nothing in a foreign directory is
    # touched (no ACL change, no file), with or without --rotate
    run = tmp_path / "run"
    run.mkdir()
    (run / "notes.txt").write_text("operator data", encoding="utf-8")
    for extra in ((), ("--rotate",)):
        r = _gen(tmp_path, "--certs-cmd", _fake_certs_cmd(tmp_path), *extra)
        assert r.returncode == 1 and "not an e2e run dir" in r.stderr, r.stderr
        assert sorted(p.name for p in run.iterdir()) == ["notes.txt"]


class _PosixOs:
    """The generator's view of a POSIX host: os.name 'posix' and recorded chmod calls; the rest is the real os."""

    name = "posix"

    def __init__(self, modes: dict[str, int]) -> None:
        self._modes = modes

    def chmod(self, path, mode) -> None:
        self._modes[Path(path).name] = mode

    def __getattr__(self, attr):
        import os

        return getattr(os, attr)


def test_mvp_039_posix_modes_let_the_containers_read_what_they_mount(tmp_path, monkeypatch):
    # Compose file secrets are bind mounts that keep the host owner and mode. The app services and migrate run as
    # appuser (uid 1000) and read the CAs, so the CA and certificate files are 0644. The keys stay 0600 (owner: the
    # generating user): mailpit (image without USER) and postgres (its TLS wrapper copies the key before stepping
    # down) read them as root. The run dir and secrets/ are 0700; a mounted file is not reached through them.
    from scripts.test import gen_e2e_env

    modes: dict[str, int] = {}
    monkeypatch.setattr(gen_e2e_env, "os", _PosixOs(modes))
    fake = tmp_path / "fake_certs.py"
    fake.write_text(FAKE_CERTS, encoding="utf-8")
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    gen_e2e_env._write_certs([sys.executable, str(fake)], secrets_dir, strict=True)
    assert modes == {"e2e-ca.crt": 0o644, "mailpit.crt": 0o644, "postgres-ca.crt": 0o644, "server.crt": 0o644,
                     "mailpit.key": 0o600, "server.key": 0o600}


def test_mvp_039_posix_run_files_have_the_documented_modes(tmp_path):
    import os
    import stat

    if os.name == "nt":
        import pytest

        pytest.skip("file modes are POSIX; Windows restricts the directory ACL instead")
    assert _gen(tmp_path, "--certs-cmd", _fake_certs_cmd(tmp_path), profile="strict",
                environment="production").returncode == 0
    run = tmp_path / "run"
    mode = {p.relative_to(run).as_posix(): stat.S_IMODE(p.stat().st_mode) for p in run.rglob("*")}
    assert mode["secrets"] == 0o700 and mode["e2e.env"] == 0o600 and stat.S_IMODE(run.stat().st_mode) == 0o700
    assert mode["secrets/e2e-ca.crt"] == mode["secrets/postgres-ca.crt"] == mode["secrets/server.crt"] == 0o644
    assert mode["secrets/mailpit.crt"] == 0o644 and mode["secrets/mailpit.key"] == mode["secrets/server.key"] == 0o600
