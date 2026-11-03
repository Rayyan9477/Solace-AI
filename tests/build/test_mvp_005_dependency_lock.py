"""MVP-005/012/312/358/359: one hash-pinned lock that resolves, matches the tested versions and ships no known-bad pins."""
from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
LOCK, DEV_LOCK = ROOT / "requirements.lock", ROOT / "requirements-dev.lock"
ACCEPTED = ROOT / "scripts" / "ci" / "accepted_vulns.yaml"
CHECKPOINT_PATH = frozenset({"langgraph", "langgraph-checkpoint", "langchain-core"})
MVP_SERVICES = ["orchestrator_service", "safety_service", "therapy_service", "diagnosis_service",
                "memory_service", "personality_service", "user-service", "notification-service",
                "analytics-service"]
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?==([^\s;\\]+)")
# top-level import name -> candidate distributions (any one in the lock satisfies it)
IMPORT_TO_DIST = {
    "fastapi": {"fastapi"}, "starlette": {"starlette"}, "uvicorn": {"uvicorn"}, "pydantic": {"pydantic"},
    "pydantic_settings": {"pydantic-settings"}, "structlog": {"structlog"}, "httpx": {"httpx"},
    "asyncpg": {"asyncpg"}, "sqlalchemy": {"sqlalchemy"}, "alembic": {"alembic"}, "redis": {"redis"},
    "aiokafka": {"aiokafka"}, "cryptography": {"cryptography"}, "jwt": {"pyjwt"},
    "weaviate": {"weaviate-client"}, "bm25s": {"bm25s"}, "langgraph": {"langgraph"},
    "langchain_core": {"langchain-core"}, "langchain_anthropic": {"langchain-anthropic"},
    "portkey_ai": {"portkey-ai"}, "email_validator": {"email-validator"}, "argon2": {"argon2-cffi"},
    "passlib": {"passlib"}, "jinja2": {"jinja2"}, "aiosmtplib": {"aiosmtplib"}, "flashtext": {"flashtext"},
    "opentelemetry": {"opentelemetry-api"}, "prometheus_fastapi_instrumentator": {"prometheus-fastapi-instrumentator"},
    "prometheus_client": {"prometheus-client"}, "typing_extensions": {"typing-extensions"},
    "psycopg": {"psycopg"}, "psycopg_pool": {"psycopg-pool"}, "pyotp": {"pyotp"}, "yaml": {"pyyaml"},
    "boto3": {"boto3"}, "botocore": {"botocore"},  # D-04 Bedrock Converse provider (WS-09)
}
EXCLUDED = {  # imported somewhere, deliberately not in any image
    "torch": "D-16", "transformers": "D-16", "vaderSentiment": "D-16", "spacy": "D-16",
    "sentence_transformers": "D-16", "clickhouse_connect": "D-19 ClickHouse out of scope",
    "google": "push/FCM out of scope", "jose": "Kong out of scope (D-19); dev lock only",
    "aiohttp": "dead path in solace_infrastructure.kafka.schemas (MVP-269)",
}
RUNTIME_DIRS = ["src", "services", "migrations", "scripts/db", "infrastructure/api_gateway"]
LOCAL_MISSING_OK_ROOTS: set[str] = set()  # D-20: the venv is re-synced to the full dev lock (portkey-ai included)


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PIN.match(line)
        if m:
            pins.setdefault(canonicalize_name(m.group(1)), set()).add(m.group(2))
    forks = {k: v for k, v in pins.items() if len(v) > 1}
    assert not forks, f"forked pins: {forks}"
    return {k: next(iter(v)) for k, v in pins.items()}


def _service_names(svc: str) -> list[str]:
    names = []
    for raw in (ROOT / "services" / svc / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            assert not re.search(r"[=<>~!@]", line), f"{svc}: version specifier in {line!r}"
            names.append(canonicalize_name(re.split(r"[\[\s]", line, 1)[0]))
    return names


def test_mvp_005_lock_files_are_uv_hash_pinned() -> None:
    for path, src in ((LOCK, "requirements.in"), (DEV_LOCK, "requirements-dev.in")):
        text = path.read_text(encoding="utf-8")
        assert f"uv pip compile {src} --universal --generate-hashes --python-version 3.12 -o {path.name}" in text
        blocks = re.split(r"\n(?=[A-Za-z0-9])", text.split("\n", 1)[1])
        reqs = [b for b in blocks if PIN.match(b)]
        assert reqs and all("--hash=sha256:" in b for b in reqs)


def test_mvp_358_lock_pins_the_tested_baseline() -> None:
    pins = _pins(LOCK)
    # WS-01.F2 (D-16 security bumps): the lowest release that fixes each checkpoint-path advisory
    assert pins["langgraph"] == "1.0.10"  # PYSEC-2026-83
    assert pins["langgraph-checkpoint"] == "4.1.1"  # PYSEC-2026-2573 (4.1.1), PYSEC-2026-2574 (4.0.0)
    assert pins["langchain-core"] == "1.3.3"  # PYSEC-2026-2564
    assert pins["fastapi"] == "0.133.0"  # lowest fastapi without a starlette<1 cap
    assert pins["starlette"] == "1.3.1"  # PYSEC-2026-161, -248, -249, -2280, -2281
    assert pins["pydantic-settings"] == "2.14.2"  # GHSA-4xgf-cpjx-pc3j
    assert pins["asyncpg"] == "0.30.0"
    assert pins["pydantic"].startswith("2.13.")
    assert pins["redis"].startswith("5.3.")
    assert pins["weaviate-client"].startswith("4.19.")
    assert Version(pins["ormsgpack"]) >= Version("1.12.0")
    assert Version(pins["langgraph-checkpoint-postgres"]) >= Version("3.0.0")


def test_mvp_358_checkpoint_path_has_no_accepted_advisories() -> None:
    # WS-01.F2: the checkpoint deserialization/RCE advisories are fixed by a pin bump, never accepted
    entries = (yaml.safe_load(ACCEPTED.read_text(encoding="utf-8")) or {}).get("accepted") or []
    held = sorted((canonicalize_name(str(e["target"])), str(e["id"])) for e in entries
                  if canonicalize_name(str(e["target"])) in CHECKPOINT_PATH)
    assert held == [], f"checkpoint-path advisories accepted instead of fixed: {held}"


def test_mvp_358_checkpoint_postgres_is_exact_pinned_for_slot_017() -> None:
    # WS-10: slot 017 embeds this version's MIGRATIONS; a bump needs a slot-017 follow-up revision
    text = (ROOT / "requirements.in").read_text(encoding="utf-8")
    m = re.search(r"^langgraph-checkpoint-postgres==([0-9.]+)\s+#.*slot-017", text, re.M)
    assert m, "langgraph-checkpoint-postgres must be pinned with == and a slot-017 comment"
    assert _pins(LOCK)["langgraph-checkpoint-postgres"] == m.group(1)


def test_mvp_359_312_security_floor() -> None:
    pins = _pins(LOCK)
    assert Version(pins["pyjwt"]) >= Version("2.13.0")
    for banned in ("python-jose", "ecdsa", "torch", "transformers", "vadersentiment", "spacy", "sentence-transformers"):
        assert banned not in pins, banned
    dev = _pins(DEV_LOCK)
    assert Version(dev.get("python-jose", "3.5.0")) >= Version("3.4.0")


def test_mvp_005_required_runtime_packages_are_locked() -> None:
    pins = _pins(LOCK)
    for name in ("email-validator", "aiokafka", "langgraph-checkpoint-postgres", "psycopg", "psycopg-pool",
                 "pyotp", "prometheus-client", "alembic", "sqlalchemy", "aiosmtplib", "jinja2", "argon2-cffi",
                 "passlib", "bcrypt", "bm25s", "weaviate-client", "portkey-ai", "uvicorn", "structlog",
                 "httpx", "cryptography", "pyyaml", "flashtext", "boto3", "botocore",
                 "cffi", "pycparser"):  # cffi/pycparser: hashed for WS-03.8's sops tools image (D-07)
        assert name in pins, name


def test_d04_bedrock_sdk_is_an_exact_runtime_pin() -> None:
    # D-04: the version is the one installed in the anaconda env (pip show), unless an advisory forces a bump
    text, pins = (ROOT / "requirements.in").read_text(encoding="utf-8"), _pins(LOCK)
    for name in ("boto3", "botocore"):
        m = re.search(rf"^{name}==([0-9.]+)\s+#.*D-04", text, re.M)
        assert m, f"{name} must be pinned with == and a D-04 comment in requirements.in"
        assert pins[name] == m.group(1), name


def test_mvp_005_dev_lock_contains_runtime_lock_at_same_versions() -> None:
    runtime, dev = _pins(LOCK), _pins(DEV_LOCK)
    assert {k: (v, dev.get(k)) for k, v in runtime.items() if dev.get(k) != v} == {}
    for name in ("pytest", "pytest-asyncio", "pytest-timeout", "hypothesis", "testcontainers", "docker",
                 "websockets", "fakeredis", "ruff", "pip-audit", "packaging", "jsonschema", "boto3", "cfn-lint"):
        assert name in dev, name


@pytest.mark.parametrize("svc", MVP_SERVICES)
def test_mvp_005_service_requirements_are_unpinned_lock_members(svc: str) -> None:
    pins = _pins(LOCK)
    for name in _service_names(svc):
        assert name in pins, f"{svc}: {name}"
        assert not name.startswith("pytest"), f"{svc}: test tool {name}"


def test_mvp_012_notification_extras_include_email_validator() -> None:
    assert "email-validator" in _service_names("notification-service")


def test_mvp_005_root_requirements_points_to_dev_lock() -> None:
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert lines == ["-r requirements-dev.lock"]


def test_mvp_359_pyproject_declares_no_competing_pins() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project.get("dependencies", []) == []
    assert "optional-dependencies" not in project


def _is_local(name: str) -> bool:
    candidates = [ROOT / name, ROOT / "src" / name, ROOT / "services" / name]
    candidates += [d / "src" / name for d in (ROOT / "services").iterdir() if d.is_dir()]
    return any(c.exists() or c.with_suffix(".py").exists() for c in candidates) or name.startswith("solace_")


def test_mvp_005_runtime_imports_resolve_from_lock() -> None:
    pins, unknown, missing = _pins(LOCK), {}, {}
    for base in RUNTIME_DIRS:
        for path in (ROOT / base).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if ("/tests/" in rel or path.name.startswith("test_")
                    or rel.startswith(("services/config_service/", "src/solace_testing/"))):
                continue  # test code, the frozen config-service and the dev-only solace_testing package
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
                for top in {n.split(".")[0] for n in names}:
                    if top in sys.stdlib_module_names or _is_local(top) or top in EXCLUDED:
                        continue
                    dists = IMPORT_TO_DIST.get(top)
                    if dists is None:
                        unknown.setdefault(top, rel)
                    elif not dists & set(pins):
                        missing.setdefault(top, rel)
    assert unknown == {}, f"map these imports in IMPORT_TO_DIST or EXCLUDED: {unknown}"
    assert missing == {}, f"imported but not locked: {missing}"


def test_d16_local_interpreter_matches_dev_lock() -> None:
    from scripts.ci.lock_delta import compute_delta, exclusive_closure, parse_lock

    entries = parse_lock(DEV_LOCK)
    delta = compute_delta(DEV_LOCK, allow_missing=frozenset(exclusive_closure(entries, LOCAL_MISSING_OK_ROOTS)))
    assert delta.mismatched == {}, f"installed version differs from the lock: {delta.mismatched}"
    assert delta.missing == {}, f"locked but not installed (D-20 venv sync list): {delta.missing}"


def test_d20_clean_venv_is_built_only_from_the_dev_lock(tmp_path: Path) -> None:
    from scripts.ci import clean_venv

    venv = tmp_path / "venv-clean"
    cmds = clean_venv.commands(base_python="base-py", uv="uv", venv=venv, lock=DEV_LOCK)
    assert cmds == [["base-py", "-m", "venv", "--clear", str(venv)],
                    ["uv", "pip", "sync", "--python", str(clean_venv.venv_python(venv)), "--require-hashes", str(DEV_LOCK)]]
    assert not any("system-site-packages" in part for cmd in cmds for part in cmd)
    assert clean_venv.VENV_CLEAN == Path(r"D:\tmp\mvp-exec\venv-clean")
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_text("home = C:\\base\ninclude-system-site-packages = true\n", encoding="utf-8")
    assert clean_venv.is_isolated(cfg) is False
    cfg.write_text("home = C:\\base\ninclude-system-site-packages = false\n", encoding="utf-8")
    assert clean_venv.is_isolated(cfg) is True
