"""MVP-032/013/006/248/355: every MVP Dockerfile follows the IF-21 layout."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scripts.ci import dockerfiles
from scripts.ci.images import IMAGES

ROOT = Path(__file__).resolve().parents[2]
PY_BASE = re.compile(r"^python:3\.12-slim@sha256:[0-9a-f]{64}$")
APP_IMAGES = [i for i in IMAGES if i.module]


def _df(image):
    return dockerfiles.parse(ROOT / image.dockerfile)


@pytest.mark.parametrize("image", IMAGES, ids=lambda i: i.name)
def test_mvp_355_base_images_pinned_by_digest(image) -> None:
    refs = _df(image).from_images
    assert refs and all(PY_BASE.match(r) for r in refs), refs
    assert len(set(refs)) == 1
    assert re.search(r"COPY --from=ghcr\.io/astral-sh/uv:[0-9.]+@sha256:[0-9a-f]{64} ",
                     (ROOT / image.dockerfile).read_text(encoding="utf-8"))


@pytest.mark.parametrize("image", IMAGES, ids=lambda i: i.name)
def test_mvp_248_lock_synced_with_hashes_and_no_fallback(image) -> None:
    text = (ROOT / image.dockerfile).read_text(encoding="utf-8")
    assert "requirements-runtime.txt" not in text and "service-requirements.txt" not in text
    assert "COPY requirements.lock ./requirements.lock" in text
    assert re.search(r"uv pip sync [^\n]*--require-hashes requirements\.lock", text)
    # no install fallback anywhere; only the WS-01.6 HEALTHCHECK's `|| exit 1` (maps curl's codes to 1) is exempt
    assert "||" not in "\n".join(l for l in text.splitlines() if not l.lstrip().upper().startswith("HEALTHCHECK"))
    assert not re.search(r"\bpip install\b", text)


@pytest.mark.parametrize("image", IMAGES, ids=lambda i: i.name)
def test_mvp_032_every_copy_source_exists(image) -> None:
    for instr in _df(image).copies:
        if "from" in instr.flags:
            continue
        for src in instr.sources:
            assert (ROOT / src).exists(), f"{image.dockerfile}: COPY {src}"


@pytest.mark.parametrize("image", APP_IMAGES, ids=lambda i: i.name)
def test_mvp_013_images_carry_the_rate_limiter_package(image) -> None:
    sources = {s.rstrip("/") for c in _df(image).runtime_copies for s in c.sources}
    assert {"infrastructure/__init__.py", "infrastructure/api_gateway"} <= sources


@pytest.mark.parametrize("image", APP_IMAGES, ids=lambda i: i.name)
def test_mvp_006_cmd_runs_the_registered_module(image) -> None:
    df = _df(image)
    assert df.cmd[:3] == ["python", "-m", "uvicorn"] and f"{image.module}:app" in df.cmd
    if image.merged_src:
        svc_src = Path(image.dockerfile).parent.as_posix() + "/src/"
        assert any(c.sources == (svc_src,) and c.dest == "./src/" for c in df.runtime_copies)


@pytest.mark.parametrize("image", IMAGES, ids=lambda i: i.name)
def test_mvp_032_runtime_runs_as_uid_1000_from_app(image) -> None:
    df = _df(image)
    assert df.user == "appuser" and df.workdir == "/app"
    assert df.env["PYTHONPATH"].split(":")[:2] == ["/app", "/app/src"]
    assert "--uid 1000" in (ROOT / image.dockerfile).read_text(encoding="utf-8")


def test_mvp_032_dockerignore_keeps_secrets_tests_and_caches_out() -> None:
    lines = {l.strip() for l in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    assert {".env", ".env.*", "!.env.example", ".git", "**/tests/", "tests/", "src/models/", ".secrets", "secrets"} <= lines


def test_mvp_032_dockerignore_drops_secrets_but_keeps_package_data() -> None:
    ignore = ROOT / ".dockerignore"
    for rel in (".secrets/AUTH_SECRET_KEY", "secrets/postgres-ca.crt", ".env", "services/safety_service/tests/test_x.py"):
        assert dockerfiles.dockerignored(rel, ignore), rel
    for rel in ("src/solace_clinical/crisis_lexicon_v1.yaml", "src/solace_events/schemas.py", ".env.example",
                "scripts/db/migrate.py"):
        assert not dockerfiles.dockerignored(rel, ignore), rel


def test_mvp_032_user_service_image_carries_scripts() -> None:
    df = _df(next(i for i in IMAGES if i.name == "user-service"))
    assert any(c.sources == ("scripts/",) and c.dest == "./scripts/" for c in df.runtime_copies)


def test_mvp_032_migrate_image_carries_schema_tooling() -> None:
    df = dockerfiles.parse(ROOT / "docker" / "migrate.Dockerfile")
    sources = {s.rstrip("/") for c in df.runtime_copies for s in c.sources}
    assert {"alembic.ini", "migrations", "scripts/db", "src"} <= sources
    assert df.cmd == ["python", "-m", "scripts.db.migrate", "upgrade", "head"]
    assert df.entrypoint == ["/app/docker/entrypoint.sh"]


def test_mvp_112_build_command_uses_repo_root_context() -> None:
    from scripts.ci.build_images import build_command

    image = next(i for i in IMAGES if i.name == "user-service")
    assert build_command(image, "ci", ".") == [
        "docker", "build", "-f", "services/user-service/Dockerfile", "-t", "solace/user-service:ci", "."]


def test_mvp_112_compose_image_refs_come_from_the_builder() -> None:
    # WS-03.2 writes image_ref(<service>, COMPOSE_TAG) as each built compose service's image: value
    from scripts.ci.build_images import COMPOSE_TAG, build_command, image_ref

    image = next(i for i in IMAGES if i.name == "user-service")
    assert COMPOSE_TAG == "${SOLACE_IMAGE_TAG:-dev}"
    assert image_ref("user-service", COMPOSE_TAG) == "solace/user-service:${SOLACE_IMAGE_TAG:-dev}"
    assert build_command(image, "ci", ".")[5] == image_ref(image.name, "ci")


def test_ws_01_5_gitattributes_pins_shell_scripts_to_lf() -> None:
    # core.autocrlf=true would otherwise check the Postgres init script out with CRLF and break it
    out = subprocess.run(["git", "check-attr", "eol", "--", "infrastructure/init-db/01-roles.sh"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.strip().endswith("eol: lf"), out
    assert "*.sh text eol=lf" in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
