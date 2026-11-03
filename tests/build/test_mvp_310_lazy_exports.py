"""MVP-269/310/312: importing a leaf module must not drag in Kong, jose, weaviate, alembic or the Kafka admin stack."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATEWAY_NAMES = {"KongConfig", "KongSettings", "ServiceConfig", "UpstreamConfig", "RouteConfig", "RouteManager",
                 "RouteDefinition", "ServiceRoutes", "RateLimitPolicy", "RateLimiter", "RateLimitConfig",
                 "RateLimitResult", "JWTAuthPlugin", "JWTConfig", "TokenClaims", "AuthResult", "CORSConfig",
                 "CORSPolicy", "CORSHandler"}
JOSE_BLOCKER = (
    "import sys, importlib.abc\n"
    "class _Block(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'jose' or name.startswith('jose.'):\n"
    "            raise ImportError('blocked: jose')\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Block())\n"
)


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "test"}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)


def _loaded(import_line: str, prefixes: tuple[str, ...]) -> list[str]:
    r = _run(f"import json, sys\n{import_line}\n"
             f"print(json.dumps(sorted(m for m in sys.modules if m.startswith({prefixes!r}))))")
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_mvp_310_rate_limiter_import_loads_no_kong_modules() -> None:
    assert _loaded("import infrastructure.api_gateway.rate_limiting",
                   ("jose", "httpx", "infrastructure.api_gateway.kong_config", "infrastructure.api_gateway.routes",
                    "infrastructure.api_gateway.auth_plugin", "infrastructure.api_gateway.cors")) == []


def test_mvp_312_rate_limiter_imports_without_python_jose() -> None:
    r = _run(JOSE_BLOCKER + "import infrastructure.api_gateway.rate_limiting\nprint('OK')")
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]


def test_mvp_269_postgres_import_loads_no_optional_backends() -> None:
    assert _loaded("import solace_infrastructure.postgres",
                   ("weaviate", "alembic", "aiokafka", "solace_infrastructure.kafka",
                    "solace_infrastructure.observability", "solace_infrastructure.weaviate",
                    "solace_infrastructure.redis", "solace_infrastructure.database")) == []


def test_mvp_269_package_import_is_free() -> None:
    assert _loaded("import solace_infrastructure",
                   ("solace_infrastructure.", "weaviate", "alembic", "aiokafka", "sqlalchemy", "asyncpg", "fastapi")) == []


def test_mvp_269_every_declared_export_still_resolves() -> None:
    r = _run("import solace_infrastructure as s\n"
             "missing = [n for n in s.__all__ if getattr(s, n, None) is None]\n"
             "assert {'PostgresClient', 'WeaviateClient', 'KafkaMonitor', 'MigrationRunner', 'DBBaseModel',\n"
             "        'ObsAlertSeverity', 'setup_redis'} <= set(s.__all__)\n"
             "print(missing)")
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[]"


def test_mvp_310_gateway_exports_unchanged_and_lazy() -> None:
    r = _run("import infrastructure.api_gateway as g\n"
             f"assert set(g.__all__) == {GATEWAY_NAMES!r}\n"
             "from infrastructure.api_gateway import RateLimiter, KongConfig, JWTAuthPlugin, CORSHandler\n"
             "print('OK')")
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]
