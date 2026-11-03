"""MVP-271 (safety part, R-02.10): safety never opens a Postgres pool outside the TLS floor."""
import ast
from pathlib import Path

import pytest
import services.safety_service.src.infrastructure.database as db

SRC = Path(__file__).resolve().parents[1] / "src"

_POOLISH = {"create_pool", "create_async_engine", "create_engine"}
_SA_MODULES = {"sqlalchemy", "sqlalchemy.ext.asyncio"}

def _offenders(source):
    """Line numbers of constructs that open a private Postgres pool/engine/connection."""
    tree = ast.parse(source)
    asyncpg_names = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                     for a in n.names if a.name == "asyncpg"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            if (node.module == "asyncpg" and names & {"create_pool", "connect"}) or \
               (node.module in _SA_MODULES and names & _POOLISH):
                hits.append(node.lineno)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and (
                    f.attr in _POOLISH
                    or (f.attr == "connect" and isinstance(f.value, ast.Name) and f.value.id in asyncpg_names)):
                hits.append(node.lineno)
            elif isinstance(f, ast.Name) and f.id in _POOLISH:
                hits.append(node.lineno)
    return hits

_SNIPPETS = {
    "attr_create_pool": "import asyncpg\nasyncpg.create_pool()",
    "bare_create_pool": "from asyncpg import create_pool\ncreate_pool()",
    "aliased_create_pool": "from asyncpg import create_pool as cp\ncp()",
    "asyncpg_connect": "import asyncpg\nasyncpg.connect()",
    "aliased_module_connect": "import asyncpg as x\nx.connect()",
    "from_import_connect": "from asyncpg import connect\n",
    "sa_attr_engine": "import sqlalchemy\nsqlalchemy.create_engine('x')",
    "sa_async_attr": "import sqlalchemy.ext.asyncio as sa\nsa.create_async_engine('x')",
    "sa_bare_async": "from sqlalchemy.ext.asyncio import create_async_engine\ncreate_async_engine('x')",
    "sa_aliased": "from sqlalchemy import create_engine as ce\nce('x')",
}

@pytest.mark.parametrize("name", sorted(_SNIPPETS))
def test_mvp_271_guard_predicate_catches(name):
    assert _offenders(_SNIPPETS[name]) != []

def test_mvp_271_guard_predicate_ignores_innocent_code():
    assert _offenders("import asyncpg\nx = 'create_pool'  # create_pool()\nconn.connect()\n") == []

def test_mvp_271_safety_never_calls_create_pool():
    offenders = [f"{py.relative_to(SRC)}:{n}" for py in SRC.rglob("*.py")
                 for n in _offenders(py.read_text(encoding="utf-8"))]
    assert offenders == []

class _Pool:
    closed = False
    async def close(self): self.closed = True

async def test_mvp_271_repository_adopts_the_injected_pool(monkeypatch):
    async def _forbidden(*args, **kwargs): raise AssertionError("private pool opened")
    monkeypatch.setattr(db.asyncpg, "create_pool", _forbidden, raising=False)
    pool = _Pool()
    repo = db.ContraindicationRepository(pool=pool)
    assert await repo.connect() is True and repo.is_connected
    await repo.disconnect()
    assert pool.closed is False and repo.is_connected is False
    assert await db.ContraindicationRepository().connect() is False
