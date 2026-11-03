"""Static (AST) scan of the environment variable names that code reads.

Collected forms:

- ``os.getenv("X")`` / ``getenv("X")``, ``os.environ.get("X")`` / ``environ.get("X")``,
  ``os.environ["X"]`` (load context) and ``"X" in os.environ``;
- pydantic-settings classes: for every class whose body assigns
  ``model_config = SettingsConfigDict(env_prefix="P", ...)`` (or a pydantic v1 style
  ``class Config: env_prefix = "P"``), ``P + FIELD.upper()`` for each annotated field, plus any
  literal ``alias`` / ``validation_alias`` / ``AliasChoices(...)`` names on the field.

Any directory named ``tests`` is skipped. Only string literals are resolved; dynamic names are
invisible to this scan.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Sequence

SKIP_DIRS = frozenset({"tests", "__pycache__", ".venv", "venv", "node_modules", ".git"})


def _is_os_attr(node: ast.AST, attr: str) -> bool:
    """True for ``os.<attr>`` or a bare ``<attr>`` name (``from os import environ``)."""
    if isinstance(node, ast.Attribute):
        return node.attr == attr and isinstance(node.value, ast.Name) and node.value.id == "os"
    return isinstance(node, ast.Name) and node.id == attr


def _str_const(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def _settings_prefix(cls: ast.ClassDef) -> str | None:
    """The env prefix of a pydantic-settings class, or None when the class is not one."""
    for stmt in cls.body:
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        if any(isinstance(t, ast.Name) and t.id == "model_config" for t in targets):
            if isinstance(value, ast.Call) and _call_name(value) == "SettingsConfigDict":
                for kw in value.keywords:
                    if kw.arg == "env_prefix":
                        return _str_const(kw.value) or ""
                return ""
        if isinstance(stmt, ast.ClassDef) and stmt.name == "Config":
            for inner in stmt.body:
                if isinstance(inner, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "env_prefix" for t in inner.targets
                ):
                    return _str_const(inner.value) or ""
    return None


def _field_aliases(value: ast.AST | None) -> Iterator[str]:
    if not isinstance(value, ast.Call) or _call_name(value) != "Field":
        return
    for kw in value.keywords:
        if kw.arg not in ("alias", "validation_alias"):
            continue
        literal = _str_const(kw.value)
        if literal:
            yield literal
        elif isinstance(kw.value, ast.Call) and _call_name(kw.value) == "AliasChoices":
            for arg in kw.value.args:
                choice = _str_const(arg)
                if choice:
                    yield choice


def _settings_fields(cls: ast.ClassDef, prefix: str) -> Iterator[tuple[str, int]]:
    for stmt in cls.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        field = stmt.target.id
        if field == "model_config" or field.startswith("_"):
            continue
        annotation = ast.unparse(stmt.annotation)
        if annotation.startswith(("ClassVar", "typing.ClassVar")):
            continue
        yield prefix + field.upper(), stmt.lineno
        for alias in _field_aliases(stmt.value):
            yield alias, stmt.lineno


def _names_read(tree: ast.AST) -> Iterator[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            first = _str_const(node.args[0]) if node.args else None
            if first is None:
                continue
            if _is_os_attr(func, "getenv"):
                yield first, node.lineno
            elif isinstance(func, ast.Attribute) and func.attr == "get" and _is_os_attr(func.value, "environ"):
                yield first, node.lineno
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if _is_os_attr(node.value, "environ"):
                key = _str_const(node.slice)
                if key:
                    yield key, node.lineno
        elif isinstance(node, ast.Compare):
            key = _str_const(node.left)
            if key and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                if any(_is_os_attr(c, "environ") for c in node.comparators):
                    yield key, node.lineno
        elif isinstance(node, ast.ClassDef):
            prefix = _settings_prefix(node)
            if prefix is not None:
                yield from _settings_fields(node, prefix)


def _python_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        yield path


def collect_env_readers(roots: Sequence[Path]) -> dict[str, list[str]]:
    """Map each env name read under ``roots`` to the sorted ``path:line`` locations that read it."""
    readers: dict[str, set[str]] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in _python_files(root):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError, ValueError):
                continue
            for name, line in _names_read(tree):
                readers.setdefault(name, set()).add(f"{path.as_posix()}:{line}")
    return {name: sorted(locs) for name, locs in sorted(readers.items())}
