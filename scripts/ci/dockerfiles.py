"""IF-21: a small Dockerfile reader plus a no-Docker replica of the runtime image layout. Stdlib only."""
from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Copy:
    flags: dict[str, str]
    sources: tuple[str, ...]
    dest: str
    stage: int = 0
    workdir: str = "/"


@dataclass
class Dockerfile:
    from_images: list[str] = field(default_factory=list)
    copies: list[Copy] = field(default_factory=list)
    runtime_copies: list[Copy] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cmd: list[str] = field(default_factory=list)
    entrypoint: list[str] = field(default_factory=list)
    healthcheck: str | None = None
    user: str | None = None
    workdir: str = "/"


def _logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    buf = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith("#")):
            continue
        if buf and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        lines.append((buf + stripped).strip())
        buf = ""
    if buf.strip():
        lines.append(buf.strip())
    return lines


def _exec_form(arg: str) -> list[str]:
    arg = arg.strip()
    if arg.startswith("["):
        return [str(x) for x in json.loads(arg)]
    return shlex.split(arg)


def _parse_env(arg: str) -> dict[str, str]:
    tokens = shlex.split(arg)
    if tokens and "=" not in tokens[0]:  # legacy `ENV KEY value`
        return {tokens[0]: " ".join(tokens[1:])}
    out: dict[str, str] = {}
    for tok in tokens:
        key, _, value = tok.partition("=")
        out[key] = value
    return out


def parse(path: Path | str) -> Dockerfile:
    """Parse a Dockerfile; `runtime_*` fields, env, cmd, user and workdir describe the final stage."""
    df = Dockerfile()
    stage = -1
    for line in _logical_lines(Path(path).read_text(encoding="utf-8")):
        instr, _, arg = line.partition(" ")
        instr = instr.upper()
        if instr == "FROM":
            stage += 1
            tokens = [t for t in arg.split() if not t.startswith("--")]
            df.from_images.append(tokens[0])
            df.runtime_copies, df.env, df.cmd, df.entrypoint = [], {}, [], []
            df.healthcheck, df.user, df.workdir = None, None, "/"
        elif instr in ("COPY", "ADD"):
            flags: dict[str, str] = {}
            rest: list[str] = []
            for tok in shlex.split(arg):
                if tok.startswith("--") and not rest:
                    key, _, value = tok[2:].partition("=")
                    flags[key] = value
                else:
                    rest.append(tok)
            item = Copy(flags, tuple(rest[:-1]), rest[-1], stage, df.workdir)
            df.copies.append(item)
            if "from" not in flags:
                df.runtime_copies.append(item)
        elif instr == "ENV":
            df.env.update(_parse_env(arg))
        elif instr == "CMD":
            df.cmd = _exec_form(arg)
        elif instr == "ENTRYPOINT":
            df.entrypoint = _exec_form(arg)
        elif instr == "HEALTHCHECK":
            df.healthcheck = None if arg.strip().upper() == "NONE" else arg
        elif instr == "USER":
            df.user = arg.strip().split(":")[0]
        elif instr == "WORKDIR":
            target = arg.strip()
            df.workdir = target if target.startswith("/") else posixpath.join(df.workdir, target)
    return df


_BIND_OPTIONS = frozenset({"--host", "--port"})


# Options of uvicorn 0.40's CLI that take a value (checked against uvicorn.main.main.params); every other
# `--option` is a boolean flag and takes no value.
_VALUE_OPTIONS = frozenset({
    "--host", "--port", "--uds", "--fd", "--loop", "--http", "--ws", "--ws-max-size", "--ws-max-queue",
    "--ws-ping-interval", "--ws-ping-timeout", "--ws-per-message-deflate", "--lifespan", "--interface",
    "--reload-dir", "--reload-include", "--reload-exclude", "--reload-delay", "--workers", "--env-file",
    "--log-config", "--log-level", "--header", "--forwarded-allow-ips", "--root-path", "--limit-concurrency",
    "--backlog", "--limit-max-requests", "--timeout-keep-alive",
    "--timeout-graceful-shutdown", "--timeout-worker-healthcheck", "--ssl-keyfile", "--ssl-certfile",
    "--ssl-keyfile-password", "--ssl-version", "--ssl-cert-reqs", "--ssl-ca-certs", "--ssl-ciphers",
    "--app-dir", "--h11-max-incomplete-event-size",
})


def uvicorn_flags(source: Dockerfile | list[str]) -> list[str]:
    """The `--option [value]` arguments of the uvicorn invocation (a Dockerfile's ENTRYPOINT plus CMD, or an argv
    list), values kept, minus the bind address (`--host`, `--port`), so a caller can serve on its own socket with
    the image's server settings. A value-taking option always takes the next argument, whatever its shape
    (`--header server:x`); every other option is a boolean flag and takes none, so one placed before the app
    (`--proxy-headers main:app.asgi`) never swallows the app."""
    args = list(source) if isinstance(source, list) else [*source.entrypoint, *source.cmd]
    out: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if not arg.startswith("--"):
            continue
        option = [arg]
        if "=" not in arg and i < len(args) and arg in _VALUE_OPTIONS:
            option.append(args[i])
            i += 1
        if arg.split("=", 1)[0] not in _BIND_OPTIONS:
            out += option
    return out


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        elif pattern[i] == "[":
            # A `[...]` class (`^` negates, `a-z` ranges, `\` escapes) goes into the regex as a class, as Docker's
            # matcher does. An empty or unterminated class, which Docker rejects, stays a literal `[`.
            j, cls = i + 1, "["
            if pattern.startswith("^", j):
                j, cls = j + 1, "[^"
            start = j
            while j < len(pattern) and pattern[j] != "]":
                if pattern[j] == "\\" and j + 1 < len(pattern):
                    cls += re.escape(pattern[j + 1])
                    j += 2
                else:
                    cls += "-" if pattern[j] == "-" else re.escape(pattern[j])
                    j += 1
            if j >= len(pattern) or j == start:
                out += re.escape("[")
                i += 1
            else:
                out += cls + "]"
                i = j + 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


def _load_patterns(dockerignore: Path) -> list[tuple[bool, re.Pattern[str]]]:
    patterns: list[tuple[bool, re.Pattern[str]]] = []
    for raw in Path(dockerignore).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
        line = posixpath.normpath(line.lstrip("/"))
        patterns.append((negated, _pattern_regex(line)))
    return patterns


def _ignored(rel_path: str, patterns: list[tuple[bool, re.Pattern[str]]]) -> bool:
    parts = posixpath.normpath(rel_path.replace("\\", "/")).split("/")
    prefixes = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    ignored = False
    for negated, regex in patterns:
        if any(regex.match(p) for p in prefixes):
            ignored = not negated
    return ignored


def dockerignored(rel_path: str, dockerignore: Path) -> bool:
    """Docker semantics: ignored when the path or a parent directory matches; last match wins; `!` re-includes."""
    return _ignored(rel_path, _load_patterns(dockerignore))


def _resolve_dest(workdir: str, dest: str) -> str:
    return posixpath.normpath(dest if dest.startswith("/") else posixpath.join(workdir, dest))


def replicate_runtime_layout(df: Dockerfile, repo_root: Path, app_dir: Path, *, dockerignore: Path | None = None) -> Path:
    """Copy each runtime-stage COPY source to the path the image would hold it at, under `app_dir` (stands in for /app)."""
    repo_root, app_dir = Path(repo_root), Path(app_dir)
    patterns = _load_patterns(dockerignore) if dockerignore else []
    app_dir.mkdir(parents=True, exist_ok=True)

    def target(container_path: str) -> Path:
        rel = posixpath.relpath(container_path, "/app")
        return app_dir if rel == "." else app_dir.joinpath(*rel.split("/"))

    def put(src: Path, rel: str, dst: Path) -> None:
        if _ignored(rel, patterns):
            return
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for child in sorted(src.iterdir()):
                put(child, f"{rel}/{child.name}", dst / child.name)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for item in df.runtime_copies:
        dest = _resolve_dest(item.workdir, item.dest)
        dest_is_dir = item.dest.endswith("/") or len(item.sources) > 1
        for source in item.sources:
            rel = posixpath.normpath(source)
            src = repo_root / rel
            if src.is_dir():
                put(src, rel, target(dest))
            elif dest_is_dir:
                put(src, rel, target(dest) / src.name)
            else:
                put(src, rel, target(dest))
    return app_dir


def pythonpath_for(df: Dockerfile, app_dir: Path) -> str:
    """The image's PYTHONPATH with `/app` rebased onto the replica directory."""
    entries = []
    for entry in df.env.get("PYTHONPATH", "").split(":"):
        if not entry:
            continue
        rel = posixpath.relpath(entry, "/app")
        entries.append(str(Path(app_dir)) if rel == "." else str(Path(app_dir).joinpath(*rel.split("/"))))
    return os.pathsep.join(entries)
