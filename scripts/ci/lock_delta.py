"""Compare a uv-compiled, hash-pinned lock with the distributions the running interpreter sees (D-16, D-20).

The lock is read in uv's default ``split`` annotation style::

    name==1.2.3 ; marker \\
        --hash=sha256:... \\
        --hash=sha256:...
        # via
        #   -r requirements.in
        #   parent

Markers are evaluated for the running interpreter, and per distribution name the first copy on
``sys.path`` wins, exactly as ``importlib.metadata.version()`` resolves it, so a venv's own copy
shadows one it sees through its system site packages.

CLI:
    lock_delta.py --write-constraints OUT [--exclude a,b]   installed name==version, for uv -c
    lock_delta.py --pins-only LOCK OUT                      the lock's pins without hashes, for uv -c
    lock_delta.py --lock LOCK --write-install OUT [--allow-missing a,b]
                                                            hashed install lines for what differs
Stdlib plus ``packaging`` only.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from packaging.markers import Marker
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

_REQ = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==(?P<version>[^\s;\\]+)"
    r"\s*(?:;\s*(?P<marker>[^\\]*?))?\s*\\?\s*$"
)
_HASH = re.compile(r"--hash=(?P<hash>[A-Za-z0-9]+:[0-9A-Fa-f]+)")


@dataclass(frozen=True)
class LockEntry:
    """One pinned distribution in a lock file."""

    name: str
    version: str
    marker: str | None
    hashes: tuple[str, ...]
    via: tuple[str, ...]

    def applies(self) -> bool:
        """True when the entry's marker holds for the running interpreter (or it has none)."""
        return self.marker is None or Marker(self.marker).evaluate()


@dataclass(frozen=True)
class Delta:
    """What the running interpreter lacks (name -> locked version) or holds at another version."""

    missing: dict[str, str]
    mismatched: dict[str, tuple[str, str]]  # name -> (locked, installed)


def _source(text: str) -> str:
    """A ``via`` source: ``-r``/``-c`` lines verbatim, package names canonicalized (extras dropped)."""
    text = text.strip()
    if text.startswith("-"):
        return text
    return canonicalize_name(re.split(r"[\[\s]", text, maxsplit=1)[0])


def parse_lock(path: Path) -> dict[str, LockEntry]:
    """Parse a uv lock into canonical name -> entry.

    When a universal lock forks a name across markers, the copy whose marker holds for the running
    interpreter is kept (the first one when none or several hold).
    """
    raw: list[dict] = []
    current: dict | None = None
    in_via = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            in_via = False
            match = _REQ.match(line)
            current = None
            if match:
                current = {
                    "name": canonicalize_name(match.group("name")),
                    "version": match.group("version"),
                    "marker": (match.group("marker") or "").strip() or None,
                    "hashes": [],
                    "via": [],
                }
                raw.append(current)
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("--hash="):
            current["hashes"].extend(m.group("hash") for m in _HASH.finditer(stripped))
            in_via = False
        elif stripped.startswith("# via"):
            rest = stripped[len("# via"):].strip()
            if rest:
                current["via"].append(_source(rest))
                in_via = False
            else:
                in_via = True
        elif in_via and stripped.startswith("#"):
            item = stripped.lstrip("#").strip()
            if item:
                current["via"].append(_source(item))
    entries: dict[str, LockEntry] = {}
    for item in raw:
        entry = LockEntry(item["name"], item["version"], item["marker"], tuple(item["hashes"]), tuple(item["via"]))
        kept = entries.get(entry.name)
        if kept is None or (not kept.applies() and entry.applies()):
            entries[entry.name] = entry
    return entries


def installed_versions() -> dict[str, str]:
    """Canonical name -> version for every distribution visible on ``sys.path``; the first copy wins."""
    seen: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") if dist.metadata is not None else None
        if not name:
            continue
        seen.setdefault(canonicalize_name(name), dist.version)
    return seen


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _same_version(a: str, b: str) -> bool:
    try:
        return Version(a) == Version(b)
    except InvalidVersion:
        return a == b


def compute_delta(lock: Path, *, allow_missing: frozenset[str] = frozenset()) -> Delta:
    """Locked entries (for this interpreter's markers) that are not installed, or installed at another version."""
    allowed = {canonicalize_name(n) for n in allow_missing}
    missing: dict[str, str] = {}
    mismatched: dict[str, tuple[str, str]] = {}
    for name, entry in sorted(parse_lock(lock).items()):
        if not entry.applies():
            continue
        have = _installed_version(name)
        if have is None:
            if name not in allowed:
                missing[name] = entry.version
        elif not _same_version(have, entry.version):
            mismatched[name] = (entry.version, have)
    return Delta(missing=missing, mismatched=mismatched)


def exclusive_closure(entries: Mapping[str, LockEntry], roots: set[str]) -> set[str]:
    """The roots plus every entry pulled in only through them.

    An entry joins when it is not a direct requirement (no ``-r`` source) and every package that
    requires it is already inside. ``-c`` sources only record that a constraint applied, so they
    neither require an entry nor keep it out.
    """
    closure = {canonicalize_name(r) for r in roots}
    changed = bool(closure)
    while changed:
        changed = False
        for name, entry in entries.items():
            if name in closure:
                continue
            sources = [src for src in entry.via if not src.startswith("-c")]
            if sources and all(not src.startswith("-") and src in closure for src in sources):
                closure.add(name)
                changed = True
    return closure


def _names(csv: str | None) -> frozenset[str]:
    return frozenset(canonicalize_name(n) for n in (csv or "").split(",") if n.strip())


def _write(out: Path, lines: Iterable[str]) -> int:
    body = list(lines)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(f"{line}\n" for line in body), encoding="utf-8", newline="\n")  # no BOM
    return len(body)


def write_constraints(out: Path, exclude: frozenset[str] = frozenset()) -> int:
    lines = []
    for name, version in sorted(installed_versions().items()):
        if name in exclude:
            continue
        try:
            Version(version)
        except InvalidVersion:
            print(f"skipped (not PEP 440): {name}=={version}", file=sys.stderr)
            continue
        lines.append(f"{name}=={version}")
    return _write(out, lines)


def write_pins_only(lock: Path, out: Path) -> int:
    lines = []
    for raw in Path(lock).read_text(encoding="utf-8").splitlines():
        match = _REQ.match(raw)
        if match:
            marker = (match.group("marker") or "").strip()
            pin = f"{canonicalize_name(match.group('name'))}=={match.group('version')}"
            lines.append(f"{pin} ; {marker}" if marker else pin)
    return _write(out, lines)


def write_install(lock: Path, out: Path, allow_missing: frozenset[str] = frozenset()) -> Delta:
    entries = parse_lock(lock)
    delta = compute_delta(lock, allow_missing=allow_missing)
    names = sorted(set(delta.missing) | set(delta.mismatched))
    lines = []
    for name in names:
        entry = entries[name]
        hashes = " ".join(f"--hash={h}" for h in entry.hashes)
        lines.append(f"{name}=={entry.version} {hashes}".rstrip())
    _write(out, lines)
    return delta


def _report(delta: Delta) -> None:
    for name, version in sorted(delta.missing.items()):
        print(f"missing     {name}=={version}")
    for name, (locked, have) in sorted(delta.mismatched.items()):
        print(f"mismatched  {name}=={locked} (installed {have})")
    print(f"missing={len(delta.missing)} mismatched={len(delta.mismatched)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-constraints", metavar="OUT", type=Path)
    mode.add_argument("--pins-only", nargs=2, metavar=("LOCK", "OUT"), type=Path)
    mode.add_argument("--lock", metavar="LOCK", type=Path)
    parser.add_argument("--exclude", default="", help="comma-separated names left out of --write-constraints")
    parser.add_argument("--write-install", metavar="OUT", type=Path)
    parser.add_argument("--allow-missing", default="", help="comma-separated names that may be absent")
    args = parser.parse_args(argv)

    if args.write_constraints is not None:
        count = write_constraints(args.write_constraints, _names(args.exclude))
        print(f"wrote {count} constraints to {args.write_constraints}")
        return 0
    if args.pins_only is not None:
        lock, out = args.pins_only
        count = write_pins_only(lock, out)
        print(f"wrote {count} pins to {out}")
        return 0
    if args.write_install is not None:
        delta = write_install(args.lock, args.write_install, _names(args.allow_missing))
        _report(delta)
        print(f"wrote {len(delta.missing) + len(delta.mismatched)} install lines to {args.write_install}")
        return 0
    delta = compute_delta(args.lock, allow_missing=_names(args.allow_missing))
    _report(delta)
    return 1 if (delta.missing or delta.mismatched) else 0


if __name__ == "__main__":
    sys.exit(main())
