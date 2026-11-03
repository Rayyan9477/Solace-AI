"""MVP-008: list `tracked file -> untracked dependency` pairs. Stdlib only.

    python scripts/ci/check_tracked_closure.py [--repo DIR] [--report | --strict]

A file that git tracks but that depends on a file git does not track works in this working tree and breaks in any
clean checkout of the tracked content. Three kinds of dependency are followed, each resolved with only
`git ls-files` (tracked, the index) and `git ls-files --others --exclude-standard` (untracked, not ignored):

* Python imports of tracked .py files, resolved against the repo root, `src/` and each `services/<svc>`, plus
  relative imports;
* the COPY/ADD sources of tracked Dockerfiles (paths relative to the repo root, the build context);
* `-r` / `-c` includes in tracked requirements files (relative to the including file).

Each pair prints as `<tracked file> -> <untracked file>`. `--report` (default) always exits 0; `--strict` exits 1
when any pair exists. D-17: nothing is committed, so the real repository shows many pairs for the whole program;
the list is input for the owner's post-program commit decision and gates nothing.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):  # allow `python scripts\ci\check_tracked_closure.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci import dockerfiles  # noqa: E402

_REQ_INCLUDE = re.compile(r"^\s*(?:-r|-c|--requirement|--constraint)[\s=]+(\S+)")
_DOCKERFILE = re.compile(r"(^|/)(Dockerfile(\..+)?|[^/]+\.Dockerfile)$")
_REQUIREMENTS = re.compile(r"(^|/)(requirements[^/]*\.(txt|in)|constraints[^/]*\.txt)$")


def _ls(repo: Path, *args: str) -> set[str]:
    proc = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", *args], capture_output=True)
    if proc.returncode:
        raise SystemExit(f"git ls-files failed in {repo}: {proc.stderr.decode(errors='replace').strip()}")
    return {p for p in proc.stdout.decode("utf-8", errors="replace").split("\0") if p}


def _norm(path: PurePosixPath) -> str:
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


class Closure:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.tracked = _ls(repo)
        self.untracked = _ls(repo, "--others", "--exclude-standard")
        self.known = self.tracked | self.untracked
        self.dockerignore = repo / ".dockerignore"

    @staticmethod
    def roots_for(path: str) -> list[str]:
        """Import roots for one importing file: its own service first (merged-src and flat layouts), then the repo."""
        parts = path.split("/")
        if parts[0] == "services" and len(parts) > 2:
            svc = f"services/{parts[1]}"
            return [svc, f"{svc}/src", "", "src"]
        return ["", "src"]

    def _join(self, root: str, rel: str) -> str:
        return f"{root}/{rel}" if root else rel

    def _module_files(self, root: str, parts: list[str], leaf_only: bool = False) -> list[str]:
        """Files that make up `import a.b.c` below `root`: each package __init__ and the module file itself."""
        out: list[str] = []
        for i in range(len(parts) if leaf_only else 1, len(parts) + 1):
            if leaf_only and i != len(parts):
                continue
            base = self._join(root, "/".join(parts[:i]))
            out += [base + ".py", base + "/__init__.py"]
        return out

    def _resolve(self, parts: list[str], names: list[str], roots: list[str]) -> set[str]:
        """The first root where the imported module (or an imported submodule) exists wins; only when no root has
        it, the first root holding any package of the chain is used."""
        partial: set[str] = set()
        for root in roots:
            chain = self._module_files(root, parts) if parts else []
            found = [c for c in chain if c in self.known]
            if not found:
                continue
            leaf = [c for c in self._module_files(root, parts, leaf_only=True) if c in self.known]
            for name in names:
                sub = [c for c in self._module_files(root, [*parts, name], leaf_only=True) if c in self.known]
                found += sub
                leaf += sub
            if leaf:
                return set(found)
            partial = partial or set(found)
        return partial

    def python_deps(self, path: str) -> set[str]:
        try:
            tree = ast.parse((self.repo / path).read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            return set()
        deps: set[str] = set()
        here = PurePosixPath(path).parent
        roots = self.roots_for(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps |= self._resolve(alias.name.split("."), [], roots)
            elif isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names if a.name != "*"]
                if node.level:
                    base = here
                    for _ in range(node.level - 1):
                        base = base.parent
                    parts = node.module.split(".") if node.module else []
                    root = _norm(base)
                    chain = self._module_files(root, parts) if parts else []
                    found = [c for c in chain if c in self.known]
                    if not parts:
                        found += [self._join(root, "__init__.py")] if self._join(root, "__init__.py") in self.known else []
                    for name in names:
                        found += [c for c in self._module_files(root, [*parts, name], leaf_only=True) if c in self.known]
                    deps |= set(found)
                else:
                    deps |= self._resolve(node.module.split(".") if node.module else [], names, roots)
        return deps

    def dockerfile_deps(self, path: str) -> set[str]:
        try:
            df = dockerfiles.parse(self.repo / path)
        except (OSError, ValueError):
            return set()
        deps: set[str] = set()
        # files .dockerignore keeps out of the build context are not dependencies of the image
        pool = ({c for c in self.untracked if not dockerfiles.dockerignored(c, self.dockerignore)}
                if self.dockerignore.is_file() else self.untracked)
        for copy in df.copies:
            if "from" in copy.flags:
                continue
            for src in copy.sources:
                if "://" in src:
                    continue
                rel = _norm(PurePosixPath(src))
                if rel == "":
                    deps |= pool
                    continue
                for candidate in pool:
                    if candidate == rel or candidate.startswith(rel.rstrip("/") + "/") or fnmatch.fnmatchcase(candidate, rel):
                        deps.add(candidate)
        return deps

    def requirements_deps(self, path: str) -> set[str]:
        deps: set[str] = set()
        for line in (self.repo / path).read_text(encoding="utf-8", errors="replace").splitlines():
            match = _REQ_INCLUDE.match(line)
            if match:
                deps.add(_norm(PurePosixPath(path).parent / match.group(1)))
        return deps & self.untracked

    def pairs(self) -> list[tuple[str, str]]:
        found: set[tuple[str, str]] = set()
        for path in sorted(self.tracked):
            if not (self.repo / path).is_file():
                continue
            deps: set[str] = set()
            if path.endswith(".py"):
                deps |= self.python_deps(path)
            if _DOCKERFILE.search(path):
                deps |= self.dockerfile_deps(path)
            if _REQUIREMENTS.search(path):
                deps |= self.requirements_deps(path)
            found |= {(path, dep) for dep in deps if dep in self.untracked and dep != path}
        return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="print the pairs, always exit 0 (default)")
    mode.add_argument("--strict", action="store_true", help="exit 1 when any pair exists")
    args = parser.parse_args(argv)
    pairs = Closure(Path(args.repo)).pairs()
    for tracked, untracked in pairs:
        print(f"{tracked} -> {untracked}")
    print(f"{len(pairs)} tracked -> untracked pair(s)")
    return 1 if (args.strict and pairs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
