"""MVP-008: build and probe every registered image from a clean export of a git tree. Stdlib only.

    python scripts/ci/clean_checkout.py --tree <tree id> [--out DIR] [--compose] [--tag TAG]
                                        [--tests] [--remove-images]

D-17: nothing is committed, so the checkout is ``git archive --format=tar <tree>`` of a checkpoint tree (read-only
for the repository, byte-identical to the tree: autocrlf is switched off), extracted into
``D:\\tmp\\mvp-exec\\clean-checkout\\<tree7>`` (or ``--out``, never inside this repository). The images and probe
modules come from the export's own scripts/ci/images.py (this tool's registry only when the tree has none). Every
image is built from that directory with ``build_images.build_command`` and ``--iidfile``, then probed by importing
its modules inside the image, addressed by image id. Exit 0 only when every expected step ran and succeeded; a
failure never stops later steps, so one run lists every broken image.

The export has no ``.git``. The tests that call ``git ls-files`` or ``git check-ignore`` (tests/build) would exit 2
there, so ``--tests`` first gives the export a private git context (``git init`` plus ``git add -A`` inside the
export directory, never a commit) and then runs tests/build in it with an isolated environment.
"""
from __future__ import annotations

import argparse
import io
import os
import runpy
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts\ci\clean_checkout.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.build_images import build_command, image_ref  # noqa: E402
from scripts.ci.images import IMAGES, Image, probe_env  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(r"D:\tmp\mvp-exec\clean-checkout")
MARKER = ".clean-checkout-export"
TAIL_LINES = 25


@dataclass(frozen=True)
class Step:
    image: Image
    build: list[str]
    probes: list[list[str]] = field(default_factory=list)


def default_tag(tree_id: str) -> str:
    """A tag no other run shares (not even a second run of the same tree)."""
    return f"clean-{tree_id[:7]}-{secrets.token_hex(4)}"


def git_env(environ: Mapping[str, str]) -> dict[str, str]:
    """`environ` without GIT_* so a git call never follows the operator's GIT_DIR / GIT_INDEX_FILE."""
    return {k: v for k, v in environ.items() if not k.startswith("GIT_")}


def resolve_tree(tree: str, repo: Path = REPO) -> str:
    """The full tree id of any tree-ish, resolved once and used everywhere (tags, paths, marker)."""
    proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{tree}^{{tree}}"],
                          capture_output=True, text=True, env=git_env(os.environ))
    if proc.returncode or not proc.stdout.strip():
        raise SystemExit(f"{tree!r} does not name a tree in {repo}")
    return proc.stdout.strip()


def cleanup(built: Sequence[tuple[str, str]], run: Callable[..., subprocess.CompletedProcess] | None = None) -> None:
    """Untag each (ref, image id) this run built, by reference and without -f, and only while the ref still points at
    that id. Docker gives identical builds one id, so removing by id would delete other runs' tags."""
    run = run or subprocess.run
    for ref, image_id in dict.fromkeys(built):
        found = run(["docker", "image", "inspect", "-f", "{{.Id}}", ref], capture_output=True, text=True)
        if found.returncode == 0 and found.stdout.strip() == image_id:
            run(["docker", "rmi", ref], capture_output=True, text=True)


def probe_commands(image: Image, tag: str, env: dict[str, str] | None = None) -> list[list[str]]:
    """One `docker run` per module: the CMD target (if any) first, then the image's probe modules."""
    env_flags: list[str] = []
    for key, value in (env if env is not None else probe_env()).items():
        env_flags += ["-e", f"{key}={value}"]
    modules = ([image.module] if image.module else []) + list(image.probes)
    return [
        ["docker", "run", "--rm", "--entrypoint", "python", "-e", "ENVIRONMENT=development", *env_flags,
         image_ref(image.name, tag), "-c", f"import importlib; importlib.import_module({module!r})"]
        for module in modules
    ]


def plan(tag: str, images: Sequence[Image] = IMAGES, env_factory: Callable[[], dict[str, str]] = probe_env) -> list[Step]:
    if not images:
        raise ValueError("no images registered: nothing would be proven")
    env = env_factory()  # one throwaway set per run
    steps = [Step(i, build_command(i, tag, "."), probe_commands(i, tag, env)) for i in images]
    for step in steps:
        if not step.probes:
            raise ValueError(f"image {step.image.name!r} has neither a module nor probes: it would be built bare")
    return steps


def build_with_iid(build: list[str], iidfile: str) -> list[str]:
    """The build command that also records the built image id (context stays last)."""
    return [*build[:-1], "--iidfile", iidfile, build[-1]]


def load_registry(export: Path) -> tuple[list[Image], Callable[[], dict[str, str]], str]:
    """The export's own images.py (stdlib only), else this tool's registry; the third value says which."""
    path = Path(export) / "scripts" / "ci" / "images.py"
    if path.is_file():
        ns = runpy.run_path(str(path), run_name="export_images")
        return list(ns["IMAGES"]), ns["probe_env"], f"export ({path})"
    return list(IMAGES), probe_env, "tool (the tree has no scripts/ci/images.py)"


def has_git_context(path: Path) -> bool:
    """True when `path` is the top of a git work tree; a `git archive` export is not."""
    return (Path(path) / ".git").exists()


def resolve_out(out: str | None, tree: str) -> Path:
    path = (Path(out) if out else DEFAULT_ROOT / tree[:7]).resolve()
    repo = REPO.resolve()
    if path == repo or repo in path.parents:
        raise SystemExit(f"refusing --out {path}: it is inside the repository ({repo})")
    return path


def tests_env(out: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """The environment for tests/build inside the export: no SOLACE_* switches, repo paths on PYTHONPATH."""
    env = {k: v for k, v in environ.items() if not k.startswith(("SOLACE_", "GIT_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(out), str(out / "src")])
    env["ENVIRONMENT"] = "test"
    return env


def tests_command() -> list[str]:
    return [sys.executable, "-m", "pytest", "tests/build", "-o", "addopts=", "-p", "no:logging", "-q",
            "-m", "not realinfra"]


def _printable(cmd: Sequence[str]) -> str:
    """The command for the log, with the values of probe `-e KEY=value` flags hidden."""
    shown: list[str] = []
    hide = False
    for part in cmd:
        if hide:
            key = part.partition("=")[0]
            shown.append(part if key == "ENVIRONMENT" else f"{key}=***")
            hide = False
            continue
        hide = part == "-e"
        shown.append(part)
    return " ".join(shown)


def _clear_readonly_and_retry(func, path, exc) -> None:
    """rmtree error handler: Git for Windows leaves read-only loose objects; clear the bit and retry once."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_export(out: Path) -> None:
    """Delete an earlier export, the marker LAST, so a delete that fails part-way still leaves a reusable export."""
    for child in sorted(out.iterdir(), key=lambda p: p.name == MARKER):
        if child.name == MARKER:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, onexc=_clear_readonly_and_retry)
        else:
            os.chmod(child, stat.S_IWRITE)
            child.unlink()
    (out / MARKER).unlink(missing_ok=True)
    out.rmdir()


def export_tree(tree: str, out: Path, repo: Path = REPO) -> None:
    """Extract `git archive --format=tar <tree>` (no autocrlf: byte-identical) into an empty or earlier export."""
    if out.exists():
        if any(out.iterdir()) and not (out / MARKER).exists():
            raise SystemExit(f"refusing to reuse {out}: it is not empty and is not an earlier export")
        remove_export(out)
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-C", str(repo),
                           "archive", "--format=tar", tree], capture_output=True, env=git_env(os.environ))
    if proc.returncode:
        raise SystemExit(f"git archive {tree} failed: {proc.stderr.decode(errors='replace').strip()}")
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        tar.extractall(out, filter="data")
    (out / MARKER).write_text(tree + "\n", encoding="utf-8")


def _run(label: str, cmd: list[str], cwd: Path, results: list[tuple[str, bool]],
         env: Mapping[str, str] | None = None) -> bool:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, errors="replace",
                          env=dict(env) if env is not None else None)
    ok = proc.returncode == 0
    print(f"[{'ok' if ok else 'FAIL'}] {label}", flush=True)
    if not ok:
        print(f"      $ {_printable(cmd)}")
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-TAIL_LINES:]
        for line in tail:
            print(f"      | {line}")
    results.append((label, ok))
    return ok


def _git_context(out: Path) -> None:
    """A private repository inside the export so `git ls-files` works; nothing is committed."""
    env = git_env(os.environ)

    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", "core.safecrlf=false", "-C", str(out), *args], check=True,
                              capture_output=True, text=True, env=env).stdout.strip()

    git("init", "-q")
    if Path(git("rev-parse", "--absolute-git-dir")).resolve() != (out / ".git").resolve():
        raise SystemExit(f"the git context for {out} did not land in {out / '.git'}")
    git("add", "-A")


def verdict(results: list[tuple[str, bool]], steps: Sequence[Step], compose: bool, tests: bool) -> int:
    """0 only when no step failed and every expected step actually ran (never vacuous)."""
    expected = len(steps) + sum(len(s.probes) for s in steps) + int(compose) + int(tests)
    if not steps or len(results) != expected or not all(ok for _, ok in results):
        return 1
    return 0


def run(tree: str, out: Path, tag: str, compose: bool, tests: bool, remove_images: bool,
        require_registry_match: bool = False) -> int:
    export_tree(tree, out)
    if has_git_context(out):
        raise SystemExit(f"the export {out} carries VCS state; refusing to build from it")
    print(f"exported {tree} -> {out}", flush=True)
    images, env_factory, source = load_registry(out)
    print(f"tag: {tag}", flush=True)
    print(f"registry: {source}: {', '.join(i.name for i in images)}", flush=True)
    if [i.name for i in images] != [i.name for i in IMAGES]:
        print("warning: the export's image list differs from the working tree's scripts/ci/images.py", flush=True)
        if require_registry_match:
            print("[FAIL] --require-registry-match: the registries differ")
            return 1
    results: list[tuple[str, bool]] = []
    try:
        steps = plan(tag, images, env_factory)
    except ValueError as exc:
        print(f"[FAIL] plan: {exc}")
        return 1
    built: list[tuple[str, str]] = []
    try:
        for step in steps:
            ref = image_ref(step.image.name, tag)
            with tempfile.TemporaryDirectory() as tmp:
                iid = Path(tmp) / "iid.txt"
                ok = _run(f"build {step.image.name}", build_with_iid(step.build, str(iid)), out, results)
                image_id = iid.read_text(encoding="utf-8").strip() if ok and iid.is_file() else ""
            if image_id:
                built.append((ref, image_id))
            for probe in step.probes:
                label = f"probe {step.image.name}: {probe[-1].split('import_module(')[-1].rstrip(')')}"
                if not image_id:
                    results.append((label + " (not run: no built image)", False))
                    print(f"[FAIL] {label} (not run: no built image)")
                    continue
                _run(label, [image_id if part == ref else part for part in probe], out, results)
        if compose:
            _run("compose build", ["docker", "compose", "-f", "docker-compose.yml", "build"], out, results)
        if tests:
            _git_context(out)
            _run("tests/build in the export (private git context)", tests_command(), out, results,
                 tests_env(out, os.environ))
    finally:
        if remove_images:
            cleanup(built)
    failed = [label for label, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} steps ok" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    code = verdict(results, steps, compose, tests)
    if code and not failed:
        print("FAILED: not every expected step ran")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True, help="tree id (or any tree-ish) to export with git archive")
    parser.add_argument("--out", help=f"export directory outside the repository (default {DEFAULT_ROOT}\\<tree7>)")
    parser.add_argument("--tag", help="image tag (default clean-<tree7>-<random>)")
    parser.add_argument("--compose", action="store_true", help="also run `docker compose -f docker-compose.yml build`")
    parser.add_argument("--tests", action="store_true", help="also run tests/build in the export (private git context)")
    parser.add_argument("--remove-images", action="store_true", help="remove the images built here when done")
    parser.add_argument("--require-registry-match", action="store_true",
                        help="fail when the export's image list differs from this working tree's registry (gates)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tree = resolve_tree(args.tree)
    return run(tree, resolve_out(args.out, tree), args.tag or default_tag(tree),
               args.compose, args.tests, args.remove_images, args.require_registry_match)


if __name__ == "__main__":
    raise SystemExit(main())
