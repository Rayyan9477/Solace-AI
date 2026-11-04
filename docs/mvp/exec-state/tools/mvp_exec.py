"""Cross-platform execution tools for the MVP remediation program.

They replace the Windows PowerShell scripts (checkpoint.ps1, review-package.ps1, extract-brief.ps1)
and the D:\\tmp workspace. Paths are relative to the repository, so the tools work locally and in a
Claude Code cloud session (Ubuntu).

    python docs/mvp/exec-state/tools/mvp_exec.py checkpoint <label>
    python docs/mvp/exec-state/tools/mvp_exec.py review-package <base> <head> <name> [path ...]
    python docs/mvp/exec-state/tools/mvp_exec.py extract-brief <task>
    python docs/mvp/exec-state/tools/mvp_exec.py ledger "<line>"

`checkpoint` writes a tree id from a temporary index (all tracked and untracked, non-ignored files),
without touching the real index or making a commit. In a cloud session a git commit on the working
branch is the normal checkpoint; the tree id is still logged so reviews can diff tree to tree.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
STATE = REPO / "docs" / "mvp" / "exec-state"
PLAN_DIR = REPO / "docs" / "superpowers" / "plans" / "2026-10-18-mvp-remediation"


def _git(*args: str, env: dict | None = None) -> str:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, env=env,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args[:2])} failed: {out.stderr.strip()[:400]}")
    return out.stdout


def _now() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def checkpoint(label: str) -> str:
    snaps = STATE / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    idx = Path(tempfile.gettempdir()) / f"mvp-index-{uuid.uuid4().hex}"
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_INDEX_FILE"] = str(idx)
    try:
        _git("read-tree", "HEAD", env=env)
        _git("-c", "core.safecrlf=false", "add", "-A", env=env)
        tree = _git("write-tree", env=env).strip()
    finally:
        idx.unlink(missing_ok=True)
    with open(snaps / "snapshots.log", "a", encoding="utf-8") as f:
        f.write(f"{_now()}\t{label}\t{tree}\n")
    return tree


def review_package(base: str, head: str, name: str, paths: list[str]) -> Path:
    out_dir = STATE / "reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}-{_dt.datetime.now():%Y%m%d-%H%M%S}.diff.md"
    scope = ["--", *paths] if paths else []
    all_changed = _git("diff", "--name-status", base, head)
    stat = _git("diff", "--stat", base, head, *scope)
    diff = _git("diff", "-U10", base, head, *scope)
    declared = ", ".join(paths) if paths else "(whole tree)"
    out.write_text(
        f"# Review package: {name}\n\nBase tree: {base}\nHead tree: {head}\nDeclared paths: {declared}\n\n"
        f"## All changed paths (whole tree, includes other lanes)\n```\n{all_changed}```\n\n"
        f"## Stat (declared paths)\n```\n{stat}```\n\n## Diff (declared paths)\n```diff\n{diff}```\n",
        encoding="utf-8")
    return out


def extract_brief(task: str) -> Path:
    ws = task.split(".")[0]
    files = sorted(PLAN_DIR.glob(f"{ws}-*.md"))
    if not files:
        raise SystemExit(f"No plan file for {ws} in {PLAN_DIR}")
    lines = files[0].read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(f"### Task {task}:")), -1)
    if start < 0:
        raise SystemExit(f"Task {task} not found in {files[0].name}")
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("### Task ") or lines[j].startswith("## ")), len(lines))
    out_dir = STATE / "briefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{task}-brief.md"
    out.write_text("\n".join([f"<!-- brief extracted from {files[0].name} -->", *lines[start:end]]) + "\n",
                   encoding="utf-8")
    return out


def ledger(line: str) -> None:
    with open(STATE / "ledger" / "progress.md", "a", encoding="utf-8") as f:
        f.write(f"- {_now()} {line}\n")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "checkpoint" and len(rest) == 1:
        print(checkpoint(rest[0]))
    elif cmd == "review-package" and len(rest) >= 3:
        print(review_package(rest[0], rest[1], rest[2], rest[3:]))
    elif cmd == "extract-brief" and len(rest) == 1:
        print(extract_brief(rest[0]))
    elif cmd == "ledger" and len(rest) == 1:
        ledger(rest[0])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
