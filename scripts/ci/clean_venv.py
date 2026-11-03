"""Build the lock-only venv (D-20): a fresh venv WITHOUT system site packages, synced to the dev lock.

A suite that is green in the W0 venv (which also sees anaconda's packages) but red here is a lock
defect: fix it with an .in change and a recompile, never by installing into this venv by hand.
The base interpreter is only the template the venv is created from; nothing is installed into it.

    clean_venv.py --build [--venv DIR] [--lock PATH] [--base-python PATH] [--uv PATH]

Output is package names and versions only. Stdlib only.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV_CLEAN = Path(r"D:\tmp\mvp-exec\venv-clean")
DEFAULT_LOCK = ROOT / "requirements-dev.lock"
DEFAULT_BASE_PYTHON = r"C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe"
DEFAULT_UV = r"C:\Users\rayyan.a\AppData\Local\anaconda3\Scripts\uv.exe"
_PKG = re.compile(r"^\s*[+\-~]\s+([A-Za-z0-9][A-Za-z0-9._-]*==\S+)")


def venv_python(venv: Path) -> Path:
    """The venv's interpreter: Scripts\\python.exe on Windows, bin/python elsewhere."""
    return venv / "Scripts" / "python.exe" if sys.platform == "win32" else venv / "bin" / "python"


def commands(*, base_python: str, uv: str, venv: Path, lock: Path) -> list[list[str]]:
    """Create an isolated venv (never --system-site-packages), then make it hold exactly the lock."""
    return [
        [base_python, "-m", "venv", "--clear", str(venv)],
        [uv, "pip", "sync", "--python", str(venv_python(venv)), "--require-hashes", str(lock)],
    ]


def is_isolated(pyvenv_cfg: Path) -> bool:
    """True only when pyvenv.cfg says ``include-system-site-packages = false``."""
    try:
        text = Path(pyvenv_cfg).read_text(encoding="utf-8")
    except OSError:
        return False
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip().lower() == "include-system-site-packages":
            return value.strip().lower() == "false"
    return False


def build(*, venv: Path, lock: Path, base_python: str, uv: str) -> int:
    for cmd in commands(base_python=base_python, uv=uv, venv=venv, lock=lock):
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        for line in (proc.stdout + proc.stderr).splitlines():
            match = _PKG.match(line)
            if match:
                print(match.group(1))
        if proc.returncode != 0:
            print(f"FAILED (exit {proc.returncode}): {Path(cmd[0]).name} {' '.join(cmd[1:3])}", file=sys.stderr)
            # uv's own diagnostics name the packages that could not be installed
            print((proc.stderr or proc.stdout).strip(), file=sys.stderr)
            return proc.returncode
    if not is_isolated(venv / "pyvenv.cfg"):
        print(f"FAILED: {venv / 'pyvenv.cfg'} does not say include-system-site-packages = false", file=sys.stderr)
        return 1
    print("isolated: include-system-site-packages = false")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--build", action="store_true", required=True, help="create and sync the clean venv")
    parser.add_argument("--venv", type=Path, default=VENV_CLEAN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--base-python", default=DEFAULT_BASE_PYTHON)
    parser.add_argument("--uv", default=DEFAULT_UV)
    args = parser.parse_args(argv)
    return build(venv=args.venv, lock=args.lock, base_python=args.base_python, uv=args.uv)


if __name__ == "__main__":
    sys.exit(main())
