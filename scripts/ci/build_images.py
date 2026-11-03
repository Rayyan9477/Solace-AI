"""IF-21: build (and optionally probe) the MVP service images from the repository root. Stdlib only."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts\ci\build_images.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.images import IMAGES, Image  # noqa: E402

COMPOSE_TAG = "${SOLACE_IMAGE_TAG:-dev}"


def image_ref(name: str, tag: str) -> str:
    """The only place the image reference format lives."""
    return f"solace/{name}:{tag}"


def build_command(image: Image, tag: str, context: str) -> list[str]:
    return ["docker", "build", "-f", image.dockerfile, "-t", image_ref(image.name, tag), context]


def probe_command(image: Image, tag: str, module: str | None = None) -> list[str]:
    """Import the CMD target (or `module`) inside the built image."""
    code = f"import importlib; importlib.import_module({(module or image.module)!r})"
    return ["docker", "run", "--rm", "-e", "ENVIRONMENT=test", image_ref(image.name, tag), "python", "-c", code]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="image name, e.g. safety-service")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--tag")
    parser.add_argument("--context", default=".")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--list-json", action="store_true")
    args = parser.parse_args(argv)

    if args.image:
        selected = [i for i in IMAGES if i.name == args.image]
        if not selected:
            parser.error(f"unknown image {args.image!r}")
    else:
        selected = list(IMAGES)

    if args.list_json:
        print(json.dumps([{"name": i.name, "dockerfile": i.dockerfile, "module": i.module, "port": i.port}
                          for i in selected]))
        return 0
    if not args.tag:
        parser.error("--tag is required unless --list-json")

    for image in selected:
        rc = subprocess.run(build_command(image, args.tag, args.context)).returncode
        if rc:
            return rc
        if args.probe:
            # a module-less image (migrate) has no CMD target to import: probe its declared modules instead
            for module in ([image.module] if image.module else list(image.probes)):
                rc = subprocess.run(probe_command(image, args.tag, module)).returncode
                if rc:
                    return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
