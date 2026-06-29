#!/usr/bin/env python3
"""Create a tar.gz reviewer bundle from diffusion evidence artifacts."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle diffusion evidence artifacts into a tar.gz archive."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-tar", required=True)
    return parser.parse_args()


def should_include(path: Path, output_tar: Path) -> bool:
    try:
        return path.resolve() != output_tar.resolve()
    except OSError:
        return True


def build_bundle(artifact_dir: Path, output_tar: Path) -> None:
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tar, "w:gz") as archive:
        for path in sorted(artifact_dir.rglob("*")):
            if not path.is_file() or not should_include(path, output_tar):
                continue
            archive.add(path, arcname=str(path.relative_to(artifact_dir)))


def main() -> int:
    args = parse_args()
    build_bundle(Path(args.artifact_dir), Path(args.output_tar))
    print(args.output_tar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
