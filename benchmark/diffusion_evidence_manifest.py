#!/usr/bin/env python3
"""Build a reproducibility manifest for diffusion evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENV_PREFIXES = ("VLLM_", "SGLANG_", "CUDA_", "NCCL_", "TORCH_")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(command: list[str], cwd: str | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "command": command,
            "cwd": cwd,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except OSError as exc:
        return {
            "command": command,
            "cwd": cwd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def read_archive_provenance(repo: Path) -> dict[str, Any] | None:
    path = repo / ".codex-source-provenance.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}
    if not isinstance(payload, dict):
        return {"error": "archive provenance was not a JSON object", "path": str(path)}
    payload["path"] = str(path)
    return payload


def git_info(path: str) -> dict[str, Any]:
    repo = Path(path)
    info: dict[str, Any] = {"path": str(repo), "exists": repo.exists()}
    if not repo.exists():
        info["error"] = "path does not exist"
        return info

    archive_provenance = read_archive_provenance(repo)
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=str(repo))
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo))
    status = run_command(["git", "status", "--short"], cwd=str(repo))
    info.update(
        {
            "commit": commit["stdout"].strip()
            if commit["returncode"] == 0
            else (
                archive_provenance.get("commit") if archive_provenance else None
            ),
            "branch": branch["stdout"].strip()
            if branch["returncode"] == 0
            else (
                archive_provenance.get("branch") if archive_provenance else None
            ),
            "status": status["stdout"]
            if status["returncode"] == 0
            else (
                archive_provenance.get("status") if archive_provenance else None
            ),
            "archive_provenance": archive_provenance,
            "commands": {
                "commit": commit,
                "branch": branch,
                "status": status,
            },
        }
    )
    return info


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    info: dict[str, Any] = {"path": path_text, "exists": path.exists()}
    if path.exists() and path.is_file():
        info["size_bytes"] = path.stat().st_size
        info["sha256"] = sha256_file(path)
    return info


def selected_environment() -> dict[str, str]:
    rows = {}
    for key, value in os.environ.items():
        if key.startswith(ENV_PREFIXES):
            rows[key] = value
    return dict(sorted(rows.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a JSON manifest for diffusion PR evidence."
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--vllm-repo", default="/mnt/lvm/minsub/vocab/vllm")
    parser.add_argument("--sglang-repo", default="/mnt/lvm/minsub/vocab/sglang")
    parser.add_argument("--vllm-wheel", default="")
    parser.add_argument("--model-path", action="append", default=[])
    parser.add_argument("--env-flag", action="append", default=[])
    return parser.parse_args()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    wheel = file_info(args.vllm_wheel) if args.vllm_wheel else {"path": "", "exists": False}
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "repositories": {
            "vllm": git_info(args.vllm_repo),
            "sglang": git_info(args.sglang_repo),
        },
        "vllm_wheel": wheel,
        "model_paths": [file_info(path) for path in args.model_path],
        "env_flags": args.env_flag,
        "environment": selected_environment(),
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
    }


def main() -> int:
    args = parse_args()
    payload = build_manifest(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
