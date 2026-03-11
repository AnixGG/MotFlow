from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from tracking.ultralytics_runtime import get_ultralytics_version


def git_output(*args: str, cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_env_info(path: Path, repo_root: Path, ultralytics_root: Path) -> None:
    import torch

    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    lines = [
        f"repo_root: {repo_root}",
        f"repo_git_commit: {git_output('git', 'rev-parse', 'HEAD', cwd=repo_root)}",
        f"ultralytics_repo_commit: {git_output('git', 'rev-parse', 'HEAD', cwd=ultralytics_root)}",
        f"python: {platform.python_version()}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"torch_cuda: {torch.version.cuda or 'none'}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"gpu: {gpu_name}",
        f"ultralytics: {get_ultralytics_version()}",
        f"hostname: {platform.node()}",
        f"pid: {os.getpid()}",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
