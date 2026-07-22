"""Run provenance: what produced a number, so it can be reproduced or thrown away.

Colab assigns a different GPU per session, so the device name is part of every row
(protocol §5): accuracy stays auditable even though the hardware varies.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable 12-char hash of a config. Key order does not matter; values are stringified."""
    payload = json.dumps(dict(cfg), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def git_commit(repo_root: Path | None = None) -> str:
    """Current HEAD sha, or "unknown" when git is absent or this is not a repository."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def gpu_name() -> str:
    """Name of the CUDA device, or "cpu" when torch or a GPU is unavailable."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return str(torch.cuda.get_device_name(0))


def run_meta(cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """The provenance columns stamped onto every result row."""
    return {
        "config_hash": config_hash(cfg),
        "commit": git_commit(),
        "seed": int(seed),
        "gpu": gpu_name(),
    }
