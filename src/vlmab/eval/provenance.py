"""Run provenance: what produced a number, so it can be reproduced or thrown away.

Colab assigns a different GPU per session, so the device name is part of every row
(protocol §5): accuracy stays auditable even though the hardware varies.
"""
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


# src/vlmab/eval/provenance.py -> src/vlmab/eval -> src/vlmab -> src -> repo root.
# Provenance must describe the code that ran, which lives here, not whatever the
# process happens to be chdir'd into (on Colab that is /content, not the clone).
PACKAGE_REPO_ROOT = Path(__file__).resolve().parents[3]


def package_commit() -> str:
    """HEAD of the repository this package was loaded from, or "unknown".

    `PACKAGE_REPO_ROOT` only points at a checkout under an editable install. Installed
    normally, it lands inside the venv — and `git rev-parse` walks *up* from there, so
    it would happily return some unrelated ancestor repository's SHA. Stamping a wrong
    commit onto results is worse than recording none: it makes a run look reproducible
    when it is not. So we only trust the path when it actually contains a `.git`.
    """
    if not (PACKAGE_REPO_ROOT / ".git").exists():
        return "unknown"
    return git_commit(PACKAGE_REPO_ROOT)


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


def runtime_versions() -> dict[str, str | None]:
    """Installed anomalib and torch versions, for the GPU runners to stamp onto their shards.

    Protocol v0.2.21: the antialiased Resize is a torch/anomalib behaviour, so a number that
    does not say which runtime produced it cannot be told apart from the one a different
    release would give. Read from package metadata rather than by importing, so this costs
    nothing where neither is installed (CI) and None records that honestly instead of a
    fabricated version. Stamped as columns only, never part of the cfg `config_hash` covers.
    """
    out: dict[str, str | None] = {}
    for package in ("anomalib", "torch"):
        try:
            out[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[f"{package}_version"] = None
    return out


def run_meta(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance columns stamped onto every result row.

    The commit is resolved from `PACKAGE_REPO_ROOT`, not from the process cwd: a Colab
    notebook runs in /content while the clone sits elsewhere, so reading cwd would
    stamp "unknown" onto every row -- or, if cwd happened to be some other checkout, a
    completely unrelated repository's SHA. Either defeats the auditability the
    protocol commits to, on the exact platform this is built for.

    The seed is deliberately NOT here. It is stamped by `run_evaluation` from `method.seed` —
    the only value that was actually applied. When this function took a seed, `scripts/run_eval.py`
    passed `--seed` straight through and every shard recorded a number nothing had applied.
    """
    return {
        "config_hash": config_hash(cfg),
        "commit": package_commit(),
        "gpu": gpu_name(),
    }


def method_meta(method: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """`run_meta` plus the per-shard columns protocol §3 v0.2.21 requires of every runner.

    One helper so the four runners (run_mvtec_ad, run_mvtec_ad2, run_visa_secondary, run_eval)
    cannot drift apart again: before the 2026-10-19 whole-branch review the VisA runner and
    run_eval stamped neither `resize_antialias` nor the runtime versions.

    `preprocess` and `resize_antialias` are read off the method -- what is recorded has to be
    what was applied -- and None records that a method lacks the property rather than claiming
    one. Both decide a run's identity, so both go into the hashed cfg AND onto the shard as
    columns (`run_meta` hashes its cfg; it does not pass keys through). The runtime versions are
    columns only: they are provenance, not configuration, and hashing them would split one
    configuration across torch releases.
    """
    preprocess = getattr(method, "preprocess", None)
    resize_antialias = getattr(method, "resize_antialias", None)
    meta = run_meta({**dict(cfg), "preprocess": preprocess,
                     "resize_antialias": resize_antialias})
    return {**meta, "preprocess": preprocess, "resize_antialias": resize_antialias,
            **runtime_versions()}
