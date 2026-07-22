"""Latency / params / VRAM instrumentation.

Latency numbers are only valid on the reference machine (results/ENVIRONMENT.md) — protocol §5
forbids reporting latency measured in a Colab session. Torch is imported lazily so this module,
and its tests, work in a CI environment without it. API methods report tokens + cost via
Prediction.extras instead of VRAM.
"""
from typing import Any, Sequence

import numpy as np


def summarize_latencies(times_ms: Sequence[float]) -> dict[str, float]:
    """Median and p95 of per-image latencies, as protocol §4 requires."""
    a = np.asarray(list(times_ms), dtype=np.float64)
    if a.size == 0:
        raise ValueError("no timings to summarize")
    return {
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "n": int(a.size),
    }


def count_parameters(model: Any) -> int:
    """Total parameter count. Duck-typed on .parameters() so it needs no torch import."""
    return int(sum(p.numel() for p in model.parameters()))


def _cuda():
    """Return torch.cuda if torch is installed and a GPU is present, else None."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


def peak_vram_mb() -> float | None:
    """Peak allocated VRAM in MiB since the last reset, or None when unavailable."""
    cuda = _cuda()
    if cuda is None:
        return None
    return float(cuda.max_memory_allocated() / (1024 ** 2))


def reset_peak_vram() -> None:
    """Reset the peak-VRAM counter. A no-op when torch or CUDA is unavailable."""
    cuda = _cuda()
    if cuda is not None:
        cuda.reset_peak_memory_stats()
