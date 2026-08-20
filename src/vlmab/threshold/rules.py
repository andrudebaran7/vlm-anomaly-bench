"""Ground-truth-free threshold rules: one primitive, two axes.

Every rule here has the same shape -- transform the scores, pool them, cut at a quantile --
and differs only in the transform (`identity` or `robust_z`) and the calibration source
(the defect-free `validation` split, or the unlabelled test split itself). See
docs/superpowers/specs/2026-08-20-threshold-rule-design.md.

Pure numpy, no I/O: callers hand in a factory that yields anomaly maps, so this module never
learns where maps are stored.
"""
from typing import Iterable

import numpy as np


def topk_quantile(chunks: Iterable[np.ndarray], n_total: int, alpha: float) -> float:
    """The `(1 - alpha)` quantile of a pooled stream, exactly, in O(k) memory.

    That quantile is the k-th largest value with `k = ceil(alpha * n_total)`, so only the
    largest k values ever need to be resident. For Vial's validation split at alpha=1e-3
    that is ~109,000 float64s (0.4 MB) against 436 MB to pool the split -- which is why
    calibration needs no memory guard.

    `n_total` must be the true pooled size; the caller gets it from a shape-only pass
    (`np.load(..., mmap_mode="r").shape`), the pattern `aggregate.pixel_metrics` already uses.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]: got {alpha}")
    if n_total <= 0:
        raise ValueError(f"n_total must be positive: got {n_total}")

    k = max(1, min(int(np.ceil(alpha * n_total)), n_total))
    buf = np.empty(0, dtype=np.float64)
    seen = 0
    for chunk in chunks:
        v = np.asarray(chunk, dtype=np.float64).ravel()
        seen += v.size
        buf = np.concatenate([buf, v])
        if buf.size > k:
            buf = np.partition(buf, buf.size - k)[buf.size - k :]
    if seen != n_total:
        raise ValueError(
            f"stream held {seen} values but n_total said {n_total}; the counting pass and "
            "the value pass disagree, so the quantile would be computed at the wrong rank"
        )
    return float(np.partition(buf, buf.size - k)[buf.size - k])
