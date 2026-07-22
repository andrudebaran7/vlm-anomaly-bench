"""Pixel-level metrics: P-AUROC, AU-PRO (@0.3 and @0.05), SegF1.

Pure numpy/scipy/sklearn so the suite runs in CI without torch. Inputs are always
sequences of HxW arrays: `masks` where nonzero means anomalous, `amaps` float scores.
"""
from typing import Sequence

import numpy as np
from sklearn import metrics as skm


def _flatten(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray]):
    if len(masks) != len(amaps):
        raise ValueError(f"masks/amaps length mismatch: {len(masks)} vs {len(amaps)}")
    y = np.concatenate([(np.asarray(m) > 0).ravel() for m in masks])
    s = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in amaps])
    if y.shape != s.shape:
        raise ValueError("mask and anomaly map shapes differ")
    return y, s


def p_auroc(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray]) -> float:
    y, s = _flatten(masks, amaps)
    return float(skm.roc_auc_score(y.astype(np.uint8), s))


def seg_f1max(
    masks: Sequence[np.ndarray],
    amaps: Sequence[np.ndarray],
    num_thresholds: int = 200,
) -> float:
    """Best achievable pixel F1 over thresholds.

    Not yet mapped onto the official MVTec AD 2 SegF1 definition — that mapping waits on the
    server's submission documentation (docs/datasets-access.md).
    """
    y, s = _flatten(masks, amaps)
    if not y.any():
        return 0.0
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        return 0.0
    best = 0.0
    for t in np.linspace(lo, hi, num_thresholds + 1)[1:]:
        p = s >= t
        tp = int(np.count_nonzero(p & y))
        fp = int(np.count_nonzero(p & ~y))
        fn = int(np.count_nonzero(~p & y))
        denom = 2 * tp + fp + fn
        if denom:
            best = max(best, 2 * tp / denom)
    return float(best)
