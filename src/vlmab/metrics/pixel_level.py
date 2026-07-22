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
    for i, (m, a) in enumerate(zip(masks, amaps)):
        m_shape = np.asarray(m).shape
        a_shape = np.asarray(a).shape
        if m_shape != a_shape:
            raise ValueError(
                f"mask/amap shape mismatch at index {i}: {m_shape} vs {a_shape}"
            )
    y = np.concatenate([(np.asarray(m) > 0).ravel() for m in masks])
    # float32 keeps peak memory bounded on ~5MP MVTec AD 2 images (Colab-class RAM);
    # both roc_auc_score and precision_recall_curve accept float32 inputs.
    s = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in amaps])
    return y, s


def p_auroc(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray]) -> float:
    y, s = _flatten(masks, amaps)
    return float(skm.roc_auc_score(y.astype(np.uint8), s))


def seg_f1max(
    masks: Sequence[np.ndarray],
    amaps: Sequence[np.ndarray],
) -> float:
    """Best achievable pixel F1 over thresholds.

    Vectorized the same way as `i_f1max` in `image_level.py`: sklearn's
    `precision_recall_curve` sorts the pooled pixel scores once and returns
    precision/recall at every unique score value, which we turn into F1 and
    take the max. This replaces a Python loop over a fixed 200-point
    threshold grid with a single sort — one pass instead of 200 full-array
    boolean passes over the pooled pixel arrays — and it evaluates every
    achievable threshold rather than a 200-point approximation of them, so
    it is strictly more precise, not an approximation.

    Not yet mapped onto the official MVTec AD 2 SegF1 definition — that mapping waits on the
    server's submission documentation (docs/datasets-access.md).
    """
    y, s = _flatten(masks, amaps)
    if not y.any():
        return 0.0
    prec, rec, _ = skm.precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(np.nanmax(f1))
