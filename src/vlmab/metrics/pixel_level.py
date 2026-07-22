"""Pixel-level metrics: P-AUROC, AU-PRO (@0.3 and @0.05), SegF1.

Pure numpy/scipy/sklearn so the suite runs in CI without torch. Inputs are always
sequences of HxW arrays: `masks` where nonzero means anomalous, `amaps` float scores.
"""
from typing import Sequence

import numpy as np
from scipy import ndimage
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


# np.trapz is deprecated in numpy 2.x, np.trapezoid does not exist in 1.x.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def au_pro(
    masks: Sequence[np.ndarray],
    amaps: Sequence[np.ndarray],
    fpr_limit: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """Area under the per-region-overlap curve, integrated to `fpr_limit` and normalised.

    Every connected ground-truth region contributes equally to PRO regardless of its area,
    which is the whole point of the metric: it refuses to let one large defect mask the
    failure to find several small ones.

    The threshold sweep excludes the minimum anomaly value, so a degenerate all-positive
    prediction never enters the curve. When the curve never reaches `fpr_limit`, its last
    PRO value is extended flat to the limit.
    """
    if len(masks) != len(amaps):
        raise ValueError(f"masks/amaps length mismatch: {len(masks)} vs {len(amaps)}")

    masks = [np.asarray(m) > 0 for m in masks]
    amaps = [np.asarray(a, dtype=np.float64) for a in amaps]

    regions: list[tuple[int, np.ndarray]] = []
    for i, m in enumerate(masks):
        labelled, n = ndimage.label(m)
        for r in range(1, n + 1):
            regions.append((i, labelled == r))

    n_normal = int(sum((~m).sum() for m in masks))
    if not regions or n_normal == 0:
        return 0.0

    all_values = np.concatenate([a.ravel() for a in amaps])
    lo, hi = float(all_values.min()), float(all_values.max())
    if lo == hi:
        return 0.0

    pros, fprs = [], []
    for t in np.linspace(lo, hi, num_thresholds + 1)[1:]:
        binaries = [a >= t for a in amaps]
        pros.append(
            float(np.mean([
                np.count_nonzero(binaries[i] & reg) / np.count_nonzero(reg)
                for i, reg in regions
            ]))
        )
        fp = sum(int(np.count_nonzero(binaries[i] & ~masks[i])) for i in range(len(masks)))
        fprs.append(fp / n_normal)

    fpr = np.asarray(fprs, dtype=np.float64)
    pro = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fpr, kind="stable")
    fpr, pro = fpr[order], pro[order]

    keep = fpr <= fpr_limit
    x, y = fpr[keep], pro[keep]
    if x.size == 0:
        return 0.0
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    if x[-1] < fpr_limit:
        x = np.concatenate([x, [fpr_limit]])
        y = np.concatenate([y, [y[-1]]])

    return float(_trapezoid(y, x) / fpr_limit)
