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


def pro_thresholds(
    normal_sorted: np.ndarray,
    score_min: float,
    fpr_limit: float,
    num_thresholds: int,
) -> np.ndarray:
    """Thresholds spaced uniformly in FPR across `[0, fpr_limit]`.

    This is a metric-definition decision, so it is stated here in full rather than
    left implicit in the code.

    AU-PRO integrates PRO over FPR, so the grid's job is to resolve the interval
    `[0, fpr_limit]` in *FPR*. Spacing thresholds uniformly in *score* (the obvious
    `np.linspace(score_min, score_max, n)`) does not do that, because score is not
    linear in FPR: anomaly maps are heavy-tailed, and a single bright outlier
    stretches `score_max` far beyond where any meaningful mass of normal pixels
    sits. On a weak-separation fixture with such an outlier, a 200-point uniform
    score grid puts only about six points inside FPR in (0, 0.05] and reports
    AU-PRO@0.05 = 0.7369 where the converged value is 0.8324 -- 9.6 points of
    discretisation error, roughly ten times the protocol's +-1.0 tolerance, and
    biased *downward* (the curve is concave, so a coarse left-heavy grid
    under-integrates).

    FPR at threshold `t` is just the tail fraction of the pooled normal-pixel
    values, so the threshold achieving a target FPR `f` is the corresponding
    quantile of `normal_sorted`: keep the `k = floor(f * N)` largest normal values
    above the threshold by taking `normal_sorted[N - k]`. That makes every grid
    point land at a known, evenly-spaced FPR, and the achieved FPR differs from the
    target by at most one pixel (1/N) plus whatever ties force.

    The alternative -- sweeping every distinct score value -- is exact rather than
    merely convergent, but its cost scales with the number of distinct scores. A
    single 5MP MVTec AD 2 image with float32 maps yields millions of them, and PRO
    is evaluated per region at every threshold, so the sweep would be several orders
    of magnitude more expensive than the fixed 200-point grid the protocol assumes,
    while moving the result by at most ~2e-3 (0.2 AU-PRO points) on the hardest
    fixture in the suite and by exactly 0.0 on every other one. Quantile sampling is
    chosen for that reason.

    Two deliberate properties are preserved:

    * `f = 0` maps to a threshold strictly above every normal pixel
      (`nextafter(max_normal)`) rather than to `score_max`, which anchors the curve
      at exactly FPR 0 without assuming anything about where the maximum sits.
    * Thresholds at or below the global score minimum are dropped, so a degenerate
      all-positive prediction never enters the curve.
    """
    n = normal_sorted.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    targets = np.linspace(0.0, float(fpr_limit), int(num_thresholds) + 1)
    # k = how many normal pixels are allowed to sit above the threshold at this FPR.
    k = np.floor(targets * n).astype(np.int64)
    idx = n - k

    thresholds = np.empty(idx.size, dtype=np.float64)
    above_all = idx >= n  # k == 0: the threshold must exceed every normal pixel
    thresholds[above_all] = np.nextafter(float(normal_sorted[-1]), np.inf)
    thresholds[~above_all] = normal_sorted[np.clip(idx[~above_all], 0, n - 1)]

    thresholds = np.unique(thresholds)
    return thresholds[thresholds > score_min]


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

    Connectivity: regions are labelled with 8-connectivity (`ndimage.label` with a full
    3x3 structuring element), matching the AU-PRO literature and reference
    implementations (e.g. skimage's `measure.label`, whose 2D default is full
    connectivity). Plain `ndimage.label(m)` defaults to 4-connectivity, which would
    split a single diagonal scratch into many one-pixel "regions" that each get equal
    weight in PRO — a real distortion of the metric, not a cosmetic difference.

    Thresholds are spaced uniformly in FPR across `[0, fpr_limit]`, not uniformly in
    score -- see `pro_thresholds` for why that choice is forced by what AU-PRO
    integrates over, and `test_au_pro_has_converged_at_the_default_threshold_count`
    for the convergence it buys. The sweep still excludes any threshold at or below
    the minimum anomaly value, so a degenerate all-positive prediction never enters
    the curve. When the curve never reaches `fpr_limit`, its last PRO value is
    extended flat to the limit.

    Implementation note (performance): rather than recomputing `amap >= t` over full
    images for every threshold and every region (O(n_regions * n_thresholds *
    image_pixels)), we exploit the fact that "fraction of a fixed pixel set >= t" is
    monotone in `t`. For each region we extract its anomaly values once and sort them;
    "count >= t" is then a `searchsorted` lookup, O(log n) per threshold instead of
    O(image_pixels). The same trick applies to the pooled normal-pixel values used for
    FPR. This is an exact reformulation, not an approximation — see
    `test_au_pro_matches_naive_reference` for a proof against a naive per-threshold
    implementation.
    """
    if len(masks) != len(amaps):
        raise ValueError(f"masks/amaps length mismatch: {len(masks)} vs {len(amaps)}")

    masks = [np.asarray(m) > 0 for m in masks]
    amaps = [np.asarray(a, dtype=np.float32) for a in amaps]

    struct8 = np.ones((3, 3), dtype=int)
    region_sorted_values: list[np.ndarray] = []
    lo = np.inf
    hi = -np.inf
    normal_chunks: list[np.ndarray] = []
    n_normal = 0
    for m, a in zip(masks, amaps):
        labelled, n = ndimage.label(m, structure=struct8)
        for r in range(1, n + 1):
            region_sorted_values.append(np.sort(a[labelled == r]))
        normal_vals = a[~m]
        if normal_vals.size:
            normal_chunks.append(normal_vals)
        n_normal += int((~m).sum())
        if a.size:
            a_min = float(a.min())
            a_max = float(a.max())
            if a_min < lo:
                lo = a_min
            if a_max > hi:
                hi = a_max

    if not region_sorted_values or n_normal == 0:
        return 0.0
    if lo == hi:
        return 0.0

    normal_sorted = (
        np.sort(np.concatenate(normal_chunks))
        if normal_chunks
        else np.empty(0, dtype=np.float32)
    )

    thresholds = pro_thresholds(normal_sorted, lo, fpr_limit, num_thresholds)
    if thresholds.size == 0:
        return 0.0

    # Same per-region searchsorted formulation as before ("count >= t" is a lookup in
    # the region's sorted values, not a pass over the image), evaluated for the whole
    # threshold vector at once. Peak memory stays O(n_thresholds), not
    # O(n_regions * n_thresholds), which matters now that num_thresholds is a knob the
    # convergence test drives to 50k.
    pros = np.zeros(thresholds.size, dtype=np.float64)
    for v in region_sorted_values:
        pros += (v.size - np.searchsorted(v, thresholds, side="left")) / v.size
    pros /= len(region_sorted_values)

    fp_counts = normal_sorted.size - np.searchsorted(
        normal_sorted, thresholds, side="left"
    )
    fprs = fp_counts / n_normal

    fpr = np.asarray(fprs, dtype=np.float64)
    pro = pros
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
