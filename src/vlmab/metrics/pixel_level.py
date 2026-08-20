"""Pixel-level metrics: P-AUROC, AU-PRO (@0.3 and @0.05), SegF1.

Pure numpy/scipy/sklearn so the suite runs in CI without torch. Inputs are always
sequences of HxW arrays: `masks` where nonzero means anomalous, `amaps` float scores.
"""
from typing import Sequence

import numpy as np
from scipy import ndimage
from sklearn import metrics as skm


def _check_pairs(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray]) -> None:
    if len(masks) != len(amaps):
        raise ValueError(f"masks/amaps length mismatch: {len(masks)} vs {len(amaps)}")
    for i, (m, a) in enumerate(zip(masks, amaps)):
        m_shape = np.asarray(m).shape
        a_shape = np.asarray(a).shape
        if m_shape != a_shape:
            raise ValueError(
                f"mask/amap shape mismatch at index {i}: {m_shape} vs {a_shape}"
            )


def _flatten(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray]):
    _check_pairs(masks, amaps)
    y = np.concatenate([(np.asarray(m) > 0).ravel() for m in masks])
    # float32 keeps peak memory bounded on ~5MP MVTec AD 2 images (Colab-class RAM);
    # both roc_auc_score and precision_recall_curve accept float32 inputs.
    s = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in amaps])
    return y, s


def p_auroc(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray]) -> float:
    """Pooled pixel AUROC.

    Raises on a single-class category rather than returning nan. `roc_auc_score`
    only warns and returns nan when one class is absent, and that nan propagates
    silently into any mean over categories -- a published column would end up nan,
    or be quietly dropped by a nan-skipping aggregation and reported as a real
    average over fewer categories than it claims.
    """
    y, s = _flatten(masks, amaps)
    n_anomalous = int(y.sum())
    if n_anomalous == 0 or n_anomalous == y.size:
        kind = "all-normal" if n_anomalous == 0 else "all-anomalous"
        raise ValueError(
            f"p_auroc is undefined for a single-class category: masks are {kind} "
            f"({n_anomalous} anomalous of {y.size} pooled pixels)"
        )
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
    of magnitude more expensive than the fixed 200-point grid the protocol assumes.
    Quantile sampling is chosen for that reason.

    What quantile sampling costs, stated precisely because an earlier version of
    this docstring understated it. Grid points can only ever land *on* normal-pixel
    values, so a threshold that falls strictly between two adjacent normal order
    statistics is unreachable at any `num_thresholds`. Where a region's scores sit
    in such a gap, the difference from an exact sweep does not shrink as the grid
    is refined: it is structural, not coarseness. On the deliberately small,
    hand-built fixture in `tests/test_aggregate.py` the gap is 5.65 AU-PRO points
    (0.8305 here against 0.8870 exact), identical at `num_thresholds` of 200,
    2_000, 50_000 and 500_000.

    That fixture is a worst case by construction -- a handful of pixels, so
    adjacent normal order statistics are far apart. On real data the error is
    bounded by the anomalous score mass lying between adjacent normal values, and
    with tens of millions of normal pixels those gaps are around one float16 ULP,
    so no reported number is affected. The bound to remember is therefore about the
    density of normal pixels, not about `num_thresholds`: this approximation is
    safe on full-resolution categories and unsafe on toy inputs.

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

    Degenerate input raises, identically at both FPR limits. These cases used to return
    0.0, which is indistinguishable from a genuine score of zero — a real Vial run reported
    `au_pro_005 = 0.0` for every lighting group and the number could not say whether the
    metric had been computed or abandoned. Refused, with the reason named:

    * no anomalous regions (PRO averages over regions; there are none) or no normal pixels
      (FPR is a fraction of them) — the same class of undefined input `p_auroc` refuses for
      a single-class category;
    * a constant anomaly map, where the whole curve collapses and the 0.0 that came out was
      an artifact of extending a single point flat to `fpr_limit`, not a measurement.

    Raising rather than returning nan is deliberate, for the reason spelled out in
    `p_auroc`: nan is silently dropped by any nan-skipping aggregation (`DataFrame.mean()`),
    so a published column would quietly become an average over fewer categories than it
    claims. In `aggregate`, this surfaces as a `ValueError` naming the group that failed
    (`group meta_lighting='regular': au_pro is undefined ...`), the same way a single-class
    or unlabelled group already does — no row is emitted with a fabricated 0.0.

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

    if not region_sorted_values:
        raise ValueError(
            "au_pro is undefined for a category with no anomalous regions: PRO is a mean "
            f"over ground-truth regions and these {len(masks)} mask(s) contain none. A "
            "shard of only `good` images cannot be localised."
        )
    if n_normal == 0:
        raise ValueError(
            "au_pro is undefined for a category with no normal pixels: FPR is a fraction "
            f"of them and all {sum(int(m.size) for m in masks)} pooled pixels are anomalous."
        )
    if lo == hi:
        raise ValueError(
            f"au_pro is undefined for a constant anomaly map: every one of the "
            f"{sum(int(a.size) for a in amaps)} pooled scores is {lo}, so no threshold "
            "separates anything and the curve has no operating points. This is a broken "
            "prediction, not a score of zero."
        )

    normal_sorted = (
        np.sort(np.concatenate(normal_chunks))
        if normal_chunks
        else np.empty(0, dtype=np.float32)
    )

    thresholds = pro_thresholds(normal_sorted, lo, fpr_limit, num_thresholds)
    if thresholds.size == 0:
        raise ValueError(
            f"au_pro found no usable threshold at fpr_limit={fpr_limit}: every candidate "
            "sits at or below the anomaly maps' minimum value, so the PRO curve has no "
            "points to integrate."
        )

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
        raise ValueError(
            f"au_pro found no operating point at or below fpr_limit={fpr_limit}: there is "
            "no curve to integrate, so no area can be reported."
        )
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    if x[-1] < fpr_limit:
        x = np.concatenate([x, [fpr_limit]])
        y = np.concatenate([y, [y[-1]]])

    return float(_trapezoid(y, x) / fpr_limit)


def _per_image_thresholds(
    amaps: Sequence[np.ndarray], threshold: float | Sequence[float]
) -> list[float]:
    if np.isscalar(threshold):
        return [float(threshold)] * len(amaps)
    ts = [float(t) for t in threshold]
    if len(ts) != len(amaps):
        raise ValueError(
            f"expected one threshold per image: got {len(ts)} thresholds for "
            f"{len(amaps)} maps"
        )
    return ts


def seg_f1_at(
    masks: Sequence[np.ndarray],
    amaps: Sequence[np.ndarray],
    threshold: float | Sequence[float],
) -> float:
    """Pixel F1 at a *fixed* threshold — the official MVTec AD 2 metric.

    Precision and recall are computed over the complete set of pooled pixels, not averaged
    over individual images (protocol §4, v0.2.10). `threshold` is a scalar, or one value per
    image for a per-image rule.

    Streams: at a fixed threshold nothing has to be sorted, only counted, so the working set
    is one mask plus one map at a time. That is why this can pool a whole category (the
    official definition) while `seg_f1max` cannot and stays per lighting condition -- the
    oracle sorts, this counts. Not interchangeable with `seg_f1max` in a results column:
    that one inspects the ground truth to choose its threshold and is strictly optimistic.

    Predicted positive is `amap >= threshold`, fixed so the degenerate cases are decidable.
    Returns 0.0 when the group has no anomalous pixels (mirroring `seg_f1max`) and when
    nothing is predicted, where precision is undefined and F1 with it.
    """
    _check_pairs(masks, amaps)
    ts = _per_image_thresholds(amaps, threshold)
    tp = fp = fn = 0
    for m, a, t in zip(masks, amaps, ts):
        y = np.asarray(m) > 0
        pred = np.asarray(a, dtype=np.float32) >= np.float32(t)
        tp += int(np.count_nonzero(pred & y))
        fp += int(np.count_nonzero(pred & ~y))
        fn += int(np.count_nonzero(~pred & y))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def normal_pixel_fpr_at(
    masks: Sequence[np.ndarray],
    amaps: Sequence[np.ndarray],
    threshold: float | Sequence[float],
) -> float:
    """Fraction of *normal* pixels flagged at this threshold.

    The realised false-positive rate, against the `alpha` a rule targeted on `validation`.
    The gap between the two is what a lighting shift does to a threshold, and is a reported
    result in its own right. Needs masks, so it is computable on `test_public` only -- never
    on the private splits, where it would be exactly the feedback protocol §7 forbids
    iterating against.
    """
    _check_pairs(masks, amaps)
    ts = _per_image_thresholds(amaps, threshold)
    fp = n_normal = 0
    for m, a, t in zip(masks, amaps, ts):
        normal = np.asarray(m) == 0
        pred = np.asarray(a, dtype=np.float32) >= np.float32(t)
        fp += int(np.count_nonzero(pred & normal))
        n_normal += int(np.count_nonzero(normal))
    if n_normal == 0:
        raise ValueError(
            "normal_pixel_fpr_at has no normal pixels to measure: every pooled pixel in "
            "this group is anomalous"
        )
    return fp / n_normal
