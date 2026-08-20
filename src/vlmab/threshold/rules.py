"""Ground-truth-free threshold rules: one primitive, two axes.

Every rule here has the same shape -- transform the scores, pool them, cut at a quantile --
and differs only in the transform (`identity` or `robust_z`) and the calibration source
(the defect-free `validation` split, or the unlabelled test split itself). See
docs/superpowers/specs/2026-08-20-threshold-rule-design.md.

Pure numpy, no I/O: callers hand in a factory that yields anomaly maps, so this module never
learns where maps are stored.
"""
from dataclasses import dataclass
from typing import Callable, Iterable

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


#: The three pre-registered candidates (protocol §4, v0.2.11). `per_image_robust_z` is the
#: designated submission rule; the other two are reported on `test_public` for comparison.
RULES = ("global_quantile", "per_image_robust_z", "transductive_quantile")


@dataclass(frozen=True)
class Calibration:
    """What a rule fitted on `validation`, and what it had to skip to fit it."""

    rule: str
    value: float          # the threshold itself for global_quantile; k for per_image_robust_z
    alpha: float
    n_pixels: int         # pixels that actually contributed, after skipping degenerate images
    degenerate_mad: int   # images with MAD == 0, excluded from the fit


def robust_z_stats(amap: np.ndarray) -> tuple[float, float]:
    """`(median, MAD)` of one map. The MAD's breakdown point is 50%, so a defect has to
    cover half the image before it moves these -- which is what lets a per-image threshold
    absorb a lighting shift without absorbing the anomaly along with it."""
    v = np.asarray(amap, dtype=np.float64).ravel()
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med)))


def calibrate(
    rule: str,
    maps_factory: Callable[[], Iterable[np.ndarray]],
    alpha: float,
) -> Calibration:
    """Fit `rule` on defect-free maps. `maps_factory()` must yield a *fresh* iterator each
    call: this makes two passes, one to count pixels and one to stream values, and the two
    must agree.

    `alpha` is the target false-positive rate on normal pixels -- the only quantity a
    defect-free split can speak to, since it contains no positives to compute F1 against.
    """
    if rule == "transductive_quantile":
        raise ValueError(
            "transductive_quantile fits nothing on validation; it is resolved at apply time "
            "from the test split's own scores -- call thresholds_for() instead"
        )
    if rule == "global_quantile":
        n_pixels = sum(int(np.asarray(a).size) for a in maps_factory())
        if n_pixels == 0:
            raise ValueError("global_quantile got no maps to calibrate on")
        value = topk_quantile(
            (np.asarray(a, dtype=np.float64).ravel() for a in maps_factory()), n_pixels, alpha
        )
        return Calibration(rule, value, alpha, n_pixels, 0)
    if rule == "per_image_robust_z":
        stats: list[tuple[float, float] | None] = []
        n_pixels = 0
        degenerate = 0
        for a in maps_factory():
            med, mad = robust_z_stats(a)
            if mad == 0.0:
                # A constant map has no scale to standardise by. Skipping it is the only
                # defined choice, and it is counted rather than silently dropped.
                degenerate += 1
                stats.append(None)
                continue
            stats.append((med, mad))
            n_pixels += int(np.asarray(a).size)
        if n_pixels == 0:
            raise ValueError(
                f"per_image_robust_z has nothing to calibrate on: all {degenerate} maps are "
                "constant (MAD == 0)"
            )

        def standardised():
            for a, st in zip(maps_factory(), stats):
                if st is None:
                    continue
                med, mad = st
                yield (np.asarray(a, dtype=np.float64).ravel() - med) / mad

        return Calibration(rule, topk_quantile(standardised(), n_pixels, alpha), alpha, n_pixels, degenerate)
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def thresholds_for(
    rule: str,
    value: float | None,
    maps_factory: Callable[[], Iterable[np.ndarray]],
    alpha: float,
) -> tuple[list[float], int]:
    """One threshold per map, ready to hand to `seg_f1_at`, plus the degenerate-MAD count.

    `value` is the calibrated scalar (`None` for transductive_quantile, which has none).
    """
    if rule == "global_quantile":
        if value is None:
            raise ValueError("global_quantile needs its calibrated threshold, got None")
        return [float(value)] * sum(1 for _ in maps_factory()), 0
    if rule == "per_image_robust_z":
        if value is None:
            raise ValueError("per_image_robust_z needs its calibrated k, got None")
        thresholds: list[float] = []
        degenerate = 0
        for a in maps_factory():
            med, mad = robust_z_stats(a)
            if mad == 0.0:
                # No scale, so no defensible cut: flag nothing and say so, rather than
                # dividing by zero or letting the image pass with an arbitrary threshold.
                thresholds.append(np.inf)
                degenerate += 1
            else:
                thresholds.append(med + float(value) * mad)
        return thresholds, degenerate
    if rule == "transductive_quantile":
        n_images = 0
        n_pixels = 0
        for a in maps_factory():
            n_images += 1
            n_pixels += int(np.asarray(a).size)
        if n_pixels == 0:
            raise ValueError("transductive_quantile got no maps to cut on")
        t = topk_quantile(
            (np.asarray(a, dtype=np.float64).ravel() for a in maps_factory()), n_pixels, alpha
        )
        return [t] * n_images, 0
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
