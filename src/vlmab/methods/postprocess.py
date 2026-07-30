"""Map post-processing shared by every adapter.

`upsample_to` is the scoring-path utility, mandated by the protocol: it brings an anomaly map
to the input image's *native* resolution (§4 evaluates pixel metrics there against unmodified
masks, so a coarse or resized map has to be expanded, never the mask shrunk). Scored maps are
NOT brought into [0,1] — protocol v0.2.6 keeps them on the method's own consistent scale, since
per-image [0,1] normalisation would break the cross-image pixel-metric ranking.

`normalise_to_unit` does min-max scores into [0,1], but it is visualisation-only; see its own
docstring for why it must not be used on the scoring path.
"""
import numpy as np


def upsample_to(coarse: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-block upsample `coarse` to `size=(H, W)`, exact for any target.

    Each output pixel takes the value of the coarse cell it falls in, so no value the method
    did not emit is ever invented — the right behaviour for a hard localisation claim like the
    MLLM's 7x7 grid. Smooth interpolation is a per-method choice left to the wrappers that
    produce dense feature maps (they have torch's interpolate for it); this default stays
    numpy-only and faithful.
    """
    coarse = np.asarray(coarse, dtype=np.float32)
    gh, gw = coarse.shape
    h, w = size
    ys = (np.arange(h) * gh) // h
    xs = (np.arange(w) * gw) // w
    return coarse[ys][:, xs]


def normalise_to_unit(x: np.ndarray) -> np.ndarray:
    """Min-max `x` into [0,1] as float32. All-equal input -> all zeros (no signal).

    Do NOT use this to scale an anomaly map for scoring: per-image [0,1] normalisation breaks the
    cross-image pixel-metric ranking (protocol v0.2.6). It is for genuinely bounded quantities and
    visualisation only.
    """
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("map contains non-finite values; a NaN/inf map is a method bug")
    lo = float(x.min())
    hi = float(x.max())
    if hi == lo:
        return np.zeros_like(x)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def distinct_levels(amap: np.ndarray) -> int:
    """How many distinct values an anomaly map contains — a diagnostic, never a metric.

    Mask-based methods (SAA+: GroundingDINO region proposals refined by SAM) emit a map composed of a
    handful of constant-confidence regions, so it has a handful of distinct levels. Patch-based methods
    emit a near-continuous field. Every threshold-sweeping pixel metric (P-AUROC, AU-PRO, SegF1) is
    sensitive to that difference, so a low count is a confound to report alongside the metric rather
    than a result to explain away (paper §5.2).

    Counts exact distinct float32 values with no tolerance bucketing, so the figure is unambiguous and
    reproducible.
    """
    return int(np.unique(np.asarray(amap, dtype=np.float32)).size)
