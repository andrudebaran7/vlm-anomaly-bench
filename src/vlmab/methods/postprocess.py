"""Map post-processing shared by every adapter.

Two jobs, both mandated by the protocol: bring an anomaly map to the input image's *native*
resolution (§4 evaluates pixel metrics there against unmodified masks, so a coarse or resized
map has to be expanded, never the mask shrunk), and bring scores into [0,1].
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
    """Min-max `x` into [0,1] as float32. All-equal input -> all zeros (no anomaly signal)."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("map contains non-finite values; a NaN/inf map is a method bug")
    lo = float(x.min())
    hi = float(x.max())
    if hi == lo:
        return np.zeros_like(x)
    return ((x - lo) / (hi - lo)).astype(np.float32)
