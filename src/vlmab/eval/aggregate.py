"""Turn result shards into metrics.

Kept separate from the runner on purpose: the runner streams predictions to disk one category
at a time and never holds a grid's worth of anomaly maps in memory. Aggregation is where maps
are read back, so it is also where the memory ceiling has to be enforced — Colab's free tier has
roughly 12.7 GB of host RAM, and a category of 2.66 MP images adds up fast.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN
from vlmab.metrics.image_level import i_ap, i_auroc, i_f1max
from vlmab.metrics.pixel_level import au_pro, p_auroc, seg_f1max


def image_metrics(df: pd.DataFrame) -> dict[str, float]:
    """I-AUROC, I-AP and I-F1max over a shard's image scores."""
    labels = df["label"].to_numpy()
    if (labels == LABEL_UNKNOWN).any():
        raise ValueError(
            f"{int((labels == LABEL_UNKNOWN).sum())} unlabelled rows (label "
            f"{LABEL_UNKNOWN}): these come from the private splits, whose ground truth is "
            "withheld. Scoring them would mean treating unknown as normal."
        )
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "image metrics need both normal and anomalous samples; this group has only "
            f"label {int(labels[0])}"
        )
    scores = df["image_score"].to_numpy(dtype=np.float64)
    return {
        "i_auroc": i_auroc(labels, scores),
        "i_ap": i_ap(labels, scores),
        "i_f1max": i_f1max(labels, scores),
        "n": int(len(df)),
    }


def _load_pair(row: Any) -> tuple[np.ndarray, np.ndarray]:
    """One (mask, anomaly map) pair. A row with no mask file has no anomalous pixels."""
    amap = np.load(row.map_path).astype(np.float32)
    if row.mask_path is None or (isinstance(row.mask_path, float) and np.isnan(row.mask_path)):
        return np.zeros(amap.shape, dtype=np.uint8), amap
    from PIL import Image

    with Image.open(row.mask_path) as im:
        mask = (np.asarray(im.convert("L")) > 0).astype(np.uint8)
    if mask.shape != amap.shape:
        raise ValueError(
            f"mask {row.mask_path} is {mask.shape} but its anomaly map is {amap.shape}"
        )
    return mask, amap


def pixel_metrics(df: pd.DataFrame, max_pixels: int = 400_000_000) -> dict[str, float]:
    """P-AUROC, SegF1max and AU-PRO at both FPR limits, over rows that saved a map.

    `max_pixels` is a guard, not a tuning knob: exceeding it raises rather than letting the
    session get OOM-killed with no diagnostic. Raise it deliberately if the machine can take it.
    """
    if "map_path" not in df.columns:
        return {"n": 0}
    with_maps = df[df["map_path"].notna()]
    if with_maps.empty:
        return {"n": 0}

    total = 0
    for path in with_maps["map_path"]:
        shape = np.load(path, mmap_mode="r").shape
        total += int(np.prod(shape))
    if total > max_pixels:
        raise ValueError(
            f"{total:,} pixels exceeds max_pixels={max_pixels:,}. Aggregate a smaller group "
            "(e.g. one lighting condition at a time) or raise the budget if this machine "
            "genuinely has the memory."
        )

    masks, amaps = [], []
    for row in with_maps.itertuples():
        mask, amap = _load_pair(row)
        masks.append(mask)
        amaps.append(amap)

    return {
        "p_auroc": p_auroc(masks, amaps),
        "seg_f1max": seg_f1max(masks, amaps),
        "au_pro_030": au_pro(masks, amaps, fpr_limit=0.3),
        "au_pro_005": au_pro(masks, amaps, fpr_limit=0.05),
        "n": int(len(with_maps)),
    }


def aggregate(
    df: pd.DataFrame,
    by: str | None = None,
    max_pixels: int = 400_000_000,
) -> pd.DataFrame:
    """One metrics row per group, or a single row when `by` is None.

    Grouping by `meta_lighting` is the headline use: MVTec AD 2 exists to measure robustness to
    lighting shift, and that question is only answerable per condition.
    """
    if by is not None and by not in df.columns:
        raise KeyError(f"no column {by!r} to group by; columns are {sorted(df.columns)}")

    groups = [(None, df)] if by is None else sorted(df.groupby(by), key=lambda kv: kv[0])

    records = []
    for key, group in groups:
        record: dict[str, Any] = {} if key is None else {by: key}
        record.update(image_metrics(group))
        pixels = pixel_metrics(group, max_pixels=max_pixels)
        record.update({k: v for k, v in pixels.items() if k != "n"})
        record["n_pixel_rows"] = pixels["n"]
        records.append(record)
    return pd.DataFrame(records)
