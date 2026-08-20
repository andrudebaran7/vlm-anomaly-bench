"""Turn result shards into metrics.

Kept separate from the runner on purpose: the runner streams predictions to disk one category
at a time and never holds a grid's worth of anomaly maps in memory. Aggregation is where maps
are read back, so it is also where the memory ceiling has to be enforced — Colab's free tier has
roughly 12.7 GB of host RAM, and a category of 2.66 MP images adds up fast.
"""
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN
from vlmab.metrics.image_level import i_ap, i_auroc, i_f1max
from vlmab.metrics.pixel_level import (
    au_pro, normal_pixel_fpr_at, p_auroc, seg_f1_at, seg_f1max,
)
from vlmab.threshold.artifact import VALUE_KEY
from vlmab.threshold.rules import RULES, thresholds_for


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


#: Peak process memory, in bytes per pooled pixel, that one `pixel_metrics` call costs.
#
# NOT the size of the arrays this module holds: those are only 5 B/px (a float32 anomaly map,
# 4 B/px after `_load_pair`'s upcast, plus a uint8 mask, 1 B/px, real or synthesised). The
# metrics themselves allocate an order of magnitude more on top, and that memory is resident
# at the same time as the maps, so a guard that ignores it does not guard anything:
#
#   * `_flatten` (used by `p_auroc` and `seg_f1max`) builds *pooled copies* of everything it
#     was handed — a bool array over all pixels and a fresh float32 concatenation of every
#     map — while the per-image lists stay alive in the caller;
#   * sklearn then sorts the pooled scores with an int64 `argsort` (8 B/px for the indices
#     plus a permuted copy of the scores) and accumulates float64 cumulative sums (8 B/px
#     each) inside `_binary_clf_curve`.
#
# Measured on this machine (peak RSS above baseline via /proc VmHWM, one clean interpreter per
# point, float16 maps on disk as the runner writes them, 8 images per workload):
#
#   whole pixel_metrics call: 64.6 B/px @0.5 Mpx, 62.7 @1, 62.5 @2, 60.9 @4, 60.8 @6, 60.7 @8
#   per metric, on top of the resident arrays: p_auroc 56.4, seg_f1max 36.9, au_pro 17.0
#
# The total tracks p_auroc's peak (5 + 56) because the metrics run one after another; the
# slow drift with size is fixed interpreter cost being amortised. Those figures are from the
# dev machine (Python 3.13); the same measurement on the CI runner (Python 3.11) peaks at
# ~66 B/px, because the transient set of sklearn's argsort/cumsum depends on the library
# build. A guard must round the peak UP, never down, so the constant is set above the whole
# observed spread with margin rather than at one machine's number. `test_bytes_per_pixel_
# covers_the_measured_peak` re-measures it within a tolerance band that absorbs that spread,
# so a real implementation change that raises the cost still fails while a 3.11-vs-3.13
# difference does not.
BYTES_PER_PIXEL = 80


def pixel_metrics(df: pd.DataFrame, max_bytes: int = 6_000_000_000) -> dict[str, float]:
    """P-AUROC, SegF1max and AU-PRO at both FPR limits, over rows that saved a map.

    `max_bytes` is a guard, not a tuning knob: exceeding it raises rather than letting the
    session get OOM-killed with no diagnostic.

    What it means, precisely: an estimate of this call's **peak resident memory above the
    caller's own baseline**, computed as `BYTES_PER_PIXEL` (80, measured — see the constant)
    times the number of pooled pixels across every map in `df`. It is not the size of the
    maps on disk (float16, 2 B/px) and not the size of the arrays this function holds
    (5 B/px); the metrics' own working set dominates both.

    How to size it: take the machine's *free* RAM, subtract what the rest of the process needs
    (interpreter, pandas/pyarrow, the shard being aggregated, whatever the notebook already
    holds), and leave a margin — this is an estimate of a peak, not a hard allocator limit. On
    Colab's free tier (~12.7 GB total, ~2 GB already gone before aggregation starts) anything
    above ~8 GB is a number the machine cannot honour.

    The 6 GB default is chosen against the real target rather than as a round number. Vial's
    `test_public` is 140 images of 1400x1900 = 372 Mpx = 29.8 GB, which cannot run on Colab at
    all and must be refused; one lighting condition of it is ~20 images = 53 Mpx = 4.2 GB,
    which is exactly the per-condition aggregation this module exists to do and must be
    allowed. 6 GB separates the two with room on both sides while staying under half of
    Colab's ceiling. Raise it deliberately, against measured free RAM, if a machine can take it.
    """
    if "map_path" not in df.columns:
        return {"n": 0}
    with_maps = df[df["map_path"].notna()]
    if with_maps.empty:
        return {"n": 0}

    total_pixels = 0
    for path in with_maps["map_path"]:
        total_pixels += int(np.prod(np.load(path, mmap_mode="r").shape))
    total_bytes = total_pixels * BYTES_PER_PIXEL
    if total_bytes > max_bytes:
        raise ValueError(
            f"{total_pixels:,} pooled pixels need about {total_bytes:,} bytes "
            f"({total_bytes / 1e9:.1f} GB) at peak ({BYTES_PER_PIXEL} bytes/pixel: the masks "
            f"and maps held at once, plus the pooled copies and int64/float64 working arrays "
            f"the metrics allocate on top of them), which exceeds max_bytes={max_bytes:,} "
            f"({max_bytes / 1e9:.1f} GB). Aggregate a smaller group (e.g. one lighting "
            "condition at a time) or raise the budget if this machine genuinely has the memory."
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
    max_bytes: int = 6_000_000_000,
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
        try:
            record.update(image_metrics(group))
            pixels = pixel_metrics(group, max_bytes=max_bytes)
        except ValueError as exc:
            if key is None:
                raise
            raise ValueError(f"group {by}={key!r}: {exc}") from exc
        record.update({k: v for k, v in pixels.items() if k != "n"})
        record["n_pixel_rows"] = pixels["n"]
        records.append(record)
    return pd.DataFrame(records)


def threshold_metrics(
    df: pd.DataFrame,
    artifact: Mapping[str, Any],
    category: str,
) -> dict[str, float]:
    """SegF1 at each pre-registered rule, and the FPR each one actually realised.

    Kept out of `pixel_metrics` on purpose. `pixel_metrics` sorts (P-AUROC, SegF1max, AU-PRO),
    so it carries a memory guard and is aggregated per lighting condition. At a fixed threshold
    nothing is sorted, only counted: `seg_f1_at` itself streams, while this function holds the
    (mask, map) pairs at about 5 bytes per pixel -- roughly 1.9 GB for a 140-image Vial category
    -- rather than the ~80 bytes/pixel sorting peak that forces `pixel_metrics` to keep its
    memory guard and stay per lighting condition. That is cheap enough to pool a whole category,
    which is what the official SegF1 definition requires (protocol §4). No `max_bytes`
    parameter, because there is nothing here to guard against.

    Needs ground truth, so it is for `test_public` only. `seg_f1_at` alongside `seg_f1max` in
    a table is fine and is the point; in the same *column* without marking is not (protocol §4).
    """
    if "map_path" not in df.columns:
        raise ValueError(
            "threshold_metrics needs a map_path column: it scores thresholded anomaly maps, "
            "so a run without saved maps has nothing for it to threshold"
        )

    # Guard the other side of the join. `category` names which of the artifact's blocks to
    # read, but nothing upstream stops a caller handing in a df that does not actually match
    # it -- a multi-category results root pools every category's maps and cuts them all at
    # this one's threshold; a df for a different method or dataset gets scored at a threshold
    # fitted on a different method's or dataset's maps entirely, silently, because IDENTITY_
    # COLUMNS (vlmab.eval.store) are present but never checked against the artifact.
    for column in ("category", "method", "dataset"):
        if column not in df.columns:
            raise ValueError(
                f"threshold_metrics needs a {column!r} column to confirm the df it was handed "
                f"actually matches the {category!r} category it is about to be scored against"
            )
    found_categories = sorted(df["category"].unique())
    if found_categories != [category]:
        raise ValueError(
            f"threshold_metrics was asked for category {category!r} but df contains category "
            f"{found_categories}: pooling other categories' maps into this call would cut them "
            f"at {category!r}'s threshold, which is fit on a different distribution of scores"
        )
    artifact_method = artifact["method"]
    found_methods = sorted(df["method"].unique())
    if found_methods != [artifact_method]:
        raise ValueError(
            f"threshold_metrics got method(s) {found_methods} but the calibration artifact was "
            f"fitted for method {artifact_method!r}: anomaly maps are on each method's own "
            "scale (protocol §4) and a threshold fitted for one method is meaningless applied "
            "to another's"
        )
    artifact_dataset = artifact["dataset"]
    found_datasets = sorted(df["dataset"].unique())
    if found_datasets != [artifact_dataset]:
        raise ValueError(
            f"threshold_metrics got dataset(s) {found_datasets} but the calibration artifact "
            f"was fitted for dataset {artifact_dataset!r}: a threshold fitted on one dataset's "
            "validation split says nothing about another's"
        )

    labels = df["label"].to_numpy()
    if (labels == LABEL_UNKNOWN).any():
        raise ValueError(
            f"{int((labels == LABEL_UNKNOWN).sum())} unlabelled rows (label {LABEL_UNKNOWN}): "
            "these come from the private splits, whose ground truth is withheld. SegF1 and the "
            "realised FPR both need masks, and synthesising them would score unknown as normal."
        )
    if category not in artifact["categories"]:
        raise KeyError(
            f"the calibration artifact has no thresholds for category {category!r}; it "
            f"calibrated {sorted(artifact['categories'])}"
        )

    rows = list(df[df["map_path"].notna()].itertuples())
    if not rows:
        return {"n": 0}
    masks, amaps = zip(*(_load_pair(row) for row in rows))
    # One map at a time, re-read per pass: this is the streaming path, so holding the pair
    # list is already the peak and the factory adds nothing to it.
    factory = lambda: iter(amaps)

    block = artifact["categories"][category]
    alpha = float(artifact["alpha"])
    out: dict[str, float] = {}
    for rule in RULES:
        key = VALUE_KEY.get(rule)
        value = None if key is None else float(block[rule][key])
        ts, degenerate = thresholds_for(rule, value, factory, alpha)
        out[f"seg_f1_at__{rule}"] = seg_f1_at(masks, amaps, ts)
        out[f"fpr_at__{rule}"] = normal_pixel_fpr_at(masks, amaps, ts)
        out[f"degenerate_mad__{rule}"] = degenerate
    out["n"] = len(rows)
    return out
