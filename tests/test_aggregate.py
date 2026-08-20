import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN
from vlmab.eval.aggregate import (
    BYTES_PER_PIXEL, aggregate, image_metrics, pixel_metrics, threshold_metrics,
)
from vlmab.metrics.image_level import i_ap, i_auroc, i_f1max
from vlmab.metrics.pixel_level import au_pro, p_auroc, seg_f1max
from vlmab.threshold.artifact import load_artifact, write_artifact
from vlmab.threshold.rules import Calibration


def _shard(tmp_path, n=4, separable=True, with_pixels=True):
    """A shard whose scores separate perfectly, with matching masks and maps on disk."""
    rows = []
    for i in range(n):
        label = i % 2
        row = {
            "image_path": f"/fake/{i:03d}_regular.png",
            "label": label,
            "image_score": 0.9 if (label and separable) else 0.1,
            "split": "test_public",
            "meta_lighting": "regular" if i < n // 2 else "overexposed",
            "mask_path": None,
            "map_path": None,
        }
        if with_pixels:
            amap = np.zeros((6, 8), dtype=np.float16)
            mask = np.zeros((6, 8), dtype=np.uint8)
            if label:
                amap[:2, :2] = 1.0
                mask[:2, :2] = 255
                mask_path = tmp_path / f"{i:03d}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                row["mask_path"] = str(mask_path)
            map_path = tmp_path / f"{i:03d}.npy"
            np.save(map_path, amap)
            row["map_path"] = str(map_path)
        rows.append(row)
    return pd.DataFrame(rows)


def _imperfect_shard(tmp_path):
    """A shard on which every metric takes a *different*, hand-checkable value.

    `_shard` is perfectly separable, so all six metrics evaluate to 1.0 on it and no test
    built on it can tell one metric column from another: swapping i_auroc with i_ap, or the
    two AU-PRO FPR limits, or the mask/map arguments to p_auroc, leaves 1.0 == 1.0 everywhere.
    This fixture is deliberately imperfect instead.

    Image level (4 images, labels 0,1,0,1): scores 0.10, 0.90, 0.60, 0.40. One anomalous image
    (0.40) is ranked below one normal image (0.60), so separation is partial.

    Pixel level (4 images of 4x4 = 64 pooled pixels), with deliberately unequal regions:
      * image 1 (anomalous): a 2x2 region, 4 pixels, all scored 0.9 -- found easily;
      * image 3 (anomalous): a single-pixel region scored 0.5 -- found only at a threshold
        that already admits false positives;
      * images 0 and 2 (normal, no mask file): 4 pixels at 0.7 and 4 pixels at 0.45
        respectively, which are the false positives that sit between the two regions;
      * every other pixel is 0.1.
    So the pooled pixels are 4 anomalous @0.9, 1 anomalous @0.5, and 59 normal: 4 @0.7,
    4 @0.45, 51 @0.1. Region sizes differ by 4x, which is what makes AU-PRO (equal weight per
    region) differ from the pixel-pooled metrics.

    Returns (df, masks, amaps) so a test can recompute the expected numbers by calling the
    metric functions directly on the same arrays, independently of what `pixel_metrics` did.
    """
    scores = [0.10, 0.90, 0.60, 0.40]
    amaps_f16, masks = [], []
    for i in range(4):
        amap = np.full((4, 4), 0.1, dtype=np.float16)
        mask = np.zeros((4, 4), dtype=np.uint8)
        if i == 0:
            amap[0, :] = 0.7  # 4 normal pixels above the weak region's score
        elif i == 1:
            amap[0:2, 0:2] = 0.9  # 4-pixel region, strongly detected
            mask[0:2, 0:2] = 1
        elif i == 2:
            amap[0, :] = 0.45  # 4 normal pixels just below the weak region's score
        else:
            amap[0, 0] = 0.5  # 1-pixel region, weakly detected
            mask[0, 0] = 1
        amaps_f16.append(amap)
        masks.append(mask)

    rows = []
    for i, (amap, mask) in enumerate(zip(amaps_f16, masks)):
        map_path = tmp_path / f"imperfect_{i}.npy"
        np.save(map_path, amap)
        row = {
            "image_path": f"/fake/{i:03d}_regular.png",
            "label": int(mask.any()),
            "image_score": scores[i],
            "split": "test_public",
            "meta_lighting": "regular",
            "mask_path": None,
            "map_path": str(map_path),
        }
        if mask.any():
            mask_path = tmp_path / f"imperfect_{i}_mask.png"
            Image.fromarray(mask * 255, mode="L").save(mask_path)
            row["mask_path"] = str(mask_path)
        rows.append(row)
    # float32 is what _load_pair hands the metrics, so the independent recomputation must use
    # the same dtype rather than the float16 on disk.
    return pd.DataFrame(rows), masks, [a.astype(np.float32) for a in amaps_f16]


def test_image_metrics_are_not_interchangeable_columns(tmp_path):
    """Each image-level column must carry its own metric, checked against a hand derivation.

    Ranking on `_imperfect_shard` is, descending: 0.90 (anomalous), 0.60 (normal),
    0.40 (anomalous), 0.10 (normal).

    * I-AUROC counts concordant (anomalous, normal) pairs: (0.9>0.1), (0.9>0.6), (0.4>0.1)
      concordant, (0.4>0.6) not -> 3/4 = 0.75.
    * I-AP averages precision at each anomalous rank: 1/1 at rank 1 and 2/3 at rank 3
      -> (1 + 2/3)/2 = 5/6.
    * I-F1max is the best F1 over thresholds: at 0.40 -> TP=2, FP=1, FN=0, so P=2/3, R=1,
      F1=0.8, which beats 0.667 at 0.90, 0.5 at 0.60 and 0.667 at 0.10.

    The three values are distinct by construction, so swapping two of these computations in
    `image_metrics` cannot go unnoticed.
    """
    df, _, _ = _imperfect_shard(tmp_path)
    out = image_metrics(df)

    assert out["i_auroc"] == pytest.approx(0.75)
    assert out["i_ap"] == pytest.approx(5 / 6)
    assert out["i_f1max"] == pytest.approx(0.8)
    assert out["n"] == 4

    # Cross-check of WIRING only: this calls the same metric functions that
    # pixel_metrics/image_metrics call, so it catches a transposed or misrouted
    # column, not a wrong metric. The closed-form literals above carry correctness.
    labels = df["label"].to_numpy()
    scores = df["image_score"].to_numpy()
    assert out["i_auroc"] == pytest.approx(i_auroc(labels, scores))
    assert out["i_ap"] == pytest.approx(i_ap(labels, scores))
    assert out["i_f1max"] == pytest.approx(i_f1max(labels, scores))
    assert len({round(out[k], 6) for k in ("i_auroc", "i_ap", "i_f1max")}) == 3


def test_pixel_metrics_are_not_interchangeable_columns(tmp_path):
    """Each pixel-level column must carry its own metric, at its own FPR limit.

    Hand derivation on `_imperfect_shard` (4 anomalous pixels @0.9 and 1 @0.5; 59 normal:
    4 @0.7, 4 @0.45, 51 @0.1):

    * P-AUROC over the 5x59 = 295 (anomalous, normal) pairs: each 0.9 pixel beats all 59
      normals (236 pairs), the 0.5 pixel beats the 51 @0.1 and the 4 @0.45 but loses to the
      4 @0.7 (55 pairs), no ties -> 291/295 = 0.98644.
    * SegF1max over the achievable thresholds: 0.9 -> P=1, R=4/5, F1=8/9; 0.7 -> 0.615;
      0.5 -> 10/14; 0.45 -> 10/18; 0.1 -> 10/69. Max = 8/9 = 0.88889.
    * AU-PRO@0.3: PRO is the mean over the two regions regardless of their 4:1 size
      difference. Thresholds land at FPR 0 (above every normal pixel; PRO 0.5, only the big
      region found), 4/59 (t=0.7; PRO still 0.5) and 8/59 (t=0.45; PRO 1.0, the single-pixel
      region is finally caught), then flat to 0.3. Trapezoid area = 0.5*(4/59) +
      0.75*(4/59) + 1*(0.3 - 8/59) = 0.3 - 3/59, normalised by 0.3 -> 0.83051.
    * AU-PRO@0.05: 8/59 = 0.136 is outside the window, so the weak region is never caught
      inside it; the curve is flat at PRO 0.5 across [0, 0.05] -> 0.5.

    All four values are distinct, so transposing two columns -- or swapping the 0.3 and 0.05
    limits, or passing (amaps, masks) to p_auroc instead of (masks, amaps) -- fails here.
    """
    df, masks, amaps = _imperfect_shard(tmp_path)
    out = pixel_metrics(df)

    assert out["p_auroc"] == pytest.approx(291 / 295)
    assert out["seg_f1max"] == pytest.approx(8 / 9)
    assert out["au_pro_030"] == pytest.approx((0.3 - 3 / 59) / 0.3)
    assert out["au_pro_005"] == pytest.approx(0.5)
    assert out["n"] == 4

    # Cross-check of WIRING only (same metric functions the module calls): same arrays, metric
    # functions called directly, argument order and fpr_limit spelled out here.
    assert out["p_auroc"] == pytest.approx(p_auroc(masks, amaps))
    assert out["seg_f1max"] == pytest.approx(seg_f1max(masks, amaps))
    assert out["au_pro_030"] == pytest.approx(au_pro(masks, amaps, fpr_limit=0.3))
    assert out["au_pro_005"] == pytest.approx(au_pro(masks, amaps, fpr_limit=0.05))

    values = [out[k] for k in ("p_auroc", "seg_f1max", "au_pro_030", "au_pro_005")]
    assert len({round(v, 6) for v in values}) == 4


def test_aggregate_carries_the_imperfect_values_into_the_table_row(tmp_path):
    """The columns `aggregate` emits are the paper's table columns; check them by value."""
    df, _, _ = _imperfect_shard(tmp_path)
    out = aggregate(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["i_auroc"] == pytest.approx(0.75)
    assert row["i_ap"] == pytest.approx(5 / 6)
    assert row["i_f1max"] == pytest.approx(0.8)
    assert row["p_auroc"] == pytest.approx(291 / 295)
    assert row["seg_f1max"] == pytest.approx(8 / 9)
    assert row["au_pro_030"] == pytest.approx((0.3 - 3 / 59) / 0.3)
    assert row["au_pro_005"] == pytest.approx(0.5)
    assert row["n"] == 4
    assert row["n_pixel_rows"] == 4


def test_image_metrics_on_perfect_separation(tmp_path):
    out = image_metrics(_shard(tmp_path))
    assert out["i_auroc"] == 1.0
    assert out["i_ap"] == 1.0
    assert out["i_f1max"] == pytest.approx(1.0)
    assert out["n"] == 4


def test_image_metrics_refuses_unlabelled_rows(tmp_path):
    """Withheld labels must not be silently averaged in as normal."""
    df = _shard(tmp_path)
    df.loc[0, "label"] = LABEL_UNKNOWN
    with pytest.raises(ValueError, match="unlabelled"):
        image_metrics(df)


def test_image_metrics_refuses_a_single_class(tmp_path):
    df = _shard(tmp_path)
    df["label"] = 0
    with pytest.raises(ValueError):
        image_metrics(df)


def test_pixel_metrics_on_perfect_maps(tmp_path):
    out = pixel_metrics(_shard(tmp_path))
    assert out["p_auroc"] == pytest.approx(1.0)
    assert out["seg_f1max"] == pytest.approx(1.0)
    assert out["au_pro_030"] == pytest.approx(1.0)
    assert out["au_pro_005"] == pytest.approx(1.0)
    assert out["n"] == 4


def test_pixel_metrics_treats_a_missing_mask_as_all_normal(tmp_path):
    """A `good` image has no mask file; that means no anomalous pixels, not missing data."""
    df = _shard(tmp_path)
    assert df["mask_path"].isna().any()
    out = pixel_metrics(df)
    assert out["n"] == 4
    # If a missing mask were instead treated as all-anomalous, these `good` rows' pixels
    # (which anomaly maps correctly score as clean) would look like missed detections,
    # dragging p_auroc/seg_f1max well below 1.0 instead of leaving the perfect score intact.
    assert out["p_auroc"] == pytest.approx(1.0)
    assert out["seg_f1max"] == pytest.approx(1.0)


def test_pixel_metrics_skips_rows_without_a_map(tmp_path):
    df = _shard(tmp_path)
    df.loc[0, "map_path"] = None
    out = pixel_metrics(df)
    assert out["n"] == 3


def test_pixel_metrics_returns_nothing_when_no_maps_were_saved(tmp_path):
    df = _shard(tmp_path, with_pixels=False)
    assert pixel_metrics(df) == {"n": 0}


def test_pixel_metrics_refuses_to_exceed_the_pixel_budget(tmp_path):
    """Colab has ~12.7 GB of RAM. Failing with a number beats being OOM-killed silently."""
    df = _shard(tmp_path)
    with pytest.raises(ValueError, match="max_bytes"):
        pixel_metrics(df, max_bytes=10)


def test_pixel_metrics_guard_counts_bytes_actually_held_not_just_the_map(tmp_path):
    """The loop holds both the mask (uint8) and the amap (float32, upcast on load) at once.

    2 rows of a 6x8 map = 96 amap elements, which the oldest guard compared straight against
    `max_pixels` -- 96 <= 100 would pass. Even counting only the arrays that stay resident
    (mask + map) the footprint is 96 * (4 + 1) = 480 bytes, which must trip a budget of 100;
    the accounting that is actually in force counts the transient peak on top of that
    (see BYTES_PER_PIXEL).
    """
    df = _shard(tmp_path, n=2)
    with pytest.raises(ValueError, match="max_bytes"):
        pixel_metrics(df, max_bytes=100)


def test_pixel_metrics_guard_counts_the_transient_peak_not_only_resident_arrays(tmp_path):
    """The resident mask+map (5 B/px) is a small fraction of what the call really costs.

    `p_auroc` alone allocates ~56 B/px more on top of them (pooled copies in `_flatten`,
    sklearn's int64 argsort, float64 cumulative sums), so an accounting that stops at the
    resident arrays lets a workload through that needs an order of magnitude more RAM than
    the number it printed. 2 rows of a 6x8 map = 96 pixels: the resident-only accounting
    computes 96 * 5 = 480 bytes and so passes a budget of exactly 480, while the real peak
    is 96 * BYTES_PER_PIXEL.
    """
    df = _shard(tmp_path, n=2)
    with pytest.raises(ValueError, match="max_bytes") as exc_info:
        pixel_metrics(df, max_bytes=96 * 5)
    assert f"{96 * BYTES_PER_PIXEL:,}" in str(exc_info.value)


def test_pixel_metrics_guard_is_exact_at_the_budget_boundary(tmp_path):
    """Pins the multiplier to its measured value, in literals, not in terms of itself.

    80 B/px is the conservative over-estimate `test_bytes_per_pixel_covers_the_measured_peak`
    holds to a tolerance band around the real measurement; written out here so that lowering
    the constant (dropping the transient term, or the mask term, or reverting to the 5 B/px
    resident-only accounting) is an immediate failure rather than a silent regression that
    only the slow measurement would catch. This literal moves with the constant on purpose.
    """
    assert BYTES_PER_PIXEL == 80
    df = _shard(tmp_path, n=2)
    pixels = 2 * 6 * 8
    with pytest.raises(ValueError, match="max_bytes"):
        pixel_metrics(df, max_bytes=pixels * BYTES_PER_PIXEL - 1)
    out = pixel_metrics(df, max_bytes=pixels * BYTES_PER_PIXEL)
    assert out["n"] == 2


def test_default_max_bytes_refuses_a_whole_vial_test_public_category(tmp_path, monkeypatch):
    """The default has to stop the exact case this benchmark will actually run.

    Vial's `test_public` is 140 images of 1400x1900 = 372 Mpx, which needs ~29.8 GB at the
    guard's multiplier -- well over twice Colab's entire 12.7 GB free tier. The old 5 B/px
    accounting computed 1.86 GB for it and waved it straight through the 2 GB default.

    Only the map headers are read (`mmap_mode="r"`), so one real .npy of the right shape,
    referenced 140 times, exercises the guard without writing 745 MB of fixtures. `_load_pair`
    is stubbed to fail loudly: if the guard ever stops firing, this test must report that
    instead of trying to actually materialise 372 Mpx of maps in CI.
    """
    map_path = tmp_path / "vial_sized.npy"
    np.save(map_path, np.zeros((1400, 1900), dtype=np.float16))
    df = pd.DataFrame(
        [{"label": i % 2, "image_score": 0.5, "mask_path": None, "map_path": str(map_path)}
         for i in range(140)]
    )

    def _must_not_load(row):
        raise AssertionError("the guard let a 29.8 GB workload through and started loading")

    monkeypatch.setattr("vlmab.eval.aggregate._load_pair", _must_not_load)
    with pytest.raises(ValueError, match="max_bytes") as exc_info:
        pixel_metrics(df)  # default budget
    assert "29.8" in str(exc_info.value)


#: Measures the peak RSS of one `pixel_metrics` call, in a *fresh* interpreter.
#
# It has to be a subprocess. Peak RSS is measured as VmHWM above the current VmRSS, and
# inside a long-lived pytest process the heap is already grown from earlier tests: freed
# arenas get reused, so RSS never rises and the measurement collapses towards zero (measured:
# 2.8 B/px for a call that genuinely costs ~62 B/px when run first). A new interpreter has no
# such history.
_MEASURE_PEAK = r"""
import gc, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from vlmab.eval.aggregate import pixel_metrics

def status(key):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(key):
            return int(line.split()[1]) * 1024
    raise KeyError(key)

tmp = Path(sys.argv[1])
n_images, side = 4, 707  # 4 * 707^2 = 1,999,396 px
rng = np.random.default_rng(0)
rows = []
for i in range(n_images):
    amap = rng.random((side, side)).astype(np.float16)
    map_path = tmp / f"{i}.npy"
    np.save(map_path, amap)
    row = {"label": i % 2, "image_score": 0.5, "mask_path": None, "map_path": str(map_path)}
    if i % 2:
        mask = np.zeros((side, side), dtype=np.uint8)
        mask[: side // 4, : side // 4] = 255
        Image.fromarray(mask, mode="L").save(tmp / f"{i}_mask.png")
        row["mask_path"] = str(tmp / f"{i}_mask.png")
    rows.append(row)
df = pd.DataFrame(rows)
pixels = n_images * side * side
del rng, amap

gc.collect()
Path("/proc/self/clear_refs").write_text("5")  # resets VmHWM to the current VmRSS
baseline = status("VmRSS:")
pixel_metrics(df, max_bytes=10 ** 12)
print((status("VmHWM:") - baseline) / pixels)
"""


@pytest.mark.skipif(
    not Path("/proc/self/clear_refs").exists(), reason="peak-RSS measurement needs Linux /proc"
)
def test_bytes_per_pixel_covers_the_measured_peak(tmp_path):
    """Measure the real peak RSS of a `pixel_metrics` call instead of trusting a comment.

    2 Mpx is enough to swamp the fixed interpreter costs (~125 MB of measured peak) while
    staying small enough to run in CI.

    The measured peak is NOT a fixed number: the transient working set of sklearn's
    argsort/cumsum depends on the Python and library build, so it comes out around 60 B/px on
    the dev machine (3.13) and around 66 on the CI runner (3.11). An earlier version of this
    test asserted `BYTES_PER_PIXEL >= measured` with no margin, and broke the moment CI
    measured 65.8 against a constant of 64 -- it was pinning a compile-time constant to one
    machine's exact sample. So this now checks a band: the constant is a genuine over-estimate
    of the peak (a guard must round up, never down) that sits within measurement noise of it,
    wide enough to absorb the cross-build spread but far too tight to admit the bug it exists
    to catch -- the old 5 B/px accounting, which left `measured` ~13x the constant.
    """
    out = subprocess.run(
        [sys.executable, "-c", _MEASURE_PEAK, str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr
    measured_per_pixel = float(out.stdout.strip().splitlines()[-1])

    assert measured_per_pixel > 20, measured_per_pixel  # the 5 B/px accounting was absurd
    # Constant covers the peak, allowing the measurement up to 10% above it for build-to-build
    # noise: a 10% slip is immaterial against the default guard's ~2x headroom, whereas the
    # 5 B/px bug would put `measured` far outside this bound.
    assert measured_per_pixel <= 1.10 * BYTES_PER_PIXEL, measured_per_pixel
    # ...and the constant is not absurdly conservative, or the guard would refuse valid work.
    assert BYTES_PER_PIXEL <= 2 * measured_per_pixel, measured_per_pixel



def test_aggregate_returns_one_row_with_both_metric_families(tmp_path):
    out = aggregate(_shard(tmp_path))
    assert len(out) == 1
    assert {"i_auroc", "p_auroc", "au_pro_030", "n"} <= set(out.columns)


def test_aggregate_groups_by_lighting_condition(tmp_path):
    """The lighting breakdown is the reason this dataset exists."""
    out = aggregate(_shard(tmp_path), by="meta_lighting")
    assert list(out["meta_lighting"]) == ["overexposed", "regular"]
    assert len(out) == 2


def test_aggregate_names_the_group_whose_maps_are_degenerate(tmp_path):
    """A constant anomaly map now fails loudly instead of reporting au_pro = 0.0.

    0.0 is a plausible AU-PRO value, so a shard of broken (constant) maps used to produce a
    complete-looking metrics row that could go straight into a table. `au_pro` refuses such
    input, and `aggregate` treats it like any other per-group failure: the group is named,
    the underlying reason survives, and no row is emitted for the whole frame.
    """
    df = _shard(tmp_path)
    for path in df["map_path"]:
        np.save(path, np.full((6, 8), 0.5, dtype=np.float16))
    with pytest.raises(ValueError) as exc_info:
        aggregate(df, by="meta_lighting")
    message = str(exc_info.value)
    assert "meta_lighting" in message
    assert "constant anomaly map" in message
    assert exc_info.value.__cause__ is not None


def test_aggregate_rejects_a_missing_group_column(tmp_path):
    with pytest.raises(KeyError):
        aggregate(_shard(tmp_path), by="meta_nope")


def test_aggregate_names_the_failing_group_on_a_single_class_group(tmp_path):
    """Seven lighting conditions in the real study; a bare error can't be traced back to one."""
    df = _shard(tmp_path)
    df.loc[df["meta_lighting"] == "regular", "label"] = 0
    with pytest.raises(ValueError) as exc_info:
        aggregate(df, by="meta_lighting")
    message = str(exc_info.value)
    assert "meta_lighting" in message and "regular" in message
    assert "only label 0" in message  # the underlying reason must survive, not just the label
    assert exc_info.value.__cause__ is not None  # original exception chained, not swallowed


def _threshold_shard(
    tmp_path, n=4, size=16, category="vial", dataset="mvtec_ad2", method="intensity_baseline"
):
    """A shard with real map files on disk and one anomalous square per bad image.

    Stamps the identity columns (`dataset`, `method`, `category` -- see IDENTITY_COLUMNS in
    vlmab.eval.store) a real ResultStore shard always carries, since threshold_metrics checks
    them against the artifact it is handed.
    """
    import pandas as pd
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        amap = rng.random((size, size)).astype(np.float32)
        bad = i % 2 == 1
        mask_path = None
        if bad:
            amap[2:6, 2:6] += 5.0
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            mask_path = tmp_path / f"{i}_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
        map_path = tmp_path / f"{i}.npy"
        np.save(map_path, amap.astype(np.float16))
        rows.append({
            "image_path": f"/fake/{i}.png",
            "label": int(bad),
            "image_score": float(amap.max()),
            "split": "test_public",
            "mask_path": None if mask_path is None else str(mask_path),
            "map_path": str(map_path),
            "meta_lighting": "regular",
            "dataset": dataset,
            "method": method,
            "category": category,
        })
    return pd.DataFrame(rows)


def _threshold_artifact(tmp_path, category="vial"):
    path = write_artifact(
        tmp_path / "cal.yaml",
        dataset="mvtec_ad2",
        method="intensity_baseline",
        alpha=1e-3,
        calibrated_on={"split": "validation", "n_images": 4, "lighting": ["regular"]},
        run_id="deadbeef",
        categories={category: {
            "global_quantile": Calibration("global_quantile", 1.0, 1e-3, 100, 0),
            "per_image_robust_z": Calibration("per_image_robust_z", 3.0, 1e-3, 100, 0),
            "transductive_quantile": Calibration("transductive_quantile", float("nan"), 1e-3, 0, 0),
        }},
        designated_for_submission="per_image_robust_z",
    )
    return load_artifact(path)


def test_threshold_metrics_reports_every_rule(tmp_path):
    df = _threshold_shard(tmp_path)
    out = threshold_metrics(df, _threshold_artifact(tmp_path), "vial")
    for rule in ("global_quantile", "per_image_robust_z", "transductive_quantile"):
        assert 0.0 <= out[f"seg_f1_at__{rule}"] <= 1.0
        assert 0.0 <= out[f"fpr_at__{rule}"] <= 1.0
        assert out[f"degenerate_mad__{rule}"] == 0
    assert out["n"] == 4


def test_threshold_metrics_on_this_fixture_do_not_beat_the_oracle(tmp_path):
    # NOT a universal invariant: seg_f1max upper-bounds only rules that pick one *global*
    # threshold (global_quantile, transductive_quantile) -- it searches that same space
    # exhaustively. per_image_robust_z picks one threshold per image, a different,
    # non-nested, strictly larger search space, so it is not bounded by the oracle in
    # general and can in principle score above it on heterogeneous data. Here it happens
    # to stay under the oracle too, which this test records as an observation on this
    # fixture, not as a law; a future failure of this assertion on real data would be a
    # finding about the rule's behaviour, not necessarily a bug.
    df = _threshold_shard(tmp_path)
    out = threshold_metrics(df, _threshold_artifact(tmp_path), "vial")
    oracle = pixel_metrics(df)["seg_f1max"]
    for rule in ("global_quantile", "per_image_robust_z", "transductive_quantile"):
        assert out[f"seg_f1_at__{rule}"] <= oracle + 1e-12


def test_threshold_metrics_refuses_unlabelled_rows(tmp_path):
    df = _threshold_shard(tmp_path)
    df.loc[0, "label"] = -1
    with pytest.raises(ValueError, match="unlabelled"):
        threshold_metrics(df, _threshold_artifact(tmp_path), "vial")


def test_threshold_metrics_raises_on_a_category_the_artifact_has_not_calibrated(tmp_path):
    # df's own category matches what's requested (so the join guard is satisfied); the
    # artifact itself just never calibrated that category.
    df = _threshold_shard(tmp_path, category="fabric")
    with pytest.raises(KeyError, match="fabric"):
        threshold_metrics(df, _threshold_artifact(tmp_path, category="vial"), "fabric")


def test_threshold_metrics_needs_maps(tmp_path):
    df = _threshold_shard(tmp_path).drop(columns=["map_path"])
    with pytest.raises(ValueError, match="map_path"):
        threshold_metrics(df, _threshold_artifact(tmp_path), "vial")


def test_threshold_metrics_refuses_a_df_pooling_other_categories(tmp_path):
    # A multi-category results root, loaded whole and handed in with only `category="vial"`
    # requested: without this guard, the other category's maps would be pooled in and cut at
    # vial's threshold.
    vial = _threshold_shard(tmp_path, category="vial")
    fabric = _threshold_shard(tmp_path, category="fabric")
    df = pd.concat([vial, fabric], ignore_index=True)
    with pytest.raises(ValueError, match="category"):
        threshold_metrics(df, _threshold_artifact(tmp_path, category="vial"), "vial")


def test_threshold_metrics_refuses_a_method_mismatch(tmp_path):
    # The exact notebook mistake the spec forbids: a df for method B scored against an
    # artifact calibrated for method A. Maps are on each method's own scale (protocol §4).
    df = _threshold_shard(tmp_path, method="winclip")
    artifact = _threshold_artifact(tmp_path)  # method="intensity_baseline"
    with pytest.raises(ValueError, match="winclip"):
        threshold_metrics(df, artifact, "vial")


def test_threshold_metrics_refuses_a_dataset_mismatch(tmp_path):
    df = _threshold_shard(tmp_path, dataset="visa")
    artifact = _threshold_artifact(tmp_path)  # dataset="mvtec_ad2"
    with pytest.raises(ValueError, match="visa"):
        threshold_metrics(df, artifact, "vial")
