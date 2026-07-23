import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN
from vlmab.eval.aggregate import BYTES_PER_PIXEL, aggregate, image_metrics, pixel_metrics


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

    64 B/px is what `test_bytes_per_pixel_covers_the_measured_peak` measures on real data;
    written out here so that lowering the constant (dropping the transient term, or the mask
    term, or reverting to the 5 B/px resident-only accounting) is an immediate failure rather
    than a silent regression that only the slow measurement would catch.
    """
    assert BYTES_PER_PIXEL == 64
    df = _shard(tmp_path, n=2)
    pixels = 2 * 6 * 8
    with pytest.raises(ValueError, match="max_bytes"):
        pixel_metrics(df, max_bytes=pixels * 64 - 1)
    out = pixel_metrics(df, max_bytes=pixels * 64)
    assert out["n"] == 2


def test_default_max_bytes_refuses_a_whole_vial_test_public_category(tmp_path, monkeypatch):
    """The default has to stop the exact case this benchmark will actually run.

    Vial's `test_public` is 140 images of 1400x1900 = 372 Mpx, which needs ~23.8 GB at the
    measured multiplier -- roughly twice Colab's entire 12.7 GB free tier. The old 5 B/px
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
        raise AssertionError("the guard let a 23.8 GB workload through and started loading")

    monkeypatch.setattr("vlmab.eval.aggregate._load_pair", _must_not_load)
    with pytest.raises(ValueError, match="max_bytes") as exc_info:
        pixel_metrics(df)  # default budget
    assert "23.8" in str(exc_info.value)


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
    staying small enough to run in CI. Asserted both ways: the constant must not undercount
    the measurement (the bug this pins -- 5 B/px was a ~12x undercount), and must not be
    wildly above it either, or the guard would refuse workloads that actually fit.
    """
    out = subprocess.run(
        [sys.executable, "-c", _MEASURE_PEAK, str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr
    measured_per_pixel = float(out.stdout.strip().splitlines()[-1])

    assert measured_per_pixel > 20, measured_per_pixel  # the 5 B/px accounting was absurd
    assert BYTES_PER_PIXEL >= measured_per_pixel, measured_per_pixel
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
