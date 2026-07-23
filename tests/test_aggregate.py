import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN
from vlmab.eval.aggregate import aggregate, image_metrics, pixel_metrics


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

    2 rows of a 6x8 map = 96 amap elements, which the old guard compared straight against
    `max_pixels` -- 96 <= 100 would pass. The real resident footprint once both arrays are
    held is 96 * (4 + 1) = 480 bytes, which must trip a budget of 100.
    """
    df = _shard(tmp_path, n=2)
    with pytest.raises(ValueError, match="max_bytes"):
        pixel_metrics(df, max_bytes=100)


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
