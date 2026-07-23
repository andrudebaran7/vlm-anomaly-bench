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
    with pytest.raises(ValueError, match="max_pixels"):
        pixel_metrics(df, max_pixels=10)


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
