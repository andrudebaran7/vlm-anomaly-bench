"""scripts/figure_inputs.py applies the figure selections pre-registered on 2026-10-29."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmab.eval.store import ResultStore


def _frame(scores, labels=None, lighting="regular"):
    labels = labels or [1] * len(scores)
    return pd.DataFrame({"image_path": [f"/d/{i:03d}_{lighting}.png" for i in range(len(scores))],
                         "image_score": scores, "label": labels, "meta_lighting": lighting})


def test_the_median_pick_is_the_lower_median_of_the_anomalous_images_of_the_condition():
    from figure_inputs import median_anomalous

    df = pd.concat([_frame([0.9, 0.1, 0.5, 0.3], labels=[1, 1, 1, 1]),
                    _frame([0.99], labels=[0]),                      # normal: never picked
                    _frame([0.2, 0.4], lighting="shift_1")])         # other condition
    assert median_anomalous(df)["image_score"] == 0.3      # sorted 0.1 0.3 0.5 0.9 -> index 1


def test_legitimacy_needs_every_score_equal_at_full_precision():
    from figure_inputs import legitimacy

    a = _frame([0.1, 0.2, 0.3])
    assert legitimacy(a, a.copy())["ok"] is True
    b = a.copy()
    b.loc[1, "image_score"] = 0.2 + 1e-12
    out = legitimacy(a, b)
    assert out["ok"] is False and out["n_different"] == 1
    assert legitimacy(a, a.iloc[:2])["ok"] is False


def test_first_by_filename_respects_the_condition(tmp_path):
    from figure_inputs import first_by_filename

    for name in ("002_regular.png", "000_shift_1.png", "001_regular.png"):
        (tmp_path / name).write_bytes(b"")
    assert first_by_filename(tmp_path, "regular").name == "001_regular.png"


def test_model_refuses_when_the_regenerated_scores_differ(tmp_path, capsys):
    from figure_inputs import main

    rows = [{"image_path": "/d/0.png", "label": 1, "image_score": 0.5, "split": "test_public",
             "mask_path": None, "map_path": None, "meta_lighting": "regular"}]
    ResultStore(tmp_path / "new" / "shards").write("mvtec_ad2", "patchcore_ref", "can", rows, {"seed": 0})
    rows[0]["image_score"] = 0.6
    ResultStore(tmp_path / "old").write("mvtec_ad2", "patchcore_ref", "can", rows, {"seed": 0})
    rc = main(["model", "--category", "can", "--results", str(tmp_path / "new"),
               "--committed", str(tmp_path / "old"), "--out", str(tmp_path / "out")])
    assert rc == 1 and "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_model_copies_the_pick_and_records_why(tmp_path):
    from figure_inputs import main

    rows = []
    for i, score in enumerate([0.2, 0.8, 0.5]):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        amap = np.full((8, 8), 0.1, dtype=np.float32)
        amap[2:4, 2:4] = 0.9
        img = tmp_path / f"{i:03d}_regular.png"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img)
        Image.fromarray(mask).save(tmp_path / f"{i}_mask.png")
        np.save(tmp_path / f"{i}.npy", amap.astype(np.float16))
        rows.append({"image_path": str(img), "label": 1, "image_score": score,
                     "split": "test_public", "mask_path": str(tmp_path / f"{i}_mask.png"),
                     "map_path": str(tmp_path / f"{i}.npy"), "meta_lighting": "regular"})
    ResultStore(tmp_path / "new" / "shards").write("mvtec_ad2", "patchcore_ref", "can", rows, {"seed": 0})
    ResultStore(tmp_path / "old").write("mvtec_ad2", "patchcore_ref", "can", rows, {"seed": 0})
    out = tmp_path / "out"
    assert main(["model", "--category", "can", "--results", str(tmp_path / "new"),
                 "--committed", str(tmp_path / "old"), "--out", str(out)]) == 0
    sel = json.loads((out / "selection.json").read_text())
    assert sel["image_score"] == 0.5 and sel["rank"].startswith("2 of 3")
    assert sel["legitimacy"]["ok"] is True
    assert sel["inside_outside_ratio"] == pytest.approx(0.9 / 0.1, rel=1e-3)
    assert (out / "image.png").is_file() and (out / "mask.png").is_file() and (out / "map.npy").is_file()
