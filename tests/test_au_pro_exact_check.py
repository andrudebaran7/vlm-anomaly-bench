"""scripts/au_pro_exact_check.py: grid against exact AU-PRO on a real run's maps (review #24).

The maps of a real MVTec AD 2 run exist only on the Colab VM that produced them, so the check is
a script that runs there; this tests it on a small synthetic run laid out the way the runner
writes one.
"""
import numpy as np
from PIL import Image

from vlmab.eval.store import ResultStore


def _run(tmp_path, n=6, size=64):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        lighting = "regular" if i < n // 2 else "shift_1"
        amap = rng.random((size, size)).astype(np.float32)
        mask_path = None
        if i % 2:
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[10:20, 10:20] = 255
            amap[10:20, 10:20] += rng.uniform(0.2, 0.8)
            mask_path = tmp_path / f"{i}_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
        map_path = tmp_path / f"{i}.npy"
        np.save(map_path, amap.astype(np.float16))
        rows.append({"image_path": f"/fake/{i}.png", "label": int(i % 2), "image_score": 0.5,
                     "split": "test_public", "mask_path": None if mask_path is None else str(mask_path),
                     "map_path": str(map_path), "meta_lighting": lighting})
    ResultStore(tmp_path / "shards").write("mvtec_ad2", "patchcore_ref", "toy", rows, {"seed": 0})
    return tmp_path


def test_check_reports_grid_and_exact_per_condition_and_limit(tmp_path, capsys):
    from au_pro_exact_check import check, main

    rows = check(_run(tmp_path))
    assert [(r["lighting"], r["limit"]) for r in rows] == [
        ("regular", 0.3), ("regular", 0.05), ("shift_1", 0.3), ("shift_1", 0.05)]
    for r in rows:
        assert 0.0 <= r["grid"] <= 1.0 and 0.0 <= r["exact"] <= 1.0
        assert r["diff"] == r["grid"] - r["exact"]

    assert main(["--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "max |grid - exact|" in out and "shift_1" in out
