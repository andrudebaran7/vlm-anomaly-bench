#!/usr/bin/env python3
"""Apply the pre-registered figure selections on Colab and copy only what was selected to Drive.

The rules are in docs/next-steps.md, "PRE-REGISTERED 2026-10-29 ... the figure selections",
recorded before any map was regenerated. This script is those rules and nothing else: it never
shows a map, and it writes a `selection.json` beside the files saying which rule picked each one.

    # spec figure 4: after a fresh seed-0 run of <category> into <results>
    python scripts/figure_inputs.py model --category walnuts \\
        --results results/figure_check/walnuts \\
        --committed /content/drive/MyDrive/mvtec_ad2_results/patchcore_ref_v0.2.21/walnuts/shards \\
        --out /content/drive/MyDrive/figure_inputs/walnuts

    # spec figures 1 and 2: no model
    python scripts/figure_inputs.py dataset --category walnuts --out .../figure_inputs/walnuts_dataset
    python scripts/figure_inputs.py dataset --category sheet_metal --resize --out .../sheet_metal_resize

`model` refuses (exit 1) unless the legitimacy check passes: the regenerated seed-0 image scores
must equal the committed ones exactly, or the maps are not the ones behind the reported numbers.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from vlmab.datasets.mvtec_ad2 import lighting_condition
from vlmab.eval.store import ResultStore

CONDITION = "regular"
SEED = 0


def legitimacy(regenerated: pd.DataFrame, committed: pd.DataFrame) -> dict:
    """Image-by-image comparison of `image_score`, keyed by image path. Exact equality."""
    a = regenerated.set_index("image_path")["image_score"]
    b = committed.set_index("image_path")["image_score"]
    if set(a.index) != set(b.index):
        return {"ok": False, "reason": "the two runs scored different image sets",
                "only_regenerated": len(set(a.index) - set(b.index)),
                "only_committed": len(set(b.index) - set(a.index))}
    diff = (a - b.loc[a.index]).abs()
    return {"ok": bool((diff == 0).all()), "n_images": int(len(a)),
            "n_different": int((diff != 0).sum()), "max_abs_diff": float(diff.max())}


def median_anomalous(df: pd.DataFrame, condition: str = CONDITION) -> pd.Series:
    """The pre-registered pick: anomalous rows of `condition`, by image_score then path,
    index (n - 1) // 2."""
    rows = df[(df["label"] == 1) & (df["meta_lighting"] == condition)]
    if rows.empty:
        raise ValueError(f"no anomalous image in condition {condition!r}")
    rows = rows.sort_values(["image_score", "image_path"], kind="mergesort")
    return rows.iloc[(len(rows) - 1) // 2]


def inside_outside_ratio(mask: np.ndarray, amap: np.ndarray) -> float:
    """Mean map value inside the ground-truth region over the mean outside it."""
    inside = mask > 0
    return float(amap[inside].mean() / amap[~inside].mean())


def first_by_filename(directory: Path, condition: str) -> Path:
    for p in sorted(Path(directory).glob("*.png")):
        if lighting_condition(p) == condition:
            return p
    raise FileNotFoundError(f"no {condition!r} image in {directory}")


def _shard(root: Path, seed: int) -> pd.DataFrame:
    df = ResultStore(root).load_all()
    if df.empty:
        raise FileNotFoundError(f"no shards under {root}")
    return df[df["seed"] == seed]


def cmd_model(args) -> int:
    from vlmab.eval.aggregate import _load_pair

    regenerated = _shard(args.results / "shards", SEED)
    committed = _shard(args.committed, SEED)
    check = legitimacy(regenerated, committed)
    print("legitimacy:", json.dumps(check))
    if not check["ok"]:
        print("REFUSED: the regenerated scores differ from the committed ones, so these maps are "
              "not the ones behind the reported numbers (pre-registration, 2026-10-29).")
        return 1

    row = median_anomalous(regenerated)
    mask, amap = _load_pair(row)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(row["image_path"], args.out / "image.png")
    shutil.copy2(row["mask_path"], args.out / "mask.png")
    np.save(args.out / "map.npy", amap.astype(np.float16))
    n = int(((regenerated["label"] == 1) & (regenerated["meta_lighting"] == CONDITION)).sum())
    selection = {
        "rule": "median-scoring anomalous image, condition regular, seed 0 "
                "(docs/next-steps.md, PRE-REGISTERED 2026-10-29)",
        "category": args.category, "seed": SEED, "condition": CONDITION,
        "image_path": row["image_path"], "image_score": float(row["image_score"]),
        "rank": f"{(n - 1) // 2 + 1} of {n} anomalous images, by image_score ascending",
        "inside_outside_ratio": inside_outside_ratio(mask, amap),
        "legitimacy": check,
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))
    return 0


def cmd_dataset(args) -> int:
    test = args.root / args.category / "test_public"
    good, bad = test / "good", test / "bad"
    conditions = sorted({lighting_condition(p) for p in good.glob("*.png")})
    args.out.mkdir(parents=True, exist_ok=True)
    picked = {}
    targets = [CONDITION] if args.resize else conditions
    for c in targets:
        p = first_by_filename(good, c)
        shutil.copy2(p, args.out / f"good_{c}.png")
        picked[f"good_{c}"] = str(p)
    if not args.resize:
        p = first_by_filename(bad, CONDITION)
        mask = test / "ground_truth" / "bad" / f"{p.stem}_mask.png"
        shutil.copy2(p, args.out / f"bad_{CONDITION}.png")
        shutil.copy2(mask, args.out / f"bad_{CONDITION}_mask.png")
        picked[f"bad_{CONDITION}"] = str(p)
    else:
        picked["resized"] = _resize_with_the_anchors_transform(
            args.out / f"good_{CONDITION}.png", args.out / f"good_{CONDITION}_resized.png")
    selection = {"rule": "first image by filename (docs/next-steps.md, PRE-REGISTERED "
                         "2026-10-29)", "category": args.category, "picked": picked}
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))
    return 0


def _resize_with_the_anchors_transform(src: Path, dst: Path) -> str:  # pragma: no cover - GPU env
    """Apply the Resize step of the PatchCore pre-processor anomalib 2.6.0 builds, and nothing
    else (Normalize is left out: the figure shows geometry, not normalised intensities)."""
    import torch
    from anomalib.models import Patchcore
    from PIL import Image

    steps = list(getattr(Patchcore().pre_processor.transform, "transforms", []))
    resize = [t for t in steps if type(t).__name__ == "Resize"]
    if len(resize) != 1:
        raise RuntimeError(f"expected one Resize in the pre-processor, found {steps}")
    arr = np.asarray(Image.open(src).convert("RGB"))
    x = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    y = resize[0](x)[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(y).save(dst)
    return f"{resize[0]!r} applied to {src.name} {arr.shape[1]}x{arr.shape[0]} -> {y.shape[1]}x{y.shape[0]}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--category", required=True)
    m.add_argument("--results", required=True, type=Path)
    m.add_argument("--committed", required=True, type=Path)
    m.add_argument("--out", required=True, type=Path)
    d = sub.add_parser("dataset")
    d.add_argument("--category", required=True)
    d.add_argument("--root", type=Path, default=Path("data/mvtec_ad2"))
    d.add_argument("--resize", action="store_true")
    d.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    return cmd_model(args) if args.cmd == "model" else cmd_dataset(args)


if __name__ == "__main__":
    sys.exit(main())
