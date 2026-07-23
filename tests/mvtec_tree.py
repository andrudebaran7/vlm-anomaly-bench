"""Build a synthetic MVTec AD 2 category tree.

Mirrors the layout measured from the real `vial.tar.gz`: five splits, `good`/`bad`/`ground_truth`
under `test_public`, flat unlabelled private splits, and the `{index:03d}_{condition}` filename
convention with `_mask` appended for ground truth. Keeping this in one place means a layout change
is a one-file edit rather than a hunt through test modules.
"""
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

CONDITIONS = ("regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3", "shift_4")


def _write(path: Path, size: tuple[int, int], mode: str, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    if mode == "L":
        arr = np.full((h, w), value, dtype=np.uint8)
    else:
        # Distinct per-channel values (not np.full's uniform fill) so a loader bug that
        # collapses, averages, or duplicates a single channel across all three is caught by
        # a test reading back R, G, and B independently.
        arr = np.empty((h, w, 3), dtype=np.uint8)
        arr[..., 0] = value
        arr[..., 1] = value + 1
        arr[..., 2] = value + 2
    Image.fromarray(arr, mode=mode).save(path)


def build_category(
    root: Path,
    category: str,
    conditions: Sequence[str] = CONDITIONS,
    n_good: int = 2,
    n_bad: int = 2,
    size: tuple[int, int] = (8, 6),
    mode: str = "L",
) -> Path:
    """Create <root>/<category>/ with every split populated. Returns the category directory.

    Pixel values encode provenance so a test can tell which file it loaded: good images are
    filled with 10, bad with 200, private with 50, mixed with 60, train with 20, validation
    with 30. In `mode="RGB"`, the three channels get distinct values `value`, `value + 1`,
    `value + 2` (R, G, B respectively) instead of a uniform fill, so a test can verify each
    channel is preserved independently rather than merely that some value survived. `mode="L"`
    is unaffected: it keeps the single uniform `value`. Masks are 255 in a 2x2 corner block and
    0 elsewhere.
    """
    cat = root / category

    for i in range(n_good):
        _write(cat / "train" / "good" / f"{i:03d}_regular.png", size, mode, 20)
    for i in range(n_good):
        _write(cat / "validation" / "good" / f"{i:03d}_regular.png", size, mode, 30)

    for cond in conditions:
        for i in range(n_good):
            _write(cat / "test_public" / "good" / f"{i:03d}_{cond}.png", size, mode, 10)
        for i in range(n_bad):
            _write(cat / "test_public" / "bad" / f"{i:03d}_{cond}.png", size, mode, 200)
            w, h = size
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:2, :2] = 255
            gt = cat / "test_public" / "ground_truth" / "bad" / f"{i:03d}_{cond}_mask.png"
            gt.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(gt)

    for i in range(n_bad):
        _write(cat / "test_private" / f"{i:03d}_regular.png", size, mode, 50)
        _write(cat / "test_private_mixed" / f"{i:03d}_mixed.png", size, mode, 60)

    return cat
