"""MVTec AD (classic) loader tests — reproduction dataset, PatchCore's gate (protocol §2 v0.2.12).

Built against the real `bottle.tar.xz` read 2026-09-16, not against the paper. That archive's
shape, verbatim: `train/good/`, `test/good/` plus one directory per defect type, and
`ground_truth/<defect>/<stem>_mask.png` with no `good` subdirectory, because normal test images
have no mask. bottle's counts are 209 train, 20 test good, 63 test anomalous, 63 masks.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vlmab.datasets.mvtec_ad import MVTecAD


def _cat(root, name="bottle", defects=("broken_large", "contamination"), n_train=3, n_def=2):
    """One category in the real layout, with real files."""
    def _png(path, value, mode="RGB"):
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = (np.full((4, 6, 3), value, dtype=np.uint8) if mode == "RGB"
               else np.full((4, 6), value, dtype=np.uint8))
        Image.fromarray(arr, mode=mode).save(path)

    for i in range(n_train):
        _png(root / name / "train" / "good" / f"{i:03d}.png", 10)
    for i in range(2):
        _png(root / name / "test" / "good" / f"{i:03d}.png", 20)
    for d in defects:
        for i in range(n_def):
            _png(root / name / "test" / d / f"{i:03d}.png", 200)
            mask = np.zeros((4, 6), dtype=np.uint8)
            mask[1:3, 2:4] = 255
            path = root / name / "ground_truth" / d / f"{i:03d}_mask.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(path)
    (root / name / "license.txt").write_text("cc-by-nc-sa-4.0")
    (root / name / "readme.txt").write_text("bottle")
    return root


def test_categories_are_the_directories(tmp_path):
    _cat(_cat(tmp_path), name="cable", defects=("bent_wire",))
    assert MVTecAD(tmp_path).categories() == ["bottle", "cable"]


def test_train_is_normal_only_and_maskless(tmp_path):
    samples = list(MVTecAD(_cat(tmp_path)).samples("train"))
    assert len(samples) == 3
    assert {s.label for s in samples} == {0}
    assert all(s.mask_path is None for s in samples)


def test_test_good_is_normal_and_every_defect_directory_is_anomalous(tmp_path):
    samples = list(MVTecAD(_cat(tmp_path)).samples("test"))
    assert sorted(s.label for s in samples) == [0, 0, 1, 1, 1, 1]


def test_each_anomalous_sample_finds_its_mask_by_the_mask_suffix(tmp_path):
    """`test/<defect>/000.png` pairs with `ground_truth/<defect>/000_mask.png`. The stem alone
    is not unique — every defect directory restarts at 000 — so the defect must be in the path."""
    samples = [s for s in MVTecAD(_cat(tmp_path)).samples("test") if s.label == 1]
    assert len(samples) == 4
    for s in samples:
        assert s.mask_path is not None and s.mask_path.is_file()
        assert s.mask_path.parent.name == s.meta["defect_type"]
        assert s.mask_path.name == s.image_path.stem + "_mask.png"


def test_normal_test_samples_have_no_mask(tmp_path):
    """`ground_truth/` has no `good` directory; inventing one would be a silent all-normal mask."""
    samples = [s for s in MVTecAD(_cat(tmp_path)).samples("test") if s.label == 0]
    assert samples and all(s.mask_path is None for s in samples)


def test_defect_type_is_recorded_in_meta(tmp_path):
    samples = [s for s in MVTecAD(_cat(tmp_path)).samples("test") if s.label == 1]
    assert {s.meta["defect_type"] for s in samples} == {"broken_large", "contamination"}
    normal = next(s for s in MVTecAD(_cat(tmp_path)).samples("test") if s.label == 0)
    assert normal.meta["defect_type"] == "good"


def test_a_missing_mask_raises_rather_than_scoring_as_normal(tmp_path):
    """An anomalous image whose mask is absent would otherwise be pooled as all-normal pixels,
    deflating every pixel metric with nothing to show for it."""
    root = _cat(tmp_path)
    (root / "bottle" / "ground_truth" / "broken_large" / "000_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="000_mask.png"):
        list(MVTecAD(root).samples("test"))


def test_unknown_split_raises(tmp_path):
    """MVTec AD classic has no validation split; asking for one must not yield nothing."""
    with pytest.raises(ValueError, match="validation"):
        list(MVTecAD(_cat(tmp_path)).samples("validation"))


def test_unknown_category_raises(tmp_path):
    with pytest.raises(ValueError, match="screw"):
        list(MVTecAD(_cat(tmp_path)).samples("test", category="screw"))


def test_load_image_and_mask_shapes_agree(tmp_path):
    ds = MVTecAD(_cat(tmp_path))
    s = next(x for x in ds.samples("test") if x.label == 1)
    image, mask = ds.load_image(s), ds.load_mask(s)
    assert image.shape == (4, 6, 3) and image.dtype == np.uint8
    assert mask.shape == (4, 6) and set(np.unique(mask)) <= {0, 1} and mask.sum() == 4


# --- Opt-in: the real archive ------------------------------------------------------------
# Skipped in CI, which has no dataset. bottle's counts below are MVTec AD's published figures,
# so a pass is agreement between the archive, the dataset's own numbers, and this loader.

_REAL = Path(__file__).resolve().parents[1] / "data" / "mvtec_ad"


@pytest.mark.skipif(
    not (_REAL / "bottle" / "train" / "good").is_dir(),
    reason=f"MVTec AD classic 'bottle' not present at {_REAL}",
)
def test_real_bottle_reproduces_the_published_counts():
    ds = MVTecAD(_REAL)
    train = list(ds.samples("train", category="bottle"))
    test = list(ds.samples("test", category="bottle"))
    assert len(train) == 209 and {s.label for s in train} == {0}
    assert len(test) == 83
    assert sum(1 for s in test if s.label == 0) == 20
    assert sum(1 for s in test if s.label == 1) == 63
    assert Counter(s.meta["defect_type"] for s in test) == {
        "good": 20, "broken_large": 20, "broken_small": 22, "contamination": 21,
    }
    assert all(s.mask_path.is_file() for s in test if s.label == 1)
    anomalous = next(s for s in test if s.label == 1)
    image, mask = ds.load_image(anomalous), ds.load_mask(anomalous)
    assert image.shape == (900, 900, 3) and mask.shape == (900, 900)
    assert set(np.unique(mask)) <= {0, 1} and mask.sum() > 0
