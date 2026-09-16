"""VisA loader tests.

Built against the real `split_csv/1cls.csv` read from VisA_20220922.tar (2026-09-16), not
against the paper's prose. The fixtures below reproduce that file's exact shape: the header
`object,split,label,image,mask`, `.JPG` images and `.png` masks, and an empty mask field on
every normal row.
"""
import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vlmab.datasets.visa import SPLIT_CSV, VisA


def _tree(tmp_path, objects=("candle", "pcb1")):
    """A miniature VisA: two objects, the real CSV columns, real image and mask files."""
    rows = [["object", "split", "label", "image", "mask"]]
    for obj in objects:
        for kind, n in (("Normal", 3), ("Anomaly", 2)):
            (tmp_path / obj / "Data" / "Images" / kind).mkdir(parents=True, exist_ok=True)
        (tmp_path / obj / "Data" / "Masks" / "Anomaly").mkdir(parents=True, exist_ok=True)
        # 90% of normals to train, the rest plus every anomaly to test (the 1-class protocol).
        for i in range(3):
            rel = f"{obj}/Data/Images/Normal/{i:04d}.JPG"
            Image.fromarray(np.full((4, 6, 3), 10 * i, dtype=np.uint8)).save(tmp_path / rel)
            rows.append([obj, "train" if i < 2 else "test", "normal", rel, ""])
        for i in range(2):
            rel = f"{obj}/Data/Images/Anomaly/{i:03d}.JPG"
            mrel = f"{obj}/Data/Masks/Anomaly/{i:03d}.png"
            Image.fromarray(np.full((4, 6, 3), 200, dtype=np.uint8)).save(tmp_path / rel)
            mask = np.zeros((4, 6), dtype=np.uint8)
            mask[1:3, 2:4] = 255
            Image.fromarray(mask, mode="L").save(tmp_path / mrel)
            rows.append([obj, "test", "anomaly", rel, mrel])

    split_dir = tmp_path / "split_csv"
    split_dir.mkdir()
    with open(split_dir / "1cls.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return tmp_path


def test_categories_come_from_the_csv_not_the_directory_listing(tmp_path):
    """`split_csv/` and `LICENSE-DATASET` sit beside the objects at the top level, so listing
    directories would report them as categories. The CSV is the authority."""
    root = _tree(tmp_path)
    (root / "LICENSE-DATASET").write_text("cc-by-4.0")
    assert VisA(root).categories() == ["candle", "pcb1"]


def test_train_split_is_normal_only(tmp_path):
    samples = list(VisA(_tree(tmp_path)).samples("train"))
    assert len(samples) == 4                      # 2 objects x 2 train normals
    assert {s.label for s in samples} == {0}
    assert all(s.mask_path is None for s in samples)


def test_test_split_carries_both_labels_and_masks_only_on_anomalies(tmp_path):
    samples = list(VisA(_tree(tmp_path)).samples("test", category="candle"))
    assert sorted(s.label for s in samples) == [0, 1, 1]
    by_label = {s.label: [x for x in samples if x.label == s.label] for s in samples}
    assert all(s.mask_path is not None for s in by_label[1])
    assert all(s.mask_path is None for s in by_label[0])


def test_an_unknown_split_raises_rather_than_yielding_nothing(tmp_path):
    """Silently yielding nothing would make a whole category look evaluated with n=0."""
    with pytest.raises(ValueError, match="validation"):
        list(VisA(_tree(tmp_path)).samples("validation"))


def test_an_unknown_category_raises(tmp_path):
    with pytest.raises(ValueError, match="screw"):
        list(VisA(_tree(tmp_path)).samples("test", category="screw"))


def test_paths_resolve_against_the_root(tmp_path):
    sample = next(iter(VisA(_tree(tmp_path)).samples("test", category="candle")))
    assert sample.image_path.is_file()


def test_load_image_is_hwc_uint8_rgb(tmp_path):
    ds = VisA(_tree(tmp_path))
    sample = next(iter(ds.samples("train", category="candle")))
    arr = ds.load_image(sample)
    assert arr.shape == (4, 6, 3) and arr.dtype == np.uint8


def test_load_mask_is_binary_and_none_for_normals(tmp_path):
    ds = VisA(_tree(tmp_path))
    anomalous = next(s for s in ds.samples("test", category="candle") if s.label == 1)
    mask = ds.load_mask(anomalous)
    assert mask.shape == (4, 6) and set(np.unique(mask)) <= {0, 1} and mask.sum() == 4
    normal = next(s for s in ds.samples("test", category="candle") if s.label == 0)
    assert ds.load_mask(normal) is None


def test_a_missing_split_csv_names_the_file_it_wanted(tmp_path):
    (tmp_path / "candle").mkdir()
    with pytest.raises(FileNotFoundError, match="1cls.csv"):
        VisA(tmp_path).categories()


# --- Opt-in: the real archive ------------------------------------------------------------
# Skipped in CI, which has no dataset. It runs wherever VisA is on disk and is the only test
# that checks the loader against VisA's own files rather than against fixtures this file wrote.
# The counts below are the paper's stated totals (10,821 images; 9,621 normal; 1,200
# anomalous; 12 objects) and its one-class protocol ("90% normal images to train set"), so a
# pass here is agreement between three independent things: the archive, the paper, and us.

_REAL = Path(__file__).resolve().parents[1] / "data" / "visa"

real_only = pytest.mark.skipif(
    not (_REAL / SPLIT_CSV).is_file(), reason=f"VisA not present at {_REAL}"
)


@real_only
def test_real_archive_reproduces_the_papers_totals():
    ds = VisA(_REAL)
    train = list(ds.samples("train"))
    test = list(ds.samples("test"))
    assert len(ds.categories()) == 12
    assert len(train) == 8659 and {s.label for s in train} == {0}
    assert sum(1 for s in test if s.label == 0) == 962
    assert sum(1 for s in test if s.label == 1) == 1200
    assert len(train) + len(test) == 10_821
    normals = len(train) + 962
    assert normals == 9_621
    assert round(len(train) / normals, 3) == 0.900      # "90% normal images to train set"


@real_only
def test_real_archive_images_and_masks_agree():
    """Needs an object's image files, which the CSV-only extraction does not include."""
    ds = VisA(_REAL)
    test = [s for s in ds.samples("test", category="candle")]
    if not test or not test[0].image_path.is_file():
        pytest.skip("candle's image files are not extracted")
    anomalous = next(s for s in test if s.label == 1)
    image = ds.load_image(anomalous)
    mask = ds.load_mask(anomalous)
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
    assert mask.shape == image.shape[:2]
    assert set(np.unique(mask)) <= {0, 1} and mask.sum() > 0
    assert ds.load_mask(next(s for s in test if s.label == 0)) is None
