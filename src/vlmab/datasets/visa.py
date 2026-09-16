"""VisA loader — reproduction only (protocol §2), never a primary benchmark.

Layout verified 2026-09-16 by reading `VisA_20220922.tar` (1,929,840,640 bytes, the size
recorded in docs/datasets-access.md), not inferred from the paper:

    <root>/split_csv/1cls.csv          object,split,label,image,mask
    <root>/<object>/Data/Images/Normal/0000.JPG
    <root>/<object>/Data/Images/Anomaly/000.JPG
    <root>/<object>/Data/Masks/Anomaly/000.png
    <root>/<object>/image_anno.csv     per-object, not used here
    <root>/LICENSE-DATASET

`1cls.csv` is the one-class protocol: 8659 train normals, 962 test normals, 1200 test
anomalies over 12 objects = 10,821 images, which reproduces the paper's stated totals
(10,821 / 9,621 normal / 1,200 anomalous) and its "90% normal images to train set"
(8659/9621 = 90.0%). The other two files, `2cls_highshot.csv` and `2cls_fewshot.csv`, are
the two-class protocols and are deliberately not read: the bench evaluates one-class.

Image paths in the CSV are relative to the root and carry an uppercase `.JPG`; masks are
lowercase `.png`. Normal rows have an empty mask field.
"""
import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from vlmab.datasets.base import AnomalyDataset, Sample

#: The one-class protocol file. The 2-class files exist beside it and are not this study's.
SPLIT_CSV = "split_csv/1cls.csv"

#: The CSV's own vocabulary, mapped onto the Sample contract (0 = normal, 1 = anomalous).
LABELS = {"normal": 0, "anomaly": 1}


class VisA(AnomalyDataset):
    name = "visa"

    SPLITS = ("train", "test")

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._rows: list[dict[str, str]] | None = None

    def _read(self) -> list[dict[str, str]]:
        """The CSV, read once and cached. It is ~10k rows, so this stays small."""
        if self._rows is None:
            path = self.root / SPLIT_CSV
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found: VisA's splits live in {SPLIT_CSV}, which ships inside "
                    "VisA_20220922.tar. Without it the one-class protocol cannot be "
                    "reconstructed from the directory tree, because the 90/10 normal split is "
                    "recorded only in that file."
                )
            with open(path, newline="") as fh:
                self._rows = list(csv.DictReader(fh))
        return self._rows

    def categories(self) -> list[str]:
        """From the CSV, not from `iterdir()`.

        `split_csv/` and `LICENSE-DATASET` sit beside the 12 objects at the top level, so a
        directory listing would report them as categories and the runner would try to evaluate
        them.
        """
        return sorted({row["object"] for row in self._read()})

    def samples(self, split: str, category: str | None = None) -> Iterator[Sample]:
        if split not in self.SPLITS:
            raise ValueError(
                f"unknown split {split!r}; VisA's one-class protocol has only {self.SPLITS}. "
                "It has no validation split — MVTec AD 2 does, and the threshold calibration "
                "that needs one runs there, not here."
            )
        known = self.categories()
        if category is not None and category not in known:
            raise ValueError(f"unknown category {category!r}; VisA has {known}")

        for row in self._read():
            if row["split"] != split:
                continue
            if category is not None and row["object"] != category:
                continue
            mask = row["mask"].strip()
            yield Sample(
                image_path=self.root / row["image"],
                label=LABELS[row["label"]],
                category=row["object"],
                # Empty on every normal row, a real path on every anomalous one.
                mask_path=(self.root / mask) if mask else None,
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        """HxWx3 uint8. VisA is colour throughout, but `convert("RGB")` costs nothing and
        keeps this identical to the MVTec AD 2 loader's contract."""
        from PIL import Image

        with Image.open(sample.image_path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def load_mask(self, sample: Sample) -> np.ndarray | None:
        """HxW uint8 in {0, 1}, or None when the sample has no ground truth."""
        if sample.mask_path is None:
            return None
        from PIL import Image

        with Image.open(sample.mask_path) as im:
            return (np.asarray(im.convert("L")) > 0).astype(np.uint8)
