"""MVTec AD (classic) loader — reproduction only (protocol §2), never a primary benchmark.

It is PatchCore's pass/fail gate: PatchCore's own paper reports 99.0 image-AUROC for
PatchCore-10%, which is exactly this repo's `coreset_sampling_ratio: 0.1` (protocol §2 v0.2.12).

Layout verified 2026-09-16 by reading `bottle.tar.xz`, not inferred from the paper:

    <root>/<category>/train/good/000.png
    <root>/<category>/test/good/000.png
    <root>/<category>/test/<defect_type>/000.png
    <root>/<category>/ground_truth/<defect_type>/000_mask.png
    <root>/<category>/license.txt, readme.txt

bottle's counts, for reference: 209 train, 20 test good, 63 test anomalous over three defect
types (broken_large 20, broken_small 22, contamination 21), and 63 masks -- one per anomalous
image and none for the normal ones, since `ground_truth/` has no `good` directory.

Two differences from MVTec AD 2 that the loader must respect:
  * there is **no validation split**, so threshold calibration cannot run here;
  * the test split is organised by defect type rather than by lighting condition, and the file
    stems restart at 000 inside every defect directory -- so a mask is found by
    `ground_truth/<defect>/<stem>_mask.png`, never by the stem alone.
"""
from pathlib import Path
from typing import Iterator

import numpy as np

from vlmab.datasets.base import AnomalyDataset, Sample

#: The subdirectory of `test/` holding normal images. Every sibling is a defect type.
NORMAL_DIR = "good"

#: Masks are the image's stem plus this suffix, inside the matching defect directory.
MASK_SUFFIX = "_mask.png"


class MVTecAD(AnomalyDataset):
    name = "mvtec_ad"

    #: No `validation`. MVTec AD 2 has one; this dataset does not, and the calibration script
    #: refuses any split but `validation`, so it can never be pointed here by accident.
    SPLITS = ("train", "test")

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def categories(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def samples(self, split: str, category: str | None = None) -> Iterator[Sample]:
        if split not in self.SPLITS:
            raise ValueError(
                f"unknown split {split!r}; MVTec AD classic has only {self.SPLITS} — it has no "
                "validation split, unlike MVTec AD 2"
            )
        known = self.categories()
        if category is not None and category not in known:
            raise ValueError(f"unknown category {category!r}; this root has {known}")

        for cat in [category] if category is not None else known:
            base = self.root / cat / split
            if split == "train":
                # train/ holds only `good`; anything else would be a layout this loader has
                # not seen, so iterate what is there rather than assuming.
                for sub in sorted(p for p in base.iterdir() if p.is_dir()):
                    yield from self._from_dir(sub, cat, label=0, mask_dir=None)
                continue

            for sub in sorted(p for p in base.iterdir() if p.is_dir()):
                if sub.name == NORMAL_DIR:
                    yield from self._from_dir(sub, cat, label=0, mask_dir=None)
                else:
                    yield from self._from_dir(
                        sub, cat, label=1,
                        mask_dir=self.root / cat / "ground_truth" / sub.name,
                    )

    def _from_dir(self, directory: Path, category: str, label: int, mask_dir: Path | None):
        for image in sorted(directory.glob("*.png")):
            mask_path = None
            if mask_dir is not None:
                mask_path = mask_dir / (image.stem + MASK_SUFFIX)
                if not mask_path.is_file():
                    # Falling back to None here would pool the image as all-normal pixels and
                    # deflate every pixel metric silently. The archive pairs them one to one.
                    raise FileNotFoundError(
                        f"{mask_path} is missing for anomalous image {image}: every anomalous "
                        "MVTec AD image has exactly one mask, so this is a broken extraction"
                    )
            yield Sample(
                image_path=image,
                label=label,
                category=category,
                mask_path=mask_path,
                # The defect type is the directory name. Kept because MVTec AD reports
                # per-defect breakdowns and dropping it would make them unrecoverable.
                meta={"defect_type": directory.name},
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        """HxWx3 uint8. Some MVTec AD categories are grayscale, so `convert("RGB")` is load
        bearing here exactly as it is for Vial (protocol §3)."""
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
