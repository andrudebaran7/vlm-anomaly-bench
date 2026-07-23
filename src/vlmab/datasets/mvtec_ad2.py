"""MVTec AD 2 loader.

Layout verified by reading the real archive, not the dataset paper (docs/datasets-access.md):

    <root>/<category>/
      train/good/            defect-free, regular lighting
      validation/good/       defect-free, regular lighting
      test_public/
        good/                normal, all 7 lighting conditions
        bad/                 anomalous, all 7 lighting conditions
        ground_truth/bad/    one `{stem}_mask.png` per bad image
      test_private/          flat, labels withheld
      test_private_mixed/    flat, labels withheld

The lighting condition lives in the filename (`000_shift_1.png`), so it is parsed onto
`Sample.meta["lighting"]` — that is what lets results be broken down by lighting condition,
which is the whole reason this dataset exists.

`test_public/good/` and `test_public/bad/` reuse the same stems, so a consumer must never
identify a sample by `image_path.stem` alone.
"""
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from vlmab.datasets.base import AnomalyDataset, Sample

#: Label for samples whose ground truth the dataset withholds (the private test splits).
#: Deliberately not 0: treating "unknown" as "normal" would silently corrupt every accuracy
#: metric, so downstream code is expected to reject this value rather than average over it.
LABEL_UNKNOWN = -1


def lighting_condition(image_path: Path) -> str:
    """Lighting condition encoded in a filename like `000_shift_1.png` -> `shift_1`."""
    _, _, condition = image_path.stem.partition("_")
    if not condition:
        raise ValueError(
            f"{image_path.name} has no lighting-condition suffix; expected "
            "'{index}_{condition}.png' as in '000_regular.png'"
        )
    return condition


class MVTecAD2(AnomalyDataset):
    name = "mvtec_ad2"

    LABELLED_SPLITS = ("train", "validation", "test_public")
    UNLABELLED_SPLITS = ("test_private", "test_private_mixed")
    SPLITS = LABELLED_SPLITS + UNLABELLED_SPLITS

    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root {self.root} does not exist")

    def categories(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def samples(self, split: str, category: str | None = None) -> Iterator[Sample]:
        if split not in self.SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {self.SPLITS}")
        for cat in [category] if category is not None else self.categories():
            base = self.root / cat / split
            if split in ("train", "validation"):
                yield from self._from_dir(base / "good", cat, split, label=0)
            elif split == "test_public":
                yield from self._from_dir(base / "good", cat, split, label=0)
                yield from self._from_dir(
                    base / "bad", cat, split, label=1,
                    ground_truth=base / "ground_truth" / "bad",
                )
            else:
                yield from self._from_dir(base, cat, split, label=LABEL_UNKNOWN)

    def _from_dir(
        self,
        directory: Path,
        category: str,
        split: str,
        label: int,
        ground_truth: Path | None = None,
    ) -> Iterator[Sample]:
        for image_path in sorted(directory.glob("*.png")):
            mask_path = None
            if ground_truth is not None:
                mask_path = ground_truth / f"{image_path.stem}_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"no ground-truth mask for {image_path}: expected {mask_path}. "
                        "An anomalous image without a mask would silently score as having "
                        "nothing to find."
                    )
            yield Sample(
                image_path=image_path,
                label=label,
                category=category,
                mask_path=mask_path,
                meta={"lighting": lighting_condition(image_path), "split": split},
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        """HxWx3 uint8.

        `convert("RGB")` replicates the single channel of a grayscale image and passes an
        already-RGB image through untouched, so this handles both without the loader having to
        know which categories are which — Vial is grayscale, the rest are unverified
        (protocol §3).
        """
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
