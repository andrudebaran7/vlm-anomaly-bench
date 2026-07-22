"""Common dataset contract.

Every loader yields Sample objects; the runner and metrics never touch
dataset-specific layouts. Keep this file dependency-light.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import numpy as np


@dataclass
class Sample:
    image_path: Path
    label: int                      # 0 = normal, 1 = anomalous
    category: str                   # e.g. "screw"
    mask_path: Optional[Path] = None  # pixel GT if available
    meta: dict = field(default_factory=dict)  # e.g. lighting condition (MVTec AD 2)


class AnomalyDataset:
    """Interface. Implementations: mvtec_ad2.py, real_iad.py, visa.py, mvtec_ad.py."""

    name: str = "base"

    def categories(self) -> list[str]:
        raise NotImplementedError

    def samples(self, split: str, category: str | None = None) -> Iterator[Sample]:
        raise NotImplementedError

    def load_mask(self, sample: Sample) -> np.ndarray | None:
        raise NotImplementedError

    def load_image(self, sample: Sample) -> np.ndarray:
        """Load the image as HxWx3 uint8. Kept on the dataset so decoding stays
        dataset-specific and the runner needs no image library of its own."""
        raise NotImplementedError
