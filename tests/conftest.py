"""Fake dataset and method so the runner is testable without data, torch, or PIL."""
from pathlib import Path

import numpy as np
import pytest

from vlmab.datasets.base import AnomalyDataset, Sample
from vlmab.methods.base import AnomalyMethod, Prediction


class FakeDataset(AnomalyDataset):
    name = "fake"

    def __init__(self, categories=("alpha", "beta"), n_per_category=3):
        self._categories = list(categories)
        self._n = n_per_category

    def categories(self):
        return list(self._categories)

    def samples(self, split, category=None):
        cats = [category] if category else self._categories
        for cat in cats:
            for i in range(self._n):
                yield Sample(
                    image_path=Path(f"/fake/{cat}/{i}.png"),
                    label=i % 2,
                    category=cat,
                    meta={"split": split},
                )

    def load_image(self, sample: Sample) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def load_mask(self, sample: Sample):
        return np.zeros((8, 8), dtype=np.uint8)


class CountingMethod(AnomalyMethod):
    name = "counting"
    zero_shot = True

    def __init__(self):
        self.prepare_calls = 0
        self.seen = []

    def prepare(self, device="cuda"):
        self.prepare_calls += 1

    def predict(self, image, category):
        self.seen.append(category)
        return Prediction(
            image_score=float(len(self.seen)) / 100.0,
            anomaly_map=np.full((8, 8), 0.25, dtype=np.float32),
        )


@pytest.fixture
def fake_dataset():
    return FakeDataset()


@pytest.fixture
def counting_method():
    return CountingMethod()
