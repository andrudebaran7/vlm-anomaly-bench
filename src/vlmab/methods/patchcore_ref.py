"""PatchCore reference anchor (Roth et al., CVPR 2022) — the full-shot ceiling.

PatchCore's substance is the anomalib call: build a coreset memory bank from a category's
defect-free training images, then score test patches against it. That code needs anomalib and a
GPU, so it lives behind an injected backend and is written on Colab (docs/patchcore-colab-
integration.md). Everything here — the full-shot orchestration, the native-resolution map, the
predict-before-fit guard — is tested with a fake backend.

The backend is any object with:
    fit(train_images: Iterable[np.ndarray]) -> None
    score(image: np.ndarray) -> tuple[float, np.ndarray]   # (raw image score, raw anomaly map)

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution,
never per-image normalised.
"""
from typing import Iterable

import numpy as np

from vlmab.methods.base import AnomalyMethod, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class PatchCoreRef(AnomalyMethod):
    name = "patchcore_ref"
    zero_shot = False

    def __init__(self, backend=None):
        self._backend = backend
        self._fitted_category = None

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real anomalib backend
        is Colab-only (it needs anomalib and a GPU), so this reports that cleanly."""
        if self._backend is None:  # pragma: no cover - needs anomalib + GPU
            raise MethodNotRunnable(
                "PatchCoreRef needs an anomalib backend; build it in a GPU session with anomalib "
                "installed and inject it (docs/patchcore-colab-integration.md)"
            )

    def fit(self, train_images: Iterable[np.ndarray], category: str) -> None:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("PatchCoreRef has no backend; inject one or build it on GPU")
        self._backend.fit(train_images)
        self._fitted_category = category

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("PatchCoreRef has no backend; inject one or build it on GPU")
        if self._fitted_category != category:
            raise RuntimeError(
                f"predict on {category!r} but the memory bank was fitted on "
                f"{self._fitted_category!r}; the runner must fit() this category first"
            )
        raw_score, raw_map = self._backend.score(image)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(
            image_score=float(raw_score),
            anomaly_map=amap,
            extras={"fitted_category": category},
        )
